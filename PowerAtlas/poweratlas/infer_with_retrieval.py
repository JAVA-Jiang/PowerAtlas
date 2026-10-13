# -*- coding: utf-8 -*-
"""
消融实验（路线 B）：测试时把检索到的语料直接注入 prompt，评估"知识先验"本身的价值。

与 gen_placements.py 的唯一区别：生成前，用 TF-IDF 余弦从 eccs_corpus_full_v2.jsonl 取
top-k 条最相似的语料（训练划分构造，无泄漏），以 in-context 参考块的形式插进 prompt。
其余（模型、解码、输出格式、计时）完全一致，因此 pred 目录可以和 gen_placements.py 的
输出用同一个 eval_endend.py 口径直接对比。

用法（服务器 GPU）：
  python gen_placements_rag.py \
      --model checkpoints/eccs-v2/grpo-falcon3b-e16-hf/global_step_40 \
      --corpus /root/autodl-tmp/eccs_corpus_full_v2.jsonl \
      --out_dir pred_falcon_grpo40_rag --topk 2

对照组直接用已有的 pred_falcon_e16_grpo40/（同一 checkpoint，无注入）。

依赖：只用 numpy + transformers，不需要 sklearn / FAISS / 检索服务。
"""
import argparse, ast, json, math, os, re, sys, time, statistics
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return _TOKEN.findall(s.lower())


class TfidfIndex:
    """依赖无关的 TF-IDF 余弦检索（语料仅 551 条，纯 Python 足够快）。"""

    def __init__(self, docs):
        self.docs = docs
        self.df = Counter()
        self.tf = []
        for d in docs:
            c = Counter(tokenize(d))
            self.tf.append(c)
            for w in c:
                self.df[w] += 1
        self.N = len(docs)
        self.vecs = [self._vec(c) for c in self.tf]

    def _idf(self, w):
        return math.log((self.N + 1) / (self.df.get(w, 0) + 1)) + 1.0

    def _vec(self, c):
        v = {w: (1.0 + math.log(n)) * self._idf(w) for w, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def topk(self, query, k):
        q = self._vec(Counter(tokenize(query)))
        scored = []
        for i, v in enumerate(self.vecs):
            if len(q) < len(v):
                s = sum(x * v.get(w, 0.0) for w, x in q.items())
            else:
                s = sum(x * q.get(w, 0.0) for w, x in v.items())
            scored.append((s, i))
        scored.sort(reverse=True)
        return [(i, s) for s, i in scored[:k]]


ANCHOR = "\n\nNow produce the joint plan (TASKS and POWER)."


def inject(prompt, snippets):
    block = [
        "\n\nRETRIEVED REFERENCE CASES AND KNOWLEDGE (retrieved from solved training instances "
        "and expert knowledge; use them as priors, do not copy them blindly):"
    ]
    for j, s in enumerate(snippets, 1):
        block.append(f"  [{j}] {s}")
    blob = "\n".join(block)
    if ANCHOR in prompt:
        return prompt.replace(ANCHOR, blob + ANCHOR)
    return prompt + blob + ANCHOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--corpus", default="/root/autodl-tmp/eccs_corpus_full_v2.jsonl")
    ap.add_argument("--parquet", default="/root/autodl-tmp/eccs_test.parquet")
    ap.add_argument("--mmgraph_dir", default=".")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=900, help="每条语料截断长度，防 prompt 爆长")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump_prompt", default=None, help="把第 1 条注入后的 prompt 存文件供人工检查")
    args = ap.parse_args()

    sys.path.insert(0, args.mmgraph_dir)
    from mmgraph_r1.reward.score_eccs import (extract_answer_block, split_answer_sections, parse_schedule)

    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    docs = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line)["corpus_text"])
    print(f"[rag] corpus: {len(docs)} entries from {args.corpus}")
    index = TfidfIndex(docs)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[gen] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    df = pd.read_parquet(args.parquet)
    n = len(df) if args.limit is None else min(len(df), args.limit)
    print(f"[gen] {len(df)} test instances; running {n} with top-{args.topk} injection")

    def get_ref(rm):
        try: ca = rm["candidate_answers"]
        except Exception: ca = rm["reward_model.candidate_answers"]
        return json.loads(ast.literal_eval(ca)[0])

    times = []
    for i in range(n):
        row = df.iloc[i]
        prompt = row["question"][0]["content"]
        ref = get_ref(row["reward_model"])
        iid = ref["instance_id"]
        valid = {t["task_id"] for t in ref["tasks"]}

        hits = index.topk(prompt, args.topk)
        snippets = [docs[j][:args.max_chars] for j, _ in hits]
        prompt = inject(prompt, snippets)
        if i == 0 and args.dump_prompt:
            open(args.dump_prompt, "w", encoding="utf-8").write(prompt)
            print(f"[rag] dumped injected prompt -> {args.dump_prompt}")
        print(f"  [rag] {iid} hits=" + ",".join(f"{j}({s:.3f})" for j, s in hits), flush=True)

        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(chat, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, num_beams=1,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
        times.append(dt)
        comp = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        ans = extract_answer_block(comp)
        tasks_text, power_text = split_answer_sections(ans)
        placement = parse_schedule(tasks_text or ans or "")

        tp = []
        for tid, dec in placement.items():
            if tid not in valid:
                continue
            if dec == "drop":
                tp.append({"task_id": tid, "dropped": True})
            else:
                dc, start = dec
                tp.append({"task_id": tid, "dc": dc, "start": start})
        emitted = {d["task_id"] for d in tp}
        for tid in valid:
            if tid not in emitted:
                tp.append({"task_id": tid, "dropped": True})

        json.dump({"instance_id": iid, "task_placement": tp, "power_text": power_text},
                  open(os.path.join(args.out_dir, f"{iid}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"  [{i+1}/{n}] {iid}  placed={sum(1 for d in tp if not d.get('dropped'))}/{len(valid)}  {dt:.1f}s",
              flush=True)

    ts = sorted(times)
    tpq = statistics.mean(ts)
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    print("\n" + "=" * 50)
    print(f"[gen] wrote {len(times)} placements -> {args.out_dir}/")
    print(f"[gen] TPQ(mean) = {tpq:.2f}s    p95 = {p95:.2f}s")
    print(f"[gen] 本地评估: python eval_endend.py --pred_dir {args.out_dir} "
          f"--data_dir dataset_v3_small --label +Retrieval")
    print("=" * 50)


if __name__ == "__main__":
    main()
