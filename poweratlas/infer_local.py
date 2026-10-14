# -*- coding: utf-8 -*-
"""
在 test 集上跑一个模型,把解析出的 task_placement 存成 eccs_eval.py --eval_dir 能吃的格式。
每条 test 实例 -> pred_dir/<instance_id>.json = {instance_id, task_placement:[...], power_text}
同时记录逐条生成耗时 -> 打印 TPQ(均值) 与 p95,喂给 eccs_eval 的 --tpq/--p95。

用法(服务器 GPU):
  python gen_placements.py --model checkpoints/eccs-v2/grpo-4b-v2-hf/global_step_60 \
      --out_dir pred_step60
然后把 pred_step60/ 传回本地,本地跑:
  python eccs_eval.py --eval_dir pred_step60 --data_dir dataset_v2_pow --label GRPO60 --tpq X --p95 Y
"""
import argparse, ast, json, os, sys, time, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parquet", default="/root/autodl-tmp/eccs_test.parquet")
    ap.add_argument("--mmgraph_dir", default=".")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条(调试用),默认全部")
    args = ap.parse_args()

    sys.path.insert(0, args.mmgraph_dir)
    from mmgraph_r1.reward.score_eccs import (extract_answer_block, split_answer_sections, parse_schedule)

    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[gen] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    df = pd.read_parquet(args.parquet)
    n = len(df) if args.limit is None else min(len(df), args.limit)
    print(f"[gen] {len(df)} test instances; running {n}")

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
        placement = parse_schedule(tasks_text or ans or "")   # {tid: 'drop' | (dc, start)}

        tp = []
        for tid, dec in placement.items():
            if tid not in valid:
                continue
            if dec == "drop":
                tp.append({"task_id": tid, "dropped": True})
            else:
                dc, start = dec
                tp.append({"task_id": tid, "dc": dc, "start": start})
        # 未被模型给出决策的任务 -> 记为 dropped(与满足率口径一致,避免漏算)
        emitted = {d["task_id"] for d in tp}
        for tid in valid:
            if tid not in emitted:
                tp.append({"task_id": tid, "dropped": True})

        json.dump({"instance_id": iid, "task_placement": tp, "power_text": power_text},
                  open(os.path.join(args.out_dir, f"{iid}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"  [{i+1}/{n}] {iid}  placed={sum(1 for d in tp if not d.get('dropped'))}/{len(valid)}  {dt:.1f}s", flush=True)

    ts = sorted(times)
    tpq = statistics.mean(ts)
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    print("\n" + "=" * 50)
    print(f"[gen] wrote {len(times)} placements -> {args.out_dir}/")
    print(f"[gen] TPQ(mean) = {tpq:.2f}s    p95 = {p95:.2f}s")
    print(f"[gen] 本地评估: python eccs_eval.py --eval_dir {args.out_dir} --data_dir dataset_v2_pow "
          f"--label {os.path.basename(args.model)} --tpq {tpq:.2f} --p95 {p95:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
