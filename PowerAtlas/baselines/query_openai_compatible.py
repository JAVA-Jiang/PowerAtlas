# -*- coding: utf-8 -*-
"""
用【API 前沿模型】(DeepSeek / Kimi / Qwen / ... OpenAI 兼容)对 test 集做 zero-shot 联合调度,
输出与 gen_placements.py 完全相同的 pred_dir 格式,喂给 eval_endend.py 出 Feas/Gap/Viol 填表1。

与本地 gen_placements.py 的两点区别:
  1) 用 OpenAI SDK 调 API,不需要 GPU;
  2) prompt 做 zero-shot 化 —— 删掉多轮 agent 的 <text_search> 检索步骤,改成单轮直接出 <answer>
     (前沿模型是单轮 API,没有检索工具,不改会输出一个 text_search 就空转)。

装依赖(本地):  pip install openai pandas pyarrow

用法(DeepSeek 推理版, 先测 10 条):
  export DEEPSEEK_API_KEY=sk-xxxxx
  python gen_placements_api.py --model deepseek-reasoner --out_dir pred_deepseek_r1 \
      --base_url https://api.deepseek.com --api_key $DEEPSEEK_API_KEY --limit 10
  python eval_endend.py --pred_dir pred_deepseek_r1 --data_dir dataset_v3_small --label DeepSeek-R1

换模型只改三处(base_url / model / key),例:
  Kimi : --base_url https://api.moonshot.cn/v1  --model kimi-k2-0711-preview
  Qwen : --base_url https://dashscope.aliyuncs.com/compatible-mode/v1  --model qwen-max
"""
import argparse, ast, json, os, re, sys, time, statistics


def to_zero_shot(prompt: str) -> str:
    """删掉检索步骤(step 2)、把 step 3 重编号为 2,并在结尾强调单轮直接作答。"""
    prompt = re.sub(r"\n\s*2\) On your first turn, retrieve ONCE:.*?grid-operation knowledge\.",
                    "", prompt, flags=re.DOTALL)
    prompt = prompt.replace("3) Then output the final plan", "2) Then output the final plan")
    prompt += ("\n\nRespond in a SINGLE turn: first <reason>...</reason> (concise), then the COMPLETE "
               "<answer>...</answer> with both TASKS: and POWER: sections. Do NOT request any tool, "
               "search, or extra turn — output the full plan now.")
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型名, 如 deepseek-reasoner / deepseek-chat")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY"))
    ap.add_argument("--parquet", default="eccs_test.parquet", help="本地 test parquet(含 question+reward_model)")
    ap.add_argument("--mmgraph_dir", default="MMGraph_Agent_code", help="含 mmgraph_r1 的目录(import 解析函数)")
    ap.add_argument("--max_tokens", type=int, default=8000, help="推理模型输出长,给足(reasoning+answer)")
    ap.add_argument("--temperature", type=float, default=0.0, help="推理模型(如 deepseek-reasoner)会忽略此项")
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 条(先 10 条试)")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[api] 缺少 API key: 设 --api_key 或环境变量 DEEPSEEK_API_KEY / API_KEY")

    sys.path.insert(0, args.mmgraph_dir)
    from mmgraph_r1.reward.score_eccs import (extract_answer_block, split_answer_sections, parse_schedule)
    import pandas as pd
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    n = len(df) if args.limit is None else min(len(df), args.limit)
    print(f"[api] model={args.model}  base_url={args.base_url}  running {n}/{len(df)}")

    def get_ref(rm):
        try: ca = rm["candidate_answers"]
        except Exception: ca = rm["reward_model.candidate_answers"]
        return json.loads(ast.literal_eval(ca)[0])

    def call(prompt):
        for a in range(args.retries):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                return (r.choices[0].message.content or "")
            except Exception as e:
                print(f"    [retry {a+1}/{args.retries}] {e}", flush=True)
                time.sleep(6)
        return ""

    times = []
    for i in range(n):
        row = df.iloc[i]
        prompt = to_zero_shot(row["question"][0]["content"])
        ref = get_ref(row["reward_model"]); iid = ref["instance_id"]
        valid = {t["task_id"] for t in ref["tasks"]}

        t0 = time.time()
        comp = call(prompt)
        dt = time.time() - t0
        times.append(dt)

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
        placed = sum(1 for d in tp if not d.get("dropped"))
        got = "OK" if ans else "NO-<answer>"
        print(f"  [{i+1}/{n}] {iid}  placed={placed}/{len(valid)}  pow={'y' if power_text else 'n'}  {got}  {dt:.1f}s", flush=True)

    ts = sorted(times)
    tpq = statistics.mean(ts) if ts else 0.0
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))] if ts else 0.0
    print("\n" + "=" * 54)
    print(f"[api] wrote {len(times)} placements -> {args.out_dir}/")
    print(f"[api] TPQ(mean)={tpq:.2f}s   p95={p95:.2f}s")
    print(f"[api] 评估: python eval_endend.py --pred_dir {args.out_dir} --data_dir dataset_v3_small "
          f"--label {args.model} --tpq {tpq:.2f} --p95 {p95:.2f}")
    print("=" * 54)


if __name__ == "__main__":
    main()
