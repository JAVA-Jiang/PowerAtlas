# -*- coding: utf-8 -*-
"""
ckpt 测试集评估 —— 在 test 集(40条)上贪心生成一次,过 score_eccs 出平均 quality + 分项。
用于:①选最佳 GRPO ckpt ②对比 SFT 基线看 GRPO 是否真更好(泛化)。
贪心解码(do_sample=False)= 模型真实会产出的方案,确定性、可复现。

用法(服务器,逐个模型跑):
  python eval_ckpt_test.py --model checkpoints/eccs-sft/qwen4b/global_step_72        --tag SFT
  python eval_ckpt_test.py --model checkpoints/eccs-v2/grpo-4b-v2/global_step_20      --tag GRPO20
  python eval_ckpt_test.py --model checkpoints/eccs-v2/grpo-4b-v2/global_step_40      --tag GRPO40
  python eval_ckpt_test.py --model checkpoints/eccs-v2/grpo-4b-v2/global_step_60      --tag GRPO60
结果同时追加到 eval_ckpt_test_results.jsonl,方便最后汇总成表。
"""
import argparse, ast, json, sys, statistics, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="模型标签,如 SFT / GRPO40")
    ap.add_argument("--parquet", default="eccs_test.parquet")
    ap.add_argument("--mmgraph_dir", default="MMGraph_Agent_code")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--batch_size", type=int, default=8, help="批量生成条数,越大越快越吃显存")
    ap.add_argument("--out", default="eval_ckpt_test_results.jsonl")
    args = ap.parse_args()

    sys.path.insert(0, args.mmgraph_dir)
    from mmgraph_r1.reward.score_eccs import compute_score_eccs

    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 批量生成需左 padding(decoder-only 才能正确对齐生成起点)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    df = pd.read_parquet(args.parquet)
    print(f"[eval] {len(df)} test instances")

    def get_prompt(q):
        return q[0]["content"]

    def get_ref(rm):
        try:
            ca = rm["candidate_answers"]
        except Exception:
            ca = rm["reward_model.candidate_answers"]
        return json.loads(ast.literal_eval(ca)[0])

    agg = {k: [] for k in ("quality", "compute_quality", "grid_quality",
                           "s_val", "s_cnt", "covered", "viol_mw", "feas_factor")}
    n_ans = n_task_feas = n_grid_feas = 0
    per_inst = []

    # 预处理:所有 prompt + ref + iid
    prompts, refs, iids = [], [], []
    for i in range(len(df)):
        row = df.iloc[i]
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": get_prompt(row["question"])}],
            add_generation_prompt=True, tokenize=False))
        ref = get_ref(row["reward_model"])
        refs.append(ref)
        iids.append(ref.get("instance_id", f"inst{i}"))

    # 批量贪心生成
    comps = []
    B = args.batch_size
    for s in range(0, len(prompts), B):
        batch = prompts[s:s + B]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, num_beams=1,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]  # 左 padding 后所有样本输入等长,可统一切片
        for r in range(gen.shape[0]):
            comps.append(tok.decode(gen[r], skip_special_tokens=True))
        print(f"  [gen] {min(s + B, len(prompts))}/{len(prompts)}", flush=True)

    # 逐条打分
    for iid, ref, comp in zip(iids, refs, comps):
        try:
            ov, q, sc, info = compute_score_eccs([comp], ref)
        except Exception as e:
            print(f"  [warn] score failed on {iid}: {e}")
            ov, q, info = -0.5, 0.0, {}

        for k in agg:
            agg[k].append(float(info.get(k, 0.0)))
        if "<answer>" in comp:
            n_ans += 1
        if info.get("feasible"):
            n_task_feas += 1
        if info.get("grid_feasible"):
            n_grid_feas += 1
        per_inst.append({"iid": iid, "quality": round(q, 4),
                         "compute_quality": round(info.get("compute_quality", 0.0), 4),
                         "grid_quality": round(info.get("grid_quality", 0.0), 4),
                         "viol_mw": round(info.get("viol_mw", 0.0), 1),
                         "grid_feasible": bool(info.get("grid_feasible", False))})

    N = len(df)
    m = {k: round(statistics.mean(v), 4) for k, v in agg.items()}
    summary = {
        "tag": args.tag, "model": args.model, "n_test": N,
        "quality": m["quality"], "compute_quality": m["compute_quality"],
        "grid_quality": m["grid_quality"],
        "s_val": m["s_val"], "s_cnt": m["s_cnt"], "covered": m["covered"],
        "viol_mw": m["viol_mw"], "feas_factor": m["feas_factor"],
        "answer_rate": round(n_ans / N, 3),
        "task_feasible_rate": round(n_task_feas / N, 3),
        "grid_feasible_rate": round(n_grid_feas / N, 3),
    }

    print("\n" + "=" * 60)
    print(f"[{args.tag}] test-set 平均 (N={N})")
    print(f"  quality          = {summary['quality']:.4f}   ← 主指标(=0.5算力+0.5电力)")
    print(f"    compute_quality= {summary['compute_quality']:.4f}")
    print(f"    grid_quality   = {summary['grid_quality']:.4f}")
    print(f"  s_val(价值满足率) = {summary['s_val']:.4f}   s_cnt(数量) = {summary['s_cnt']:.4f}")
    print(f"  covered(决策覆盖) = {summary['covered']:.4f}")
    print(f"  viol_mw(系统级)   = {summary['viol_mw']:.1f}   feas_factor = {summary['feas_factor']:.4f}")
    print(f"  出<answer>率      = {summary['answer_rate']:.3f}")
    print(f"  任务层可行率      = {summary['task_feasible_rate']:.3f}")
    print(f"  电网可行率        = {summary['grid_feasible_rate']:.3f}")
    print("=" * 60)

    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    with open(f"eval_perinst_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(per_inst, f, ensure_ascii=False, indent=1)
    print(f"[eval] 汇总已追加到 {args.out};逐实例明细 eval_perinst_{args.tag}.json")


if __name__ == "__main__":
    main()
