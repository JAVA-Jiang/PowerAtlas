# -*- coding: utf-8 -*-
"""
把 dataset_v2_pow 的最优解打包成【SFT 预热】parquet（verl SFTDataset 格式）。
每条样本 = (question=v2联合prompt字符串, response=目标answer字符串)。
loss 只在 response 上算（SFTDataset 自动 mask prompt）。

response = <reason>简短</reason> <answer>
TASKS:
<task_id>: drop | dc=<d> start=<t>   （逐任务，来自 solution.task_placement）
POWER:
<gen_uid>: v1,...,v24                （5 台可调，swing 排除；来自 solution.adjustable_dispatch_pct）
</answer>

设计：单轮 SFT（不含检索）。检索行为留给 GRPO 学。见对话记录。

用法: python build_eccs_sft.py [--in_dir dataset_v2_pow]
输出: sft_train.parquet / sft_val.parquet
"""
import os, json, glob, argparse

# 复用 v2 prompt 构造器
import importlib.util
_spec = importlib.util.spec_from_file_location("bep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_eccs_parquet.py"))
bep = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bep)
instance_to_prompt = bep.instance_to_prompt

REASON = ("Serve high-value tasks first, drop or time-shift low-value ones under congestion, "
          "and dispatch the controllable generators cheapest-first to cover the residual "
          "(the swing unit balances any remainder).")


def build_sft_target(inst, sol):
    """solution -> 目标 answer 串（TASKS + POWER，swing 排除）。"""
    # ---- TASKS：逐任务放置（按 task_id 排序）----
    tlines = []
    for p in sorted(sol["task_placement"], key=lambda x: x["task_id"]):
        if p.get("dropped"):
            tlines.append(f"{p['task_id']}: drop")
        else:
            tlines.append(f"{p['task_id']}: dc={p['dc']} start={p['start']}")
    tasks_block = "\n".join(tlines)

    # ---- POWER：只出可调（非 swing）机组的 24h 档位 ----
    ctrl = [g["uid"] for g in inst["adjustable_gens"] if not g["is_swing"]]
    lab = sol["adjustable_dispatch_pct"]
    plines = [f"{u}: " + ",".join(str(v) for v in lab[u]) for u in ctrl]
    power_block = "\n".join(plines)

    return (f"<reason>{REASON}</reason> <answer>\n"
            f"TASKS:\n{tasks_block}\n"
            f"POWER:\n{power_block}\n"
            f"</answer>")


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset_v2_pow")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd

    inst_files = sorted(glob.glob(os.path.join(args.in_dir, "instance_*.json")))
    train_rows, val_rows = [], []
    n_skip = 0
    for ip in inst_files:
        inst = json.load(open(ip, encoding="utf-8"))
        sp = ip.replace("instance_", "solution_")
        if not os.path.exists(sp):
            n_skip += 1; continue
        sol = json.load(open(sp, encoding="utf-8"))
        # 只用求到最优的实例当 SFT 标签（非最优/失败跳过，避免坏标签）
        if sol.get("status") != "Optimal":
            n_skip += 1; continue

        row = {
            "question": instance_to_prompt(inst),      # prompt（纯字符串，SFTDataset 会套 chat template）
            "response": build_sft_target(inst, sol),   # 目标 answer 串
            "instance_id": inst["instance_id"],
            "split_tag": inst.get("split_tag"),
        }
        (train_rows if inst.get("split_tag") == "train" else val_rows).append(row)
        if args.limit and (len(train_rows) + len(val_rows)) >= args.limit:
            break

    pd.DataFrame(train_rows).to_parquet("sft_train.parquet")
    pd.DataFrame(val_rows).to_parquet("sft_val.parquet")
    print(f"[done] sft_train={len(train_rows)}  sft_val={len(val_rows)}  skipped(non-optimal/no-sol)={n_skip}")
    if train_rows:
        print("\n=== response 样例(前 1000 字) ===\n")
        print(train_rows[0]["response"][:1000])


if __name__ == "__main__":
    build()
