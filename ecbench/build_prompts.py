# -*- coding: utf-8 -*-
"""
把 dataset_v2 的调度实例打包成 MMGraph/verl 训练用的 parquet（v1：先跑通）。
每条样本 = 一段"排班 prompt" + 给奖励用的参照(实例结构 + 最优目标)。

v1 prompt 内容：场景 + DC 参数 + 任务表；要求模型按 <reason>/<search>/<answer> 输出，
  <answer> 里每任务一行 "<id>: drop" 或 "<id>: dc=<d> start=<t>"。
(v2 再往 prompt 里加"每小时电网信号"，并给奖励加电网成本代理)

输出:
  eccs_train.parquet   (split_tag=train: rho 0.05/0.10)
  eccs_test.parquet    (split_tag=test_id/test_ood: rho 0.20/0.30)

用法: python build_eccs_parquet.py [--limit N]
"""
import os, json, glob, argparse

SEASON = {1: "winter", 4: "spring", 7: "summer", 10: "autumn"}

SYSTEM_INSTRUCTION = (
    "You are an expert in power-system operation and cloud data-center scheduling. You have a deep "
    "command of Unit Commitment (UC), economic dispatch under a DC power-flow grid, and edge-cloud "
    "compute task scheduling, and you understand how the two couple through data-center electricity "
    "demand.\n\n"
    "YOUR TASK: jointly decide (1) the placement of compute tasks across data centers and (2) the "
    "hourly output of the controllable generators, so as to MINIMIZE total cost = grid generation "
    "cost + task drop penalties + migration cost, subject to all constraints below.\n\n"
    "DECISIONS:\n"
    "  A) Task scheduling — for each task, DROP it, or PLACE it at a data center (DC) starting at a "
    "specific hour.\n"
    "  B) Power dispatch — for each CONTROLLABLE generator, set its 24 hourly levels; each level is "
    "'off' (shut down) or a number in [0,100] (decimals allowed) meaning percent between Pmin and "
    "Pmax (0=Pmin, 100=Pmax). "
    "A SWING unit automatically absorbs the remaining imbalance, so you need not match load exactly.\n\n"
    "CONSTRAINTS:\n"
    "  - a placed task starts within [release, deadline - duration + 1];\n"
    "  - a non-migratable task runs at its origin DC; a migratable one may run at any DC;\n"
    "  - at every DC and hour: active rack-demand <= DC capacity C, and DC load (idle + task power) "
    "<= DC peak power;\n"
    "  - the generators (fixed + controllable + swing) must supply the grid background PLUS all "
    "data-center power; each task you PLACE adds its 'power' (MW) to that load during its hours, so "
    "your TASK and POWER decisions are coupled; keep each controllable generator within [Pmin, Pmax];\n"
    "  - protect higher-value tasks; shift flexible tasks to cheaper hours; prefer cheaper generators.\n"
    "  (task 'demand' = number of racks (integer); task 'power' = MW draw.)\n\n"
    "DISPATCH GUIDANCE:\n"
    "  - Make each generator's hourly output FOLLOW the residual: run generators HIGH in "
    "high-residual hours and LOW/off only in low-residual hours. NEVER leave a large generator "
    "off when that hour's residual is high — under-supplying load (lost load) is the most heavily "
    "penalized failure.\n"
    "  - Prefer to SHIFT a task to a lower-load hour within its deadline, or MIGRATE it, BEFORE "
    "dropping. Drop a task only if it truly cannot be placed feasibly.\n\n"
    "PROCEDURE (follow EXACTLY):\n"
    "  1) Reason CONCISELY inside <reason>...</reason> — at most ~150 words; leave room to answer.\n"
    "  2) On your first turn, retrieve ONCE: end the turn with <text_search>a short query</text_search> "
    "to consult similar solved cases and grid-operation knowledge.\n"
    "  3) Then output the final plan inside <answer>...</answer> with EXACTLY two sections:\n"
    "     TASKS:\n"
    "     one line per task — '<task_id>: drop' or '<task_id>: dc=<d> start=<t>', covering ALL tasks.\n"
    "     POWER:\n"
    "     one line per CONTROLLABLE generator (NOT the swing) — '<gen_uid>: v1,v2,...,v24' with all 24 "
    "hourly values, each 'off' or a number in [0,100] (decimals allowed).\n"
    "  The <answer> block with BOTH sections is mandatory; do not spend all your tokens reasoning.\n"
)


def instance_to_prompt(inst):
    dcP = inst["dc"]
    season = SEASON.get(inst["month"], "a")
    pen = round(inst["penetration"] * 100)
    header = (
        f"Instance {inst['instance_id']}\n"
        f"Scenario: {season} day (month {inst['month']}, day {inst['day']}), "
        f"{pen}% data-center penetration (DCs peak at {pen}% of system peak), "
        f"{inst['horizon']}-hour horizon.\n"
        f"Data centers: {len(inst['dc_buses'])} DCs, each with capacity C={dcP['C']} racks, "
        f"peak power {dcP['Pmax']} MW, idle power {dcP['Pidle']} MW.\n"
        f"Tasks ({inst['n_tasks']}): columns = "
        f"id | origin_dc | release | deadline | duration | demand | power | migratable | priority | drop_penalty | mig_cost\n"
    )
    rows = []
    for t in inst["tasks"]:
        rows.append(
            f"{t['task_id']} | {t['origin_dc']} | {t['release']} | {t['deadline']} | "
            f"{t['duration']} | {t['demand']} | {t['power']} | "
            f"{'yes' if t['migratable'] else 'no'} | {t['priority']} | "
            f"{t['drop_penalty']} | {t['mig_cost']}"
        )

    # ---- v2: grid signal + controllable generators (requires re-generated instances) ----
    assert "grid_signal" in inst, "instance lacks v2 fields; re-run eccs_dataset_gen_v2.py first"
    gs = inst["grid_signal"]
    n_dc = len(inst["dc_buses"])
    idle_total = round(n_dc * dcP["Pidle"], 1)   # always-on DC idle load (MW), fixed per instance
    grid_hdr = (
        "\n\nPOWER COUPLING (read carefully): generators (fixed + controllable + swing) must supply "
        f"the grid background PLUS data-center power. DC power = always-on idle (~{idle_total} MW total "
        f"across {n_dc} DCs) + the 'power' (MW) of every task you PLACE, during its running hours. "
        "Placing/shifting/dropping tasks therefore changes the load your generators must cover.\n"
        "'base_residual' = net_load + idle - fixed_gen = what controllable + swing must produce for the "
        "FIXED background; your PLACED-task power adds ON TOP of this, hour by hour:\n"
        "hour | net_load | fixed_gen | base_residual\n"
    )
    grid_rows = [
        f"{t+1} | {gs['net'][t]} | {gs['fixed_gen_agg'][t]} | "
        f"{round(gs['net'][t] + idle_total - gs['fixed_gen_agg'][t], 1)}"
        for t in range(inst["horizon"])
    ]
    gen_hdr = (
        "\nControllable generators (set the 'control' units; the 'SWING' unit auto-balances and is "
        "NOT part of your output):\n"
        "uid | bus | Pmin | Pmax | cost($/MWh) | role\n"
    )
    gen_rows = [
        f"{g['uid']} | {g['bus']} | {g['Pmin']} | {g['Pmax']} | {g['cP']} | "
        f"{'SWING' if g['is_swing'] else 'control'}"
        for g in inst["adjustable_gens"]
    ]

    return (SYSTEM_INSTRUCTION + "\n" + header + "\n".join(rows)
            + grid_hdr + "\n".join(grid_rows)
            + gen_hdr + "\n".join(gen_rows)
            + "\n\nNow produce the joint plan (TASKS and POWER).")


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset_v2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd

    inst_files = sorted(glob.glob(os.path.join(args.in_dir, "instance_*.json")))
    train_rows, test_rows = [], []
    for idx, ip in enumerate(inst_files):
        inst = json.load(open(ip, encoding="utf-8"))
        sp = ip.replace("instance_", "solution_")
        sol = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
        opt_obj = sol.get("objective") if sol.get("status") == "Optimal" else None

        # 给奖励用的参照：实例结构精简版 (tasks + dc + dc_buses) + 最优目标
        ref = {
            "instance_id": inst["instance_id"],
            "dc": inst["dc"],
            "dc_buses": inst["dc_buses"],
            "tasks": inst["tasks"],
            "optimal_obj": opt_obj,
            # ---- v2 power co-design (system-level; fast reward is self-contained, no per-bus needed) ----
            "adjustable_gens": inst["adjustable_gens"],   # 6 gen params (incl. is_swing)
            "swing_gen": inst["swing_gen"],
            "pct_step": inst["pct_step"],
            "grid_signal": inst["grid_signal"],           # net / fixed_gen_agg per hour
            "dispatch_label_pct": sol.get("adjustable_dispatch_pct"),  # optimal 6-gen dispatch (label)
            "grid_cost_full": sol.get("grid_cost_full"),  # same-口径 gap baseline
        }
        # [讲解] reward.py 读的是 reward_model['candidate_answers']，并对它做 eval() 得到一个 list。
        # [讲解]   我们把"实例 JSON 字符串"塞成这个 list 的唯一元素：candidate_answers = repr([json_str])。
        # [讲解]   repr 保证是合法 Python 字面量、eval 能原样还原（JSON 里的引号不会破坏 eval）。
        # [讲解]   score._default_compute_score 对 ECCS-RTS 会取 ground_truth[0] 反序列化回实例。
        ref_json = json.dumps(ref, ensure_ascii=False)

        row = {
            "data_source": "ECCS-RTS",
            # verl 风格：question 是一段 chat（这里放一条 user 消息）
            "question": [{"role": "user", "content": instance_to_prompt(inst)}],
            "images": None,  # 纯文本任务，无图
            # 奖励用参照（score_eccs 会用 tasks/dc 检查可行性，用 optimal_obj 算 gap）
            "reward_model": {"style": "eccs",
                             "reference": ref_json,
                             "candidate_answers": repr([ref_json])},
            "extra_info": {"index": idx, "instance_id": inst["instance_id"],
                           "split": inst.get("split_tag")},
        }
        (train_rows if inst.get("split_tag") == "train" else test_rows).append(row)
        if args.limit and (len(train_rows) + len(test_rows)) >= args.limit:
            break

    pd.DataFrame(train_rows).to_parquet("eccs_train.parquet")
    pd.DataFrame(test_rows).to_parquet("eccs_test.parquet")
    print(f"[done] train={len(train_rows)}  test={len(test_rows)}  "
          f"-> eccs_train.parquet / eccs_test.parquet")
    print("\n=== prompt 样例(前 1200 字) ===\n")
    print(train_rows[0]["question"][0]["content"][:1200])


if __name__ == "__main__":
    build()
