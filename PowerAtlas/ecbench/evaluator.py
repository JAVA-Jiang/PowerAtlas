# -*- coding: utf-8 -*-
"""
方案 b：端到端评估 LLM 的完整联合输出(放置 + 自己的电力调度)。
与 eccs_eval.py(方案 a=固定放置、Gurobi 重解电网)不同:这里 **不重新优化发电**,
而是锁死 LLM 的 placement + LLM 的 5 台可调出力,swing 作为自动平衡器,只解一个
逐母线直流潮流的"配送可行性 LP"(θ/弃风光pc/失负荷ls/过发og/线路过载ov 为 slack),
量化 LLM 自己这套方案在真实电网(逐线路限值)下的违反量、可行性、成本、对最优的 gap。

用法(本地,需 dataset_v2_pow/ + network.json + RTS 数据 + PuLP):
  python eval_endend.py --pred_dir pred_GRPO60 --data_dir dataset_v2_pow --label GRPO60
  python eval_endend.py --pred_dir pred_SFT    --data_dir dataset_v2_pow --label SFT
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, glob, json, os
import pulp
import grid_model as gen
from metrics import expand_placement, satisfaction, table_row, print_row

T = gen.T
C_LOL = gen.C_LOL


def pct_to_mw(g, v):
    if v == "off" or v is None:
        return 0.0
    return g["Pmin"] + (v / 100.0) * (g["Pmax"] - g["Pmin"])


def parse_power(power_text, ctrl_uids):
    disp = {}
    for line in (power_text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        uid, rhs = line.split(":", 1); uid = uid.strip()
        if uid not in ctrl_uids:
            continue
        vals = []
        for x in rhs.split(","):
            x = x.strip().lower()
            if x == "off":
                vals.append("off")
            else:
                try:
                    vals.append(max(0.0, min(100.0, float(x))))
                except Exception:
                    vals.append(None)
        disp[uid] = vals
    return disp


def gen_cost(g, mw_arr):
    total, on_prev = 0.0, False
    for mw in mw_arr:
        on = mw > 1e-6
        total += g["cP"] * mw
        if on:
            total += g.get("cNL", 0.0)
            if not on_prev:
                total += g.get("cSU", 0.0)
        on_prev = on
    return total


def eval_instance(inst, sol, placement, power_text, NET, time_limit=30):
    month, day = inst["month"], inst["day"]
    D, Pre, _ = gen.build_grid(month, day)
    dcP = inst["dc"]; Pidle = dcP["Pidle"]
    dc_buses = {int(k): int(v) for k, v in inst["dc_buses"].items()}
    adj = inst["adjustable_gens"]; gmap = {g["uid"]: g for g in adj}
    swing = gmap[inst["swing_gen"]]
    ctrl = [g for g in adj if not g["is_swing"]]
    bus_fixed = {int(b): arr for b, arr in inst["bus_fixed_gen"].items()}
    BUSES = NET["buses"]; REF = NET["ref_bus"]; LINES = NET["lines"]

    # 逐母线 DC 负载(含常开 idle + 放置任务功率)
    tasks = {t["task_id"]: t for t in inst["tasks"]}
    dc_load = {}
    for j, busn in dc_buses.items():
        for t in range(1, T + 1):
            dc_load[(busn, t)] = dc_load.get((busn, t), 0.0) + Pidle
    for pl in placement:
        if pl.get("dropped"):
            continue
        tk = tasks.get(pl["task_id"])
        busn = dc_buses.get(pl.get("dc"))
        if tk is None or busn is None:
            continue
        for t in range(pl["start"], pl["start"] + tk["duration"]):
            if 1 <= t <= T:
                dc_load[(busn, t)] = dc_load.get((busn, t), 0.0) + tk["power"]

    # LLM 可调机组 MW
    ctrl_uids = {g["uid"] for g in ctrl}
    disp = parse_power(power_text, ctrl_uids)
    ctrl_mw = {g["uid"]: [pct_to_mw(g, (disp.get(g["uid"], []) + [None] * T)[t]) for t in range(T)] for g in ctrl}

    # 逐时潮流可行性 LP
    ls_tot = ov_tot = og_tot = 0.0
    ls_hour = [0.0] * T
    swing_mw = [0.0] * T
    for ti in range(T):
        t = ti + 1
        m = pulp.LpProblem(f"flow_{t}", pulp.LpMinimize)
        th = {n: pulp.LpVariable(f"th_{n}") for n in BUSES}
        fl = {L["idx"]: pulp.LpVariable(f"f_{L['idx']}") for L in LINES}
        ov = {L["idx"]: pulp.LpVariable(f"ov_{L['idx']}", lowBound=0) for L in LINES}
        pc = {n: pulp.LpVariable(f"pc_{n}", lowBound=0) for n in BUSES}
        ls = {n: pulp.LpVariable(f"ls_{n}", lowBound=0) for n in BUSES}
        og = {n: pulp.LpVariable(f"og_{n}", lowBound=0) for n in BUSES}
        sw = pulp.LpVariable("sw", lowBound=0, upBound=swing["Pmax"])
        m += pulp.lpSum(ov.values()) + pulp.lpSum(ls.values()) + pulp.lpSum(og.values())
        for n in BUSES:
            m += pc[n] <= Pre[(n, t)]
        lf = {n: [] for n in BUSES}; lt = {n: [] for n in BUSES}
        for L in LINES:
            a, b, x = L["frm"], L["to"], L["x"]
            m += fl[L["idx"]] == (1.0 / x) * (th[a] - th[b])
            if L["limit"] is not None:
                m += fl[L["idx"]] <= L["limit"] + ov[L["idx"]]
                m += fl[L["idx"]] >= -L["limit"] - ov[L["idx"]]
            lf[a].append(L["idx"]); lt[b].append(L["idx"])
        m += th[REF] == 0
        for n in BUSES:
            gfix = bus_fixed.get(n, [0.0] * T)[ti]
            gctrl = sum(ctrl_mw[g["uid"]][ti] for g in ctrl if g["bus"] == n)
            gsw = sw if swing["bus"] == n else 0.0
            m += (gfix + gctrl + gsw + (Pre[(n, t)] - pc[n]) + ls[n] - og[n]
                  - D[(n, t)] - dc_load.get((n, t), 0.0)
                  == pulp.lpSum(fl[li] for li in lf[n]) - pulp.lpSum(fl[li] for li in lt[n]))
        m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        v = lambda x: (0.0 if pulp.value(x) is None else pulp.value(x))
        ls_h = sum(v(ls[n]) for n in BUSES)
        ls_tot += ls_h
        ls_hour[ti] = ls_h
        ov_tot += sum(v(ov[L["idx"]]) for L in LINES)
        og_tot += sum(v(og[n]) for n in BUSES)
        swing_mw[ti] = v(sw)

    viol_mw = ls_tot + ov_tot + og_tot

    # 成本 & gap(同口径:fixed67 相同,差异只在 6 台可调 + LLM 的失负荷)
    cost_llm6 = sum(gen_cost(gmap[g["uid"]], ctrl_mw[g["uid"]]) for g in ctrl) + gen_cost(swing, swing_mw)
    lab = sol.get("adjustable_dispatch_pct") or {}
    cost_opt6 = sum(gen_cost(gmap[u], [pct_to_mw(gmap[u], x) for x in lab[u]]) for u in lab if u in gmap)
    grid_full = sol["grid_cost_full"]
    fixed67 = grid_full - cost_opt6
    llm_grid = fixed67 + cost_llm6

    _, viols_task, viol_task_mw = expand_placement(inst, placement)
    task_feasible = (len(viols_task) == 0)
    drop_sum = sum(tasks[pl["task_id"]]["drop_penalty"] for pl in placement if pl.get("dropped"))
    mig_sum = sum(tasks[pl["task_id"]]["mig_cost"] for pl in placement
                  if (not pl.get("dropped")) and pl.get("dc") != tasks[pl["task_id"]]["origin_dc"])
    # 统一总成本口径:失负荷 ls + 线路过载 ov + 过发 og 全部按 VOLL(C_LOL) 折成惩罚成本,
    # 使 Feas=0 的方案也有一个可比的"总成本"(电网违反越大 → 成本越高)。
    llm_obj = llm_grid + C_LOL * (ls_tot + ov_tot + og_tot) + drop_sum + mig_sum
    oracle_obj = sol["objective"]

    feasible = bool(task_feasible and viol_mw <= 1.0)
    res = {"feasible": feasible, "task_feasible": task_feasible,
           "viol_mw": round(viol_mw + viol_task_mw, 2),
           "viol_line": round(ov_tot, 1), "viol_ls": round(ls_tot, 1), "viol_og": round(og_tot, 1),
           "grid": round(llm_grid, 1), "objective": round(llm_obj, 1),
           "gap_pct": round(100.0 * (llm_obj - oracle_obj) / oracle_obj, 3) if oracle_obj else None,
           "drop": round(drop_sum, 1)}
    # 电算耦合审核:失负荷>1MW 的小时视为"缺电",这些小时里运行的任务算力上不算满足
    unpowered = {ti + 1 for ti in range(T) if ls_hour[ti] > 1.0}
    # 任务侧硬约束违反(window/deadline/迁移/DC容量/功率)的任务,也不算满足
    illegal = {v["task_id"] for v in viols_task if "task_id" in v}
    res.update(satisfaction(inst, placement, unpowered, illegal))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--data_dir", default="dataset_v2_pow")
    ap.add_argument("--label", default="method")
    ap.add_argument("--tpq", type=float, default=None)
    ap.add_argument("--p95", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time_limit", type=int, default=30)
    args = ap.parse_args()

    NET = json.load(open(os.path.join(args.data_dir, "network.json"), encoding="utf-8"))
    insts, sols = {}, {}
    for f in glob.glob(os.path.join(args.data_dir, "instance_*.json")):
        d = json.load(open(f, encoding="utf-8")); insts[d["instance_id"]] = d
    for f in glob.glob(os.path.join(args.data_dir, "solution_*.json")):
        s = json.load(open(f, encoding="utf-8")); sols[s["instance_id"]] = s

    rows = []; oracles = []
    for pf in sorted(glob.glob(os.path.join(args.pred_dir, "*.json"))):
        p = json.load(open(pf, encoding="utf-8"))
        iid = p.get("instance_id"); inst = insts.get(iid); sol = sols.get(iid)
        if inst is None or sol is None or sol.get("status") != "Optimal":
            continue
        r = eval_instance(inst, sol, p["task_placement"], p.get("power_text", ""), NET, args.time_limit)
        rows.append(r); oracles.append(sol["objective"])
        print(f"  {iid}: feas={r['feasible']} gap%={r['gap_pct']} viol_mw={r['viol_mw']} "
              f"(line={r['viol_line']} ls={r['viol_ls']} og={r['viol_og']}) S-cnt%={r['s_cnt']}", flush=True)
        if args.limit and len(rows) >= args.limit:
            break

    row = table_row(rows, tpq=args.tpq, p95=args.p95, label=args.label)
    print_row(row)

    # 全体实例(含不可行)的统一总成本 = grid + VOLL*(ls+ov+og) + drop + mig
    import statistics as st
    tc = st.mean([r["objective"] for r in rows]) if rows else float("nan")
    orc = st.mean(oracles) if oracles else float("nan")
    print(f"  TotalCost$ (all-instance mean, n={len(rows)}) = {tc:,.0f}")
    print(f"  Oracle$    (all-instance mean)               = {orc:,.0f}")
    print(f"  Ratio vs oracle                              = {tc/orc:.2f}x" if orc else "")


if __name__ == "__main__":
    main()
