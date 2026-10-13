# -*- coding: utf-8 -*-
"""
ECCS 解评估器 (independent evaluator)

给定一个实例 + 一个候选解 (task_placement)，计算 Table-1 需要的底层指标：
    - task_feasible      任务层是否可行 (容量 C / 功率 Pmax / deadline 窗口 / 可迁移性)
    - viol_mw            约束违反 (兆瓦)：任务层硬违反量 + 电网层失负荷 ls
    - grid_cost          固定 placement 后重解电网 (UC+DCOPF) 的发电成本
    - objective          候选解总成本 = grid + drop + migration + C_LOL*ls
    - gap_pct            相对最优 gap (需传入最优目标值)
    - feasible           task_feasible 且电网层 ls≈0

电网模型直接复用生成器 eccs_dataset_gen_v2 的静态数据与 build_grid，
保证与数据集生成时口径完全一致。评估 = 把任务变量 s 固定成候选解，只解剩下的电网层。

用法:
    python eccs_eval.py --self_check --limit 30      # 拿 Gurobi 解自检 (应: 全可行, gap≈0, viol≈0)
    python eccs_eval.py --gurobi_rows                # 算 Gurobi optimal 参照行 (TPQ/p95 来自现有 solve_time)
    python eccs_eval.py --resolve_unsolved --time_limit 600   # 加时限重解 25 个 Not-Solved
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys, json, glob, time, argparse, statistics as st

import pulp
import grid_model as gen   # 复用电网静态数据 + build_grid (导入即加载 RTS CSV)

T      = gen.T
C_LOL  = gen.C_LOL
R_FRAC = gen.R_FRAC
gens   = gen.gens
lines  = gen.lines
buses  = gen.buses
ref_bus = gen.ref_bus
dc_buses = gen.dc_buses

TOL = 1e-6


# ----------------------------------------------------------------------------
# 候选解 -> 每 (DC, 时段) 的占用 (z 展开)
# ----------------------------------------------------------------------------
def expand_placement(inst, placement):
    """返回 (load_jt, viols_task)。
    load_jt[(j,t)] = dict(demand=.., power=..)  仅含被占用的格子。
    viols_task: 任务层硬违反列表。"""
    tasks = {t["task_id"]: t for t in inst["tasks"]}
    dcP   = inst["dc"]
    Pidle, Pmax_dc, C = dcP["Pidle"], dcP["Pmax"], dcP["C"]

    load = {}   # (j,t) -> [demand_sum, power_sum]
    viols = []
    for pl in placement:
        if pl.get("dropped"):
            continue
        tk = tasks[pl["task_id"]]
        j, start = pl["dc"], pl["start"]
        dur = tk["duration"]

        # deadline / release 窗口
        if not (tk["release"] <= start <= tk["deadline"] - dur + 1):
            viols.append({"type": "window", "task_id": tk["task_id"],
                          "start": start, "release": tk["release"],
                          "deadline": tk["deadline"], "duration": dur})
        # 可迁移性
        if not tk["migratable"] and j != tk["origin_dc"]:
            viols.append({"type": "migratability", "task_id": tk["task_id"],
                          "dc": j, "origin_dc": tk["origin_dc"]})
        if j not in (1, 2, 3, 4):
            viols.append({"type": "bad_dc", "task_id": tk["task_id"], "dc": j})

        for t in range(start, start + dur):
            if t < 1 or t > T:
                continue
            cell = load.setdefault((j, t), [0.0, 0.0])
            cell[0] += tk["demand"]
            cell[1] += tk["power"]

    # 容量 C 与功率 Pmax 违反 (折算成 MW 量级)
    viol_mw_task = 0.0
    for (j, t), (dem, pw) in load.items():
        if dem > C + TOL:
            viols.append({"type": "capacity", "dc": j, "t": t,
                          "demand": dem, "C": C})
            viol_mw_task += (dem - C) * dcP["pi"]   # 超出机架 -> 折算功率
        if Pidle + pw > Pmax_dc + TOL:
            viols.append({"type": "dc_power", "dc": j, "t": t,
                          "load": Pidle + pw, "Pmax": Pmax_dc})
            viol_mw_task += (Pidle + pw - Pmax_dc)
    return load, viols, viol_mw_task


# ----------------------------------------------------------------------------
# 固定 placement 后重解电网 (UC + DCOPF)
# ----------------------------------------------------------------------------
def make_solver(time_limit, gap=0.02):
    try:
        s = pulp.GUROBI(msg=0, timeLimit=time_limit, gapRel=gap)
        if s.available():
            return s
    except Exception:
        pass
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap, threads=4)


def grid_solve(inst, load_jt, time_limit=60):
    """L(j,t) 固定为 load_jt 给出的常量负载，解发电+潮流，返回 (grid_cost, lost_load_mw, status)。"""
    month, day = inst["month"], inst["day"]
    D, Pre, _ = gen.build_grid(month, day)
    dcP = inst["dc"]
    Pidle = dcP["Pidle"]

    # 每 (j,t) 的固定 DC 负载 = Pidle + 占用功率 (没占用也有 Pidle 空载)
    Lconst = {}
    for j in (1, 2, 3, 4):
        for t in range(1, T + 1):
            pw = load_jt.get((j, t), [0.0, 0.0])[1]
            Lconst[(j, t)] = Pidle + pw

    TT = range(1, T + 1)
    m = pulp.LpProblem("ECCS_grid", pulp.LpMinimize)
    u  = {(g, t): pulp.LpVariable(f"u_{g}_{t}", cat="Binary") for g in gens for t in TT}
    vv = {(g, t): pulp.LpVariable(f"v_{g}_{t}", cat="Binary") for g in gens for t in TT}
    w  = {(g, t): pulp.LpVariable(f"w_{g}_{t}", cat="Binary") for g in gens for t in TT}
    P  = {(g, t): pulp.LpVariable(f"P_{g}_{t}", lowBound=0) for g in gens for t in TT}
    th = {(n, t): pulp.LpVariable(f"th_{n}_{t}") for n in buses for t in TT}
    fl = {(li, t): pulp.LpVariable(f"f_{li}_{t}") for li in range(len(lines)) for t in TT}
    pc = {(n, t): pulp.LpVariable(f"pc_{n}_{t}", lowBound=0) for n in buses for t in TT}
    ls = {(n, t): pulp.LpVariable(f"ls_{n}_{t}", lowBound=0) for n in buses for t in TT}

    m += (pulp.lpSum(gens[g]["cP"] * P[(g, t)] + gens[g]["cNL"] * u[(g, t)]
                     + gens[g]["cSU"] * vv[(g, t)] for g in gens for t in TT)
          + pulp.lpSum(C_LOL * ls[(n, t)] for n in buses for t in TT))

    for g in gens:
        G = gens[g]
        for t in TT:
            up = u[(g, t - 1)] if t > 1 else 0
            Pp = P[(g, t - 1)] if t > 1 else 0
            m += vv[(g, t)] - w[(g, t)] == u[(g, t)] - up
            m += vv[(g, t)] + w[(g, t)] <= 1
            m += P[(g, t)] >= G["Pmin"] * u[(g, t)]
            m += P[(g, t)] <= G["Pmax"] * u[(g, t)]
            m += P[(g, t)] - Pp <= G["RU"] * up + G["SU"] * vv[(g, t)]
            m += Pp - P[(g, t)] <= G["RD"] * u[(g, t)] + G["SD"] * w[(g, t)]
        for t in TT:
            if t >= G["UT"]:
                m += pulp.lpSum(vv[(g, tau)] for tau in range(t - G["UT"] + 1, t + 1)) <= u[(g, t)]
            if t >= G["DT"]:
                m += pulp.lpSum(w[(g, tau)] for tau in range(t - G["DT"] + 1, t + 1)) <= 1 - u[(g, t)]

    for t in TT:
        m += pulp.lpSum(gens[g]["Pmax"] * u[(g, t)] - P[(g, t)] for g in gens) >= R_FRAC * sum(D[(b, t)] for b in buses)
    for n in buses:
        for t in TT:
            m += pc[(n, t)] <= Pre[(n, t)]
    for li, (a, b, x, F) in enumerate(lines):
        for t in TT:
            m += fl[(li, t)] == (1.0 / x) * (th[(a, t)] - th[(b, t)])
            m += fl[(li, t)] <= F
            m += fl[(li, t)] >= -F
    for t in TT:
        m += th[(ref_bus, t)] == 0

    lf = {n: [] for n in buses}
    lt = {n: [] for n in buses}
    for li, (a, b, x, F) in enumerate(lines):
        lf[a].append(li)
        lt[b].append(li)
    for n in buses:
        for t in TT:
            gen_n = pulp.lpSum(P[(g, t)] for g in gens if gens[g]["bus"] == n)
            dc_n = pulp.lpSum(Lconst[(j, t)] for j in (1, 2, 3, 4) if dc_buses[j] == n)
            m += (gen_n + (Pre[(n, t)] - pc[(n, t)]) + ls[(n, t)] - D[(n, t)] - dc_n
                  == pulp.lpSum(fl[(li, t)] for li in lf[n]) - pulp.lpSum(fl[(li, t)] for li in lt[n]))

    solver = make_solver(time_limit)
    m.solve(solver)
    val = lambda x: (0.0 if pulp.value(x) is None else pulp.value(x))
    grid_cost = sum(gens[g]["cP"] * val(P[(g, t)]) + gens[g]["cNL"] * val(u[(g, t)])
                    + gens[g]["cSU"] * val(vv[(g, t)]) for g in gens for t in TT)
    lost_load = sum(val(ls[(n, t)]) for n in buses for t in TT)
    return grid_cost, lost_load, pulp.LpStatus[m.status]


# ----------------------------------------------------------------------------
# 算力满足指标（纯算术，无需解电网）
# ----------------------------------------------------------------------------
def satisfaction(inst, placement, unpowered_hours=None, illegal_tasks=None):
    """算力满足指标。一个任务只有【同时满足】以下三条才算"真满足":
      ① 非 drop;
      ② 放置合法(不在 illegal_tasks 里 —— 即不违反 window/deadline/迁移/DC容量/功率);
      ③ 运行时段电网供得上电(不落在 unpowered_hours 缺电小时里)。
    两个集合都为 None 时退化为旧口径(只看放置,不看约束/电网)。"""
    tasks = inst["tasks"]; tmap = {t["task_id"]: t for t in tasks}
    up = unpowered_hours or set()
    bad = illegal_tasks or set()
    def _powered(p):
        if p.get("start") is None:
            return True
        dur = tmap[p["task_id"]]["duration"]
        return all(t not in up for t in range(p["start"], p["start"] + dur))
    placed = {p["task_id"] for p in placement
              if not p.get("dropped") and p["task_id"] not in bad and _powered(p)}
    n = len(tasks)
    val_tot = sum(t["drop_penalty"] for t in tasks)
    val_lost = sum(tmap[i]["drop_penalty"] for i in tmap if i not in placed)
    rackh_dem = sum(t["demand"] * t["duration"] for t in tasks)
    rackh_srv = sum(tmap[i]["demand"] * tmap[i]["duration"] for i in placed)
    def sp(p):
        ids = [t["task_id"] for t in tasks if t["priority"] == p]
        return 100.0 * sum(1 for i in ids if i in placed) / len(ids) if ids else 100.0
    return dict(s_cnt=round(100.0 * len(placed) / n, 2),
                s_val=round(100.0 * (1 - val_lost / val_tot), 2) if val_tot else 100.0,
                s_rh=round(100.0 * rackh_srv / rackh_dem, 2) if rackh_dem else 100.0,
                s_h=round(sp("high"), 2), s_m=round(sp("mid"), 2), s_l=round(sp("low"), 2))


# ----------------------------------------------------------------------------
# 评估单个候选解
# ----------------------------------------------------------------------------
def evaluate(inst, placement, optimal_obj=None, time_limit=60, solve_grid=True):
    tasks = {t["task_id"]: t for t in inst["tasks"]}
    load, viols, viol_mw_task = expand_placement(inst, placement)
    task_feasible = (len(viols) == 0)

    drop_sum = sum(tasks[pl["task_id"]]["drop_penalty"]
                   for pl in placement if pl.get("dropped"))
    mig_sum = sum(tasks[pl["task_id"]]["mig_cost"] for pl in placement
                  if (not pl.get("dropped")) and pl["dc"] != tasks[pl["task_id"]]["origin_dc"])

    res = {"task_feasible": task_feasible, "n_task_violations": len(viols),
           "viol_task_mw": round(viol_mw_task, 3), "drop_cost": round(drop_sum, 1),
           "migration_cost": round(mig_sum, 1)}

    res.update(satisfaction(inst, placement))

    if solve_grid:
        grid_cost, lost_load, status = grid_solve(inst, load, time_limit)
        objective = grid_cost + C_LOL * lost_load + drop_sum + mig_sum
        res.update(grid_cost=round(grid_cost, 1), lost_load_mw=round(lost_load, 3),
                   grid_status=status, objective=round(objective, 1),
                   grid=round(grid_cost, 1), drop=round(objective - grid_cost, 1),
                   viol_mw=round(viol_mw_task + lost_load, 3),
                   feasible=bool(task_feasible and lost_load <= 1e-3))
        if optimal_obj:
            res["gap_pct"] = round(100.0 * (objective - optimal_obj) / optimal_obj, 3)
    else:
        res["feasible"] = task_feasible
        res["viol_mw"] = round(viol_mw_task, 3)
    return res


# ----------------------------------------------------------------------------
# 把一组 evaluate() 结果聚合成表格的一整行（15 列）
# ----------------------------------------------------------------------------
def table_row(rows, tpq=None, p95=None, label="method"):
    n = len(rows)
    feas = [r for r in rows if r.get("feasible")]
    def m(key, subset=None):
        src = rows if subset is None else subset
        vals = [r[key] for r in src if r.get(key) is not None]
        return st.mean(vals) if vals else float("nan")
    def R(x, nd=None):                       # NaN 安全的四舍五入
        if isinstance(x, float) and x != x:
            return float("nan")
        return round(x, nd) if nd is not None else round(x)
    return {
        "label": label, "n": n,
        "Feas%": R(100.0 * len(feas) / n, 1) if n else float("nan"),
        "Gap%": R(m("gap_pct", feas), 2),
        "Viol-MW": R(m("viol_mw"), 3),
        "Grid$": R(m("grid", feas)),
        "S-cnt%": R(m("s_cnt"), 1), "S-val%": R(m("s_val"), 1), "S-rh%": R(m("s_rh"), 1),
        "S-H%": R(m("s_h"), 1), "S-M%": R(m("s_m"), 1), "S-L%": R(m("s_l"), 1),
        "Drop$": R(m("drop", feas)),
        "TPQ-s": tpq, "p95-s": p95,
    }


def print_row(row):
    print(f"\n=== {row['label']}  (n={row['n']}) ===")
    for k in ["Feas%", "Gap%", "Viol-MW", "Grid$", "S-cnt%", "S-val%", "S-rh%",
              "S-H%", "S-M%", "S-L%", "Drop$", "TPQ-s", "p95-s"]:
        print(f"  {k:8s} = {row[k]}")
    # LaTeX 行（按你表格列序：Feas-pre Feas-post Gap Viol-MW Grid$ S-cnt S-val S-rh S-H S-M S-L Drop$ TPQ p95）
    def c(x): return "--" if x is None or (isinstance(x, float) and x != x) else x
    latex = " & ".join(str(c(row[k])) for k in
                       ["Feas%", "Feas%", "Gap%", "Viol-MW", "Grid$", "S-cnt%", "S-val%",
                        "S-rh%", "S-H%", "S-M%", "S-L%", "Drop$", "TPQ-s", "p95-s"])
    print(f"  LaTeX: {row['label']} & {latex} \\\\")


# ----------------------------------------------------------------------------
# 模式 4: 评估一个方法输出的解目录 -> 整行
# ----------------------------------------------------------------------------
def eval_dir(pred_dir, data_dir="dataset_v2_pow", label="method", tpq=None, p95=None, limit=None,
             time_limit=60, fast=False):
    insts = {}
    for f in glob.glob(os.path.join(data_dir, "instance_*.json")):
        d = json.load(open(f, encoding="utf-8")); insts[d["instance_id"]] = d
    opt = {}
    for f in glob.glob(os.path.join(data_dir, "solution_*.json")):
        s = json.load(open(f, encoding="utf-8"))
        if s["status"] == "Optimal":
            opt[s["instance_id"]] = s["objective"]

    rows = []
    for pf in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        p = json.load(open(pf, encoding="utf-8"))
        if not isinstance(p, dict) or "task_placement" not in p:
            continue  # 跳过 manifest 等非解文件
        iid = p.get("instance_id")
        inst = insts.get(iid)
        if inst is None:
            continue
        r = evaluate(inst, p["task_placement"], optimal_obj=opt.get(iid),
                     time_limit=time_limit, solve_grid=not fast)
        rows.append(r)
        print(f"  {iid}: feasible={r.get('feasible')} gap%={r.get('gap_pct')} "
              f"S-cnt%={r['s_cnt']}", flush=True)
        if limit and len(rows) >= limit:
            break
    row = table_row(rows, tpq=tpq, p95=p95, label=label)
    print_row(row)
    return row


# ----------------------------------------------------------------------------
# 模式 1: 自检 (拿 Gurobi 解评估，应 全可行 / gap≈0 / viol≈0)
# ----------------------------------------------------------------------------
def self_check(limit=None, time_limit=60):
    inst_files = sorted(glob.glob("dataset_v2/instance_*.json"))
    rows, gaps, ngood = [], [], 0
    for f in inst_files:
        iid = os.path.basename(f).split("__", 1)[1][:-5]
        sf = f.replace("instance_", "solution_")
        if not os.path.exists(sf):
            continue
        inst = json.load(open(f, encoding="utf-8"))
        sol = json.load(open(sf, encoding="utf-8"))
        if sol["status"] != "Optimal":
            continue
        r = evaluate(inst, sol["task_placement"], optimal_obj=sol["objective"],
                     time_limit=time_limit)
        r["instance_id"] = inst["instance_id"]
        r["stored_grid"] = sol["cost_breakdown"]["grid"]
        rows.append(r)
        gaps.append(r.get("gap_pct", 0.0))
        if r["task_feasible"] and abs(r.get("gap_pct", 0)) < 0.5 and r["lost_load_mw"] < 1e-2:
            ngood += 1
        print(f"  {r['instance_id']:32s} task_feas={r['task_feasible']} "
              f"gap%={r.get('gap_pct'):+.3f} viol_mw={r['viol_mw']:.3f} "
              f"grid {r['grid_cost']:.0f} vs stored {r['stored_grid']:.0f}", flush=True)
        if limit and len(rows) >= limit:
            break
    print(f"\n[SELF-CHECK] n={len(rows)} | task-feasible "
          f"{sum(1 for r in rows if r['task_feasible'])}/{len(rows)} | "
          f"reproduced(gap<0.5%,viol≈0) {ngood}/{len(rows)} | "
          f"mean|gap%|={st.mean([abs(g) for g in gaps]):.3f} max|gap%|={max(abs(g) for g in gaps):.3f}")
    return rows


# ----------------------------------------------------------------------------
# 模式 2: Gurobi 参照行 (optimal)
# ----------------------------------------------------------------------------
def gurobi_rows(data_dir="dataset_v2_pow"):
    sols = [json.load(open(f, encoding="utf-8"))
            for f in glob.glob(os.path.join(data_dir, "solution_*.json"))]
    opt = [s for s in sols if s["status"] == "Optimal"]
    ts = sorted(s["solve_time_s"] for s in opt)
    p95 = ts[int(0.95 * len(ts))]
    print(f"[Gurobi (optimal)] n={len(opt)}/{len(sols)} "
          f"(剩 {len(sols)-len(opt)} 个 Not-Solved 不算)")
    print(f"  Feas-pre=100  Feas-post=100  Gap%=0.00  Viol-MW=0.0  "
          f"TPQ-s={st.mean(ts):.1f}  p95-s={p95:.1f}")


# ----------------------------------------------------------------------------
# 模式 3: 加时限重解 25 个 Not-Solved
# ----------------------------------------------------------------------------
def resolve_unsolved(time_limit=600):
    sols = glob.glob("dataset_v2/solution_*.json")
    todo = []
    for sf in sols:
        s = json.load(open(sf, encoding="utf-8"))
        if s["status"] != "Optimal":
            todo.append(s["instance_id"])
    print(f"[RESOLVE] {len(todo)} 个 Not-Solved 待重解 (time_limit={time_limit}s):")
    for iid in todo:
        print("   ", iid)
    print("\n请用更长时限重跑生成器补这些 (它会跳过已 Optimal 的)。"
          "把 eccs_dataset_gen_v2.py 顶部 TIME_LIMIT 调大、删掉这些 Not-Solved 的 solution 文件后重跑即可。")
    return todo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_check", action="store_true")
    ap.add_argument("--gurobi_rows", action="store_true")
    ap.add_argument("--resolve_unsolved", action="store_true")
    ap.add_argument("--eval_dir", type=str, default=None,
                    help="方法输出的解目录（每个 *.json 含 instance_id + task_placement）-> 整行")
    ap.add_argument("--label", type=str, default="method")
    ap.add_argument("--tpq", type=float, default=None, help="该方法的 TPQ-s（外部计时）")
    ap.add_argument("--p95", type=float, default=None, help="该方法的 p95-s（外部计时）")
    ap.add_argument("--fast", action="store_true", help="跳过电网重解，只算满足率/任务层可行")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--data_dir", type=str, default="dataset_v2_pow",
                    help="实例/解目录（instance_*.json + solution_*.json）")
    args = ap.parse_args()
    if args.gurobi_rows:
        gurobi_rows(data_dir=args.data_dir)
    if args.resolve_unsolved:
        resolve_unsolved(args.time_limit)
    if args.self_check:
        self_check(limit=args.limit, time_limit=args.time_limit)
    if args.eval_dir:
        eval_dir(args.eval_dir, data_dir=args.data_dir, label=args.label, tpq=args.tpq, p95=args.p95,
                 limit=args.limit, time_limit=args.time_limit, fast=args.fast)
    if not (args.self_check or args.gurobi_rows or args.resolve_unsolved or args.eval_dir):
        ap.print_help()
