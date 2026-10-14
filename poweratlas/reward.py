# -*- coding: utf-8 -*-
# ============================================================
# [讲解] 本文件 = ECCS 调度任务的奖励函数（对应原 F1 版 score.py）。
# [讲解] 这是给 RL 训练用的"快速奖励"：只做任务层可行性 + 价值满足率，不重解电网（太慢）。
# [讲解]   完整的 Gap / 电网成本由 eccs_eval.py 单独算，当验证指标（不进训练回路）。
# [讲解]
# [讲解] ★模型输出格式约定（多轮，最后一轮里给出 <answer>）：
# [讲解]     <reason> ...推理... </reason> <answer>
# [讲解]     0: dc=4 start=12
# [讲解]     1: drop
# [讲解]     2: dc=2 start=20
# [讲解]     </answer>
# [讲解]   每个任务一行：
# [讲解]     "<task_id>: drop"                 → 丢弃该任务
# [讲解]     "<task_id>: dc=<d> start=<t>"     → 放到数据中心 d、第 t 小时开始
# [讲解]
# [讲解] 奖励结构（沿用原版思路，把 F1 换成"调度质量"）：
# [讲解]     quality ∈ [0,1] = 可行则用价值满足率 s_val，不可行则打折；再乘覆盖率
# [讲解]     格式对+有搜索:  reward = quality × (1 − 2/(搜索数+2))
# [讲解]     格式错+有搜索:  reward = −0.5 + quality × (…)
# [讲解]     完全没搜索:     reward = −0.5     ← 逼模型学会"先检索再作答"
# [讲解]
# [讲解] 格式/搜索次数直接复用 score.py 的 format_reward（它已按 <text_search>/<answer>
# [讲解]   多轮规则检查），保证和 rollout 里的搜索触发标签完全一致，避免两边规则跑偏。
# ============================================================
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re

from format_score import format_reward

PRIORITIES = ("high", "mid", "low")


# ===========================================================================
# 1) 解析模型输出
# ===========================================================================
def extract_answer_block(prediction: str):
    """抠出最后一个 <answer>...</answer> 里的文本，没有返回 None。"""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", prediction, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def parse_schedule(answer_text: str):
    """把 <answer> 里的每行解析成 {task_id: ('drop' | (dc, start))}。
    容错：忽略解析不了的行。返回 dict。"""
    placement = {}
    if not answer_text:
        return placement
    for line in answer_text.splitlines():
        line = line.strip().lower()
        if not line:
            continue
        m = re.match(r"(\d+)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        tid = int(m.group(1))
        body = m.group(2)
        if "drop" in body:
            placement[tid] = "drop"
            continue
        dc_m = re.search(r"dc\s*=?\s*(\d+)", body)
        st_m = re.search(r"start\s*=?\s*(\d+)", body)
        if dc_m and st_m:
            placement[tid] = (int(dc_m.group(1)), int(st_m.group(1)))
    return placement


# ===========================================================================
# 2) 任务层可行性 + 满足率（快速，不解电网）
# ===========================================================================
def evaluate_schedule(instance: dict, placement: dict):
    """返回 dict：feasible / n_violations / s_val / s_cnt / covered。
    instance 需含 tasks(list) 和 dc(含 C, Pmax, Pidle)、dc_buses(dict)。"""
    tasks = {t["task_id"]: t for t in instance["tasks"]}
    dcP = instance["dc"]
    C, Pmax_dc, Pidle = dcP["C"], dcP["Pmax"], dcP["Pidle"]
    dc_ids = set(int(k) for k in instance["dc_buses"].keys())

    viols = []
    load = {}  # (dc, t) -> [demand_sum, power_sum]
    placed_ids = set()

    for tid, dec in placement.items():
        if tid not in tasks:
            viols.append(("bad_task_id", tid)); continue
        if dec == "drop":
            continue
        dc, start = dec
        tk = tasks[tid]
        dur = tk["duration"]
        # 迁移性：不可迁移必须在原 DC
        allowed = dc_ids if tk["migratable"] else {tk["origin_dc"]}
        if dc not in allowed:
            viols.append(("migratability", tid))
        # 时间窗：start ∈ [release, deadline-duration+1]
        if not (tk["release"] <= start <= tk["deadline"] - dur + 1):
            viols.append(("window", tid))
        placed_ids.add(tid)
        for t in range(start, start + dur):
            cell = load.setdefault((dc, t), [0.0, 0.0])
            cell[0] += tk["demand"]
            cell[1] += tk["power"]

    # 容量 C 与功率 Pmax（每个 DC 每小时）
    for (dc, t), (dem, pw) in load.items():
        if dem > C + 1e-6:
            viols.append(("capacity", dc, t))
        if Pidle + pw > Pmax_dc + 1e-6:
            viols.append(("dc_power", dc, t))

    n = len(tasks)
    # 价值满足率 = 已服务价值 / 总价值（drop_penalty 作价值权重）
    val_tot = sum(t["drop_penalty"] for t in tasks.values())
    val_srv = sum(tasks[i]["drop_penalty"] for i in placed_ids)
    s_val = (val_srv / val_tot) if val_tot else 1.0
    s_cnt = len(placed_ids) / n if n else 1.0
    covered = len(placement) / n if n else 0.0  # 模型给了多少任务的决策

    return dict(feasible=(len(viols) == 0), n_violations=len(viols),
                s_val=round(s_val, 4), s_cnt=round(s_cnt, 4),
                covered=round(covered, 4))


# ===========================================================================
# 2b) 电力层评估（系统级；与 prompt 的 base_residual 口径严格一致）
#     发电需求 = base_residual + 放置任务功率(耦合)；swing 补平衡；成本对比最优标签。
#     线路潮流/Viol(MW) 的完整版留给全量评估器（需 per-bus），此处只做系统级快评。
# ===========================================================================
def split_answer_sections(answer_text: str):
    """把 <answer> 拆成 (tasks_text, power_text)。无 POWER 段时 power_text=''（兼容 v1）。"""
    if not answer_text:
        return "", ""
    low = answer_text.lower()
    ti, pi = low.find("tasks:"), low.find("power:")
    if pi == -1:
        return (answer_text[ti + 6:] if ti != -1 else answer_text), ""
    return answer_text[(ti + 6 if ti != -1 else 0):pi], answer_text[pi + 6:]


def _parse_level(tok: str):
    """单个出力档位：'off'->'off'；[0,100] 浮点->float；非法->None。"""
    s = tok.strip().lower()
    if s == "off":
        return "off"
    try:
        v = float(s)
        if -1e-6 <= v <= 100 + 1e-6:
            return max(0.0, min(100.0, v))
    except Exception:
        pass
    return None


def parse_dispatch(power_text: str, controllable_uids: set):
    """POWER 段 -> {uid: [24 个 ('off'|int|None)]}，只收可调（非 swing）机组。"""
    disp = {}
    for line in (power_text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        uid, rhs = line.split(":", 1)
        uid = uid.strip()
        if uid in controllable_uids:
            disp[uid] = [_parse_level(x) for x in rhs.split(",") if x.strip() != ""]
    return disp


def _pct_to_mw(g, val):
    """('off'|int|None)->MW。off/None->0；int p -> Pmin + p/100*(Pmax-Pmin)。"""
    if val == "off" or val is None:
        return 0.0
    return g["Pmin"] + (val / 100.0) * (g["Pmax"] - g["Pmin"])


def _gen_cost(g, mw_arr):
    """一台机组 24h 成本：cP*P + cNL*(on) + cSU*(启动)，与 oracle grid_v 口径一致。"""
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


def evaluate_power(instance: dict, placement: dict, dispatch: dict,
                   infeasible_factor: float = 0.5):
    """系统级电力评估 -> grid_quality ∈ [0,1] + 明细。
    需 instance 含 adjustable_gens/swing_gen/grid_signal/dc/dc_buses/tasks/dispatch_label_pct。"""
    adj = instance["adjustable_gens"]
    gmap = {g["uid"]: g for g in adj}
    swing_uid = instance["swing_gen"]; swing = gmap[swing_uid]
    ctrl = [g for g in adj if not g["is_swing"]]
    gs = instance["grid_signal"]; net, fixed = gs["net"], gs["fixed_gen_agg"]
    T = len(net)
    idle_total = len(instance["dc_buses"]) * instance["dc"]["Pidle"]

    # 放置任务的逐时功率（系统级；DC 位置不影响功率平衡，线路留给全量评估器）
    tasks = {t["task_id"]: t for t in instance["tasks"]}
    dc_pow = [0.0] * T
    for tid, dec in placement.items():
        if dec == "drop" or tid not in tasks:
            continue
        _dc, start = dec; tk = tasks[tid]
        for t in range(start, start + tk["duration"]):
            if 1 <= t <= T:
                dc_pow[t - 1] += tk["power"]

    # 可调机组出力（MW）+ 统计非法/缺失档位
    n_bad = 0; ctrl_mw = {}
    for g in ctrl:
        arr = dispatch.get(g["uid"], [])
        mw = []
        for t in range(T):
            v = arr[t] if t < len(arr) else None
            if v is None:
                n_bad += 1
            mw.append(_pct_to_mw(g, v))
        ctrl_mw[g["uid"]] = mw

    # 平衡（含弃风光消纳）：热机总量(固定67 + 6台)可落在 [need, need+renewable] 内 ——
    #   need = base_residual + dc_pow（全额用新能源时的下界）；弃风光则需更多热机(上界 +renewable)。
    #   => 6台总量 Σ6 = Σ(可调)+swing 必须 ∈ [need, need+renewable]，swing ∈ {0}∪[Pmin,Pmax]。
    #   取最省(=出力最低=多用免费新能源)的可行 swing；落不进则记违规(缺电/过发超可弃量)。
    renew = gs["renewable"]
    Pmn, Pmx = swing["Pmin"], swing["Pmax"]
    TOL = 1.0                                       # 浮点误差容差(MW)
    swing_mw = [0.0] * T; n_grid_viol = 0; viol_mw = 0.0
    for t in range(T):
        need = net[t] + idle_total - fixed[t] + dc_pow[t]
        sig = sum(ctrl_mw[g["uid"]][t] for g in ctrl)
        s_lo = need - sig - TOL                     # swing 下界(全额用新能源)
        s_hi = need + renew[t] - sig + TOL          # swing 上界(全弃新能源)
        if s_lo <= 0 <= s_hi:                       # swing 可停(必要时弃风光吸收过发)
            sw = 0.0
        elif s_hi >= Pmn and s_lo <= Pmx:           # 与 [Pmin,Pmax] 有交 -> 取最低可行出力
            sw = min(max(s_lo, Pmn), Pmx)
        else:                                       # 不可行
            if s_lo > Pmx:   v = s_lo - Pmx; sw = Pmx          # swing 顶格仍缺电
            elif s_hi < 0:   v = -s_hi;      sw = 0.0          # 过发超过可弃风光
            else:            v = min(s_lo, Pmn - s_hi); sw = Pmn   # 卡在 (0,Pmin) 缝隙
            n_grid_viol += 1; viol_mw += max(0.0, v)
        swing_mw[t] = sw
    grid_feasible = (n_grid_viol == 0 and n_bad == 0)

    # 成本对比：LLM(5可调 + swing重构) vs 最优标签(6台)
    llm_mw = dict(ctrl_mw); llm_mw[swing_uid] = [max(0.0, s) for s in swing_mw]
    cost_llm = sum(_gen_cost(gmap[u], llm_mw[u]) for u in gmap)
    lab = instance.get("dispatch_label_pct") or {}
    cost_opt = None
    if lab:
        cost_opt = sum(_gen_cost(gmap[u], [_pct_to_mw(gmap[u], v) for v in lab[u]])
                       for u in gmap if u in lab)
    # [v3 修复] 原 cost_ratio×feas_factor 会奖励"少发电"(省钱→cost_ratio顶满,缺电只轻扣)。
    #   改为直接镜像真实目标(grid_cost + C_LOL*失负荷):把失负荷折进 LLM 成本,少发电立刻变贵。
    #   可行时 grid_quality = 成本效率 cost_opt/cost_llm;缺电越多分越低(feas_factor 由此内生)。
    LOL_PRICE = 3000.0                                          # 失负荷惩罚价($/MWh);真实 C_LOL=9000,可调
    llm_grid_obj = cost_llm + LOL_PRICE * viol_mw
    grid_quality = min(1.0, cost_opt / llm_grid_obj) if (cost_opt and llm_grid_obj > 1e-6) else 0.0
    # feas_factor 仅保留作 info 展示(不再进 grid_quality)
    feas_factor = round(1.0 - min(1.0, viol_mw / max((swing["Pmax"] - swing["Pmin"]) * T, 1.0)), 4)

    return dict(grid_quality=round(grid_quality, 4), grid_feasible=grid_feasible,
                feas_factor=round(feas_factor, 4),
                n_grid_viol=n_grid_viol, viol_mw=round(viol_mw, 1), n_bad_levels=n_bad,
                cost_llm=round(cost_llm, 1),
                cost_opt=(round(cost_opt, 1) if cost_opt else None),
                cost_ratio=round(cost_ratio, 4))


# ===========================================================================
# 3) 合成总奖励：joint_quality = w_compute*算力 + w_power*电力，再套格式/搜索系数
# ===========================================================================
def compute_score_eccs(prediction: list, instance: dict,
                       infeasible_factor: float = 0.5,
                       w_compute: float = 0.5, w_power: float = 0.5):
    """prediction: 多轮响应文本列表；instance: 该实例。
    v2: <answer> 分 TASKS: / POWER: 两段，联合评算力+电力。返回 (overall, quality, search_count, info)。"""
    assert len(prediction) > 0, "Model responses are empty!"
    answer_text = extract_answer_block(prediction[-1])
    tasks_text, power_text = split_answer_sections(answer_text)

    # 算力层（可行用价值满足率，不可行打折，再乘覆盖率）
    placement = parse_schedule(tasks_text or answer_text)
    ev = evaluate_schedule(instance, placement)
    # [v3] 服务度 = 价值0.7 + 数量0.3:数量项抑制"过度丢弃低价值任务"(模型曾丢11个 vs oracle丢3个)。
    #   原来只用 s_val(价值)→ 丢低价值几乎不扣分 → 敢乱丢来给烂调度减负荷。
    served = 0.7 * ev["s_val"] + 0.3 * ev["s_cnt"]
    compute_quality = (served if ev["feasible"] else infeasible_factor * served) * ev["covered"]

    # 电力层（v2；实例缺 adjustable_gens 则退回纯算力，兼容 v1）
    power_info = {}
    if "adjustable_gens" in instance:
        ctrl_uids = {g["uid"] for g in instance["adjustable_gens"] if not g["is_swing"]}
        dispatch = parse_dispatch(power_text, ctrl_uids)
        power_info = evaluate_power(instance, placement, dispatch, infeasible_factor)
        quality = w_compute * compute_quality + w_power * power_info["grid_quality"]
    else:
        quality = compute_quality

    fmt, search_count = format_reward(prediction)
    # [v2 修复] 原逻辑"没搜索→overall=-0.5"会把 quality 清零:SFT 只教了单轮不搜索,
    #   GRPO rollout 全部 search_count=0 → 每样本恒 -0.5 → 组内零方差 → 零优势 → 零梯度,学不动。
    #   改为:格式合法就直接用 joint quality 当 reward(它有方差,给 GRPO 梯度);检索给一个
    #   封顶的加法 bonus(用了才加,保留"知识引导"激励但不再因不搜索而清零);格式坏才重罚 -0.5。
    #   (弃掉原乘法折扣 (1-2/(n+2)):那是多跳 QA"越搜越好"的口径,ECCS 不需要为搜而搜。)
    if fmt == 1:
        search_bonus = 0.1 * min(search_count, 2)   # 0 / 0.1 / 0.2 封顶
        overall = quality + search_bonus
    else:
        overall = -0.5

    info = {"quality": round(quality, 4), "compute_quality": round(compute_quality, 4),
            "format": fmt, "search_count": search_count, **ev, **power_info}
    return overall, quality, (search_count if fmt == 1 else 0), info
