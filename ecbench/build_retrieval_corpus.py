# -*- coding: utf-8 -*-
"""
把 dataset_v2 的已解实例序列化成 MMGraph-Agent 可检索的案例库 corpus。

输出 JSONL，每行一条 (MMGraph build.py 要求的格式):
    {"corpus_text": <英文案例文本>, "corpus_images": [], "corpus_source": "ECCS-RTS"}

设计:
  - 每个实例 = 一条案例文档 (实例概况 + 最优决策结果 + 关键决策)。
  - 默认只收录 train split (rho<=0.10) 且 status==Optimal，避免推理时检索到测试答案 (泄题)。
  - 无图，corpus_images 恒为 []。

用法:
    python build_corpus.py                          # -> eccs_corpus_train.jsonl (仅 train)
    python build_corpus.py --splits train,test_id   # 自定义收录哪些 split
    python build_corpus.py --out all_datasets_corpus.jsonl --source ECCS-RTS
"""
import os, json, glob, argparse

SEASON = {1: "winter", 4: "spring", 7: "summer", 10: "autumn"}


def money(x):
    return f"{round(x):,}"


def case_text(inst, sol):
    tasks = inst["tasks"]
    n = len(tasks)
    prio = {p: sum(1 for t in tasks if t["priority"] == p) for p in ("high", "mid", "low")}
    n_mig = sum(1 for t in tasks if t["migratable"])
    durs = [t["duration"] for t in tasks]
    dems = [t["demand"] for t in tasks]
    rackh = sum(t["demand"] * t["duration"] for t in tasks)

    dcP = inst["dc"]
    bus_list = ", ".join(str(b) for b in inst["dc_buses"].values())
    season = SEASON.get(inst["month"], "a")
    pen = round(inst["penetration"] * 100)

    cb = sol["cost_breakdown"]
    # 满足率 (数量/价值/分级)
    tmap = {t["task_id"]: t for t in tasks}
    placed_ids = {pl["task_id"] for pl in sol["task_placement"] if not pl.get("dropped")}
    dropped = [pl["task_id"] for pl in sol["task_placement"] if pl.get("dropped")]
    migrated = [pl["task_id"] for pl in sol["task_placement"]
                if not pl.get("dropped") and pl.get("migrated")]
    val_tot = sum(t["drop_penalty"] for t in tasks)
    val_lost = sum(tmap[i]["drop_penalty"] for i in dropped)
    s_cnt = 100.0 * len(placed_ids) / n
    s_val = 100.0 * (1 - val_lost / val_tot) if val_tot else 100.0
    def sat_p(p):
        ids = [t["task_id"] for t in tasks if t["priority"] == p]
        if not ids:
            return 100.0
        return 100.0 * sum(1 for i in ids if i in placed_ids) / len(ids)

    dropped_str = (", ".join(f"#{i}({tmap[i]['priority']})" for i in dropped)
                   if dropped else "none")
    migrated_str = (", ".join(f"#{i}" for i in migrated) if migrated else "none")

    return (
        f"Case {inst['instance_id']}: edge-cloud compute co-scheduling on the "
        f"RTS-GMLC {inst['n_buses']}-bus power grid. "
        f"Scenario: {season} day (month {inst['month']}, day {inst['day']}), "
        f"{pen}% renewable penetration, {inst['horizon']}-hour horizon. "
        f"Four data centers sit at buses {bus_list}, each with {dcP['C']} racks, "
        f"peak power {dcP['Pmax']} MW and idle power {dcP['Pidle']} MW "
        f"(per-rack power {dcP['pi']} MW). "
        f"The workload has {n} tasks ({prio['high']} high-, {prio['mid']} mid-, "
        f"{prio['low']} low-priority; {n_mig} migratable), with durations "
        f"{min(durs)}-{max(durs)} h and demands {min(dems)}-{max(dems)} racks, "
        f"totaling {rackh} rack-hours of compute. "
        f"Cost-optimal schedule (Gurobi): {sol['n_placed']} tasks placed, "
        f"{sol['n_dropped']} dropped, {sol['n_shifted']} time-shifted, "
        f"{sol['n_migrated']} migrated across data centers. "
        f"Total cost ${money(sol['objective'])}: grid ${money(cb['grid'])}, "
        f"drop penalty ${money(cb['drop'])}, migration ${money(cb['migration'])}. "
        f"Compute satisfaction {s_cnt:.1f}% by count and {s_val:.1f}% by value; "
        f"by priority high {sat_p('high'):.1f}%, mid {sat_p('mid'):.1f}%, "
        f"low {sat_p('low'):.1f}%. "
        f"Under this congestion level the optimizer protected high-value tasks and "
        f"dropped low-priority tasks: {dropped_str}. Migrated tasks: {migrated_str}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dataset_v2")
    ap.add_argument("--out", default="eccs_corpus_train.jsonl")
    ap.add_argument("--source", default="ECCS-RTS")
    ap.add_argument("--splits", default="train",
                    help="逗号分隔: train,test_id,test_ood")
    ap.add_argument("--include_unsolved", action="store_true",
                    help="也收录 Not-Solved 的可行解 (默认只收 Optimal)")
    args = ap.parse_args()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    inst_files = sorted(glob.glob(os.path.join(args.in_dir, "instance_*.json")))
    n_written = 0
    skipped_split = skipped_status = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for ip in inst_files:
            inst = json.load(open(ip, encoding="utf-8"))
            if inst.get("split_tag") not in splits:
                skipped_split += 1
                continue
            sp = ip.replace("instance_", "solution_")
            if not os.path.exists(sp):
                continue
            sol = json.load(open(sp, encoding="utf-8"))
            if sol["status"] != "Optimal" and not args.include_unsolved:
                skipped_status += 1
                continue
            rec = {"corpus_text": case_text(inst, sol),
                   "corpus_images": [],
                   "corpus_source": args.source}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] wrote {n_written} case docs -> {args.out}")
    print(f"       splits kept: {sorted(splits)} | skipped(other split)={skipped_split}"
          f" | skipped(not optimal)={skipped_status}")


if __name__ == "__main__":
    main()
