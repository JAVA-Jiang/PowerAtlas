# -*- coding: utf-8 -*-
"""
ECCS-RTS dataset_v2  ->  结构超图 (structural hypergraph) 转换器

每个 instance_XXXX__*.json 转成一张超图：
    节点 = 任务 (task)
    超边 = 把任务耦合在一起的约束组 (origin-DC / time-window / priority)
标签来自配套的 solution_XXXX__*.json。

输出 (默认写到 ./dataset_v2_hyper/):
    <instance_id>.json   人类可读 + 可复用的超图 (节点/特征/标签/超边/稀疏关联表)
    <instance_id>.npz    数值张量 (X, H, 各标签数组)，HGNN 直接加载  [需要 numpy]
    manifest.csv         一行一个实例的汇总
    feature_names.json   节点特征列名 (所有实例一致)

用法:
    python eccs_to_hypergraph.py
    python eccs_to_hypergraph.py --in_dir dataset_v2 --out_dir dataset_v2_hyper
    python eccs_to_hypergraph.py --hyperedges dc_origin,time_window     # 自定义超边类型
"""
import os
import re
import csv
import json
import glob
import argparse

# ----------------------------------------------------------------------------
# 建模配置 (要改超边定义 / 特征，改这里)
# ----------------------------------------------------------------------------
PRIORITY_ORDER = ["low", "mid", "high"]          # priority one-hot 顺序
ALL_HYPEREDGE_TYPES = ["dc_origin", "time_window", "priority"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="dataset_v2")
    p.add_argument("--out_dir", default="dataset_v2_hyper")
    p.add_argument("--hyperedges", default=",".join(ALL_HYPEREDGE_TYPES),
                   help="逗号分隔，可选: " + ",".join(ALL_HYPEREDGE_TYPES))
    p.add_argument("--no_npz", action="store_true", help="不写 .npz (没装 numpy 时用)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 节点特征
# ----------------------------------------------------------------------------
def build_feature_names(dc_ids):
    names = ["release", "deadline", "duration", "demand", "power",
             "drop_penalty", "mig_cost", "migratable", "slack"]
    names += [f"priority_{p}" for p in PRIORITY_ORDER]
    names += [f"origin_dc_{d}" for d in dc_ids]
    return names


def task_feature_vector(task, dc_ids):
    slack = task["deadline"] - task["release"] - task["duration"]
    vec = [
        float(task["release"]),
        float(task["deadline"]),
        float(task["duration"]),
        float(task["demand"]),
        float(task["power"]),
        float(task["drop_penalty"]),
        float(task["mig_cost"]),
        1.0 if task["migratable"] else 0.0,
        float(slack),
    ]
    pr = task.get("priority", "low")
    vec += [1.0 if pr == p else 0.0 for p in PRIORITY_ORDER]
    vec += [1.0 if task["origin_dc"] == d else 0.0 for d in dc_ids]
    return vec


# ----------------------------------------------------------------------------
# 超边构建
# ----------------------------------------------------------------------------
def build_hyperedges(tasks, dc_ids, horizon, types):
    """返回 list[dict(id,type,key,members=[task_id...])]"""
    edges = []

    if "dc_origin" in types:
        for d in dc_ids:
            members = [t["task_id"] for t in tasks if t["origin_dc"] == d]
            if members:
                edges.append({"type": "dc_origin", "key": d, "members": members})

    if "time_window" in types:
        for h in range(1, horizon + 1):
            members = [t["task_id"] for t in tasks
                       if t["release"] <= h <= t["deadline"]]
            if members:
                edges.append({"type": "time_window", "key": h, "members": members})

    if "priority" in types:
        for p in PRIORITY_ORDER:
            members = [t["task_id"] for t in tasks if t.get("priority") == p]
            if members:
                edges.append({"type": "priority", "key": p, "members": members})

    for i, e in enumerate(edges):
        e_id = f'{e["type"]}:{e["key"]}'
        edges[i] = {"id": e_id, **e}
    return edges


# ----------------------------------------------------------------------------
# 标签 (来自 solution)
# ----------------------------------------------------------------------------
def build_label(placement):
    """placement: solution 里某个 task 的 dict。返回统一标签 dict。"""
    if placement is None:
        # solution 里没这个 task —— 当作未知 (-1)
        return {"dropped": -1, "start": -1, "dc_assigned": -1,
                "shifted": -1, "migrated": -1}
    if placement.get("dropped"):
        return {"dropped": 1, "start": -1, "dc_assigned": -1,
                "shifted": -1, "migrated": -1}
    return {
        "dropped": 0,
        "start": int(placement.get("start", -1)),
        "dc_assigned": int(placement.get("dc", -1)),
        "shifted": 1 if placement.get("shifted") else 0,
        "migrated": 1 if placement.get("migrated") else 0,
    }


# ----------------------------------------------------------------------------
# 单实例转换
# ----------------------------------------------------------------------------
def convert_instance(inst, sol, types):
    tasks = sorted(inst["tasks"], key=lambda t: t["task_id"])
    dc_ids = sorted(int(k) for k in inst["dc_buses"].keys())
    horizon = int(inst["horizon"])
    feature_names = build_feature_names(dc_ids)

    # task_id -> placement
    sol_map = {}
    if sol is not None:
        for pl in sol.get("task_placement", []):
            sol_map[pl["task_id"]] = pl

    # 节点
    tid_to_idx = {t["task_id"]: i for i, t in enumerate(tasks)}
    nodes = []
    for t in tasks:
        nodes.append({
            "task_id": t["task_id"],
            "feature_vector": task_feature_vector(t, dc_ids),
            "label": build_label(sol_map.get(t["task_id"])),
        })

    # 超边
    hyperedges = build_hyperedges(tasks, dc_ids, horizon, types)

    # 稀疏关联表 (node_idx, edge_idx)
    pairs = []
    for ej, e in enumerate(hyperedges):
        for tid in e["members"]:
            pairs.append([tid_to_idx[tid], ej])

    return {
        "instance_id": inst["instance_id"],
        "source": inst.get("source"),
        "split_tag": inst.get("split_tag"),
        "horizon": horizon,
        "dc_ids": dc_ids,
        "n_nodes": len(nodes),
        "n_hyperedges": len(hyperedges),
        "feature_names": feature_names,
        "hyperedge_types": types,
        "nodes": nodes,
        "hyperedges": hyperedges,
        "incidence": {"n_nodes": len(nodes), "n_edges": len(hyperedges),
                      "pairs": pairs},
        "has_solution": sol is not None,
    }


def save_npz(hg, path):
    import numpy as np
    X = np.array([n["feature_vector"] for n in hg["nodes"]], dtype=np.float32)
    N, M = hg["n_nodes"], hg["n_hyperedges"]
    H = np.zeros((N, M), dtype=np.float32)
    for ni, ej in hg["incidence"]["pairs"]:
        H[ni, ej] = 1.0
    lab = {k: np.array([n["label"][k] for n in hg["nodes"]], dtype=np.int64)
           for k in ["dropped", "start", "dc_assigned", "shifted", "migrated"]}
    np.savez(path, X=X, H=H,
             task_id=np.array([n["task_id"] for n in hg["nodes"]], dtype=np.int64),
             **{f"y_{k}": v for k, v in lab.items()})


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def find_solution(inst_path):
    sol = inst_path.replace("instance_", "solution_")
    return sol if os.path.exists(sol) else None


def main():
    args = parse_args()
    types = [t.strip() for t in args.hyperedges.split(",") if t.strip()]
    bad = [t for t in types if t not in ALL_HYPEREDGE_TYPES]
    if bad:
        raise SystemExit(f"未知超边类型 {bad}，可选 {ALL_HYPEREDGE_TYPES}")

    os.makedirs(args.out_dir, exist_ok=True)
    inst_files = sorted(glob.glob(os.path.join(args.in_dir, "instance_*.json")))
    if not inst_files:
        raise SystemExit(f"在 {args.in_dir} 没找到 instance_*.json")

    manifest = []
    feat_names_written = False
    for ip in inst_files:
        with open(ip, "r", encoding="utf-8") as f:
            inst = json.load(f)
        sp = find_solution(ip)
        sol = None
        if sp:
            with open(sp, "r", encoding="utf-8") as f:
                sol = json.load(f)

        hg = convert_instance(inst, sol, types)

        out_json = os.path.join(args.out_dir, hg["instance_id"] + ".json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(hg, f, ensure_ascii=False, indent=2)

        if not args.no_npz:
            try:
                save_npz(hg, os.path.join(args.out_dir, hg["instance_id"] + ".npz"))
            except ImportError:
                args.no_npz = True
                print("[warn] 没装 numpy，跳过 .npz（只输出 .json）")

        if not feat_names_written:
            with open(os.path.join(args.out_dir, "feature_names.json"),
                      "w", encoding="utf-8") as f:
                json.dump(hg["feature_names"], f, ensure_ascii=False, indent=2)
            feat_names_written = True

        n_drop = sum(1 for n in hg["nodes"] if n["label"]["dropped"] == 1)
        manifest.append({
            "instance_id": hg["instance_id"],
            "split_tag": hg["split_tag"],
            "n_nodes": hg["n_nodes"],
            "n_hyperedges": hg["n_hyperedges"],
            "n_dropped": n_drop,
            "has_solution": hg["has_solution"],
        })

    with open(os.path.join(args.out_dir, "manifest.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"[done] {len(inst_files)} 个实例 -> {args.out_dir}")
    print(f"       超边类型: {types}")
    print(f"       样例: {manifest[0]}")


if __name__ == "__main__":
    main()
