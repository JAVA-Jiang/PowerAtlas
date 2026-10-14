# -*- coding: utf-8 -*-
"""
生成「领域知识」corpus（外部知识库），供 MMGraph 检索。
两部分：
  A. ECCS 调度常识/规则 —— 从本数据集的 MILP 目标与约束提炼（最贴本问题）
  B. 电网运行常识 —— 来自公开权威来源（DOE / ISO-NE / arXiv 论文等），带出处，
     满足论文"使用了外部知识"的说法。

输出 eccs_knowledge.jsonl，每行 {corpus_text, corpus_images:[], corpus_source}
可直接喂 build.py，或与 eccs_corpus_train.jsonl 合并后一起建库。

用法: python build_knowledge_corpus.py            # -> eccs_knowledge.jsonl
"""
import json, argparse

# ---------------------------------------------------------------------------
# A. ECCS 调度常识（从 MILP 提炼；每条是可直接指导某个决策的规则）
# ---------------------------------------------------------------------------
ECCS_RULES = [
    "Task value density and dropping: in ECCS the penalty for dropping a compute task equals its value-of-compute (high=200, mid=70, low=25 $ per MW-hour) times its power times its duration. Under congestion the cost-optimal policy drops the tasks with the lowest drop penalty first, i.e. low-priority, low-power, short tasks, while protecting high-priority high-value tasks. This is why in optimal schedules high-priority completion stays near 100% while low-priority tasks absorb most of the drops.",

    "Temporal shifting of flexible tasks: a task has slack = deadline - release - duration. Tasks with positive slack can be delayed within their release-deadline window to hours when the grid's marginal generation cost is lower or renewable output is higher, cutting total electricity cost without dropping the task. Tight-slack and high-power tasks are the most valuable to shift because they drive the largest grid cost swings.",

    "Cross-data-center migration: a migratable task may run at a data center other than its origin. Migration is only worthwhile when the destination lowers total cost (less local congestion, cheaper local generation, or more headroom under the DC power cap) by more than the migration cost, which scales with the task's rack demand. Non-migratable tasks must run at their origin data center.",

    "Data-center power and capacity limits: each data center draws an idle baseline power even with no active tasks (about 25% of its peak power), plus per-rack power for running tasks. Two hard limits must hold every hour: total DC load must stay below the DC peak power cap, and the number of concurrently active racks must stay within the DC rack capacity (C=11). Violating either makes a schedule infeasible.",

    "Congestion resolution: when aggregate compute demand cannot be served economically given grid generation cost, transmission limits and DC capacity, the optimizer prefers to drop the least valuable tasks rather than pay very high grid cost or shed firm load. The result is a graceful degradation that sacrifices low-value compute to keep the overall cost minimal.",

    "Objective structure: the ECCS cost has four parts — thermal generation cost (fuel + no-load + startup), a large lost-load penalty for any unserved grid load, drop penalties for tasks not scheduled, and migration costs for tasks moved off their origin. A good schedule trades a small amount of drop/migration cost to avoid large grid or lost-load cost.",

    "Why hourly grid cost varies: generation cost per hour depends on demand level, renewable availability, generator ramping limits and minimum up/down times, and a spinning-reserve requirement. Peak-demand hours and low-renewable hours are the most expensive, so shifting flexible compute away from those hours yields the biggest savings.",

    "Renewable penetration effect: higher renewable penetration lowers and reshapes the hourly cost curve, creating cheaper low-carbon windows that flexible tasks should target. It also increases variability, so the value of temporal shifting grows with penetration; when a day is nearly flat in cost there is little to gain from shifting.",

    "Deadline feasibility: a placed task must start no earlier than its release hour and finish (start + duration - 1) no later than its deadline. Feasible start times therefore lie in [release, deadline - duration + 1]; a task whose window is too tight given capacity contention may have to be dropped.",

    "Priority-first, value-weighted service: because drop penalty is proportional to value-of-compute, protecting high-priority tasks and dropping low-priority ones maximizes value-weighted completion even when raw task-count completion falls. Reporting satisfaction by count, by value, and by priority tier reveals whether a scheduler is sacrificing the right (low-value) tasks.",
]

# ---------------------------------------------------------------------------
# B. 电网运行常识（公开来源，带出处）
#    每条末尾用 (Source: ...) 标注来源，满足"外部知识"说法。
# ---------------------------------------------------------------------------
GRID_KNOWLEDGE = [
    "Demand response (DR) is a short-term, voluntary reduction in electricity consumption by end users, typically triggered by compromised grid reliability or high wholesale prices. In exchange for curtailing or shifting load, customers receive rate discounts, bill credits or other payments. DR helps flatten demand peaks and reduces the need for expensive standby 'peaker' generation. (Source: U.S. Department of Energy, FEMP - Demand Response and Time-Variable Pricing Programs)",

    "Time-based electricity pricing enables demand response. Time-of-Use (TOU) pricing sets different rates for different blocks of the 24-hour day, while Real-Time Pricing (RTP) fluctuates hourly to reflect the wholesale market price. These time-varying prices signal when electricity is cheap or expensive, encouraging consumers to shift flexible load to cheaper hours. (Source: U.S. Department of Energy, FEMP)",

    "Reducing peak demand lowers the overall cost of electricity by reducing the need for additional generation capacity and avoiding the construction and maintenance costs of peaker or standby plants. Shifting flexible load away from peak hours therefore benefits both the consumer's bill and the grid as a whole. (Source: U.S. Department of Energy, FEMP)",

    "Unit commitment (UC) is a mixed-integer optimization that decides which generators are on/off over time to minimize total cost of serving load while respecting physical limits (min up/down times, ramping, capacity). Economic dispatch (ED) then sets each committed unit's output to meet load at least cost. Wholesale markets clear with UC followed by ED. (Source: ISO New England; Elia UC/ED appendix)",

    "Economic dispatch uses the lowest-cost generation first to meet demand. The last (highest-cost) unit dispatched is the 'marginal unit': a small increase or decrease in demand changes this unit's output. The marginal unit's cost sets the system marginal price, which is why serving extra load at peak times is disproportionately expensive. (Source: ISO New England FAQ)",

    "Locational Marginal Price (LMP) is a byproduct of economic dispatch: the cost to serve one more unit of load at a specific network node, accounting for generation cost and transmission constraints. LMPs differ by location and by hour (day-ahead hourly, real-time every five minutes), so the cheapest place and time to run flexible load both vary. (Source: ISO New England; PCI Energy Solutions)",

    "Carbon-aware temporal workload shifting schedules tasks to times of lower grid carbon intensity, which often coincide with high renewable output. It is well suited to batch/flexible workloads that tolerate delayed completion, and is the compute analogue of demand response. (Source: 'Let's Wait Awhile', arXiv:2110.13234)",

    "Carbon-aware spatial workload shifting redistributes tasks across geographically distributed data centers according to their carbon intensity or cost. Studies find spatial shifting yields the most benefit when data centers are roughly 300-400 km apart, where carbon-intensity differences are large enough to matter. (Source: 'Spatio-temporal load shifting for truly clean computing', ScienceDirect / arXiv:2405.00036)",

    "Joint spatio-temporal load shifting across a network of data centers can approach 24/7 carbon-free-energy matching. Co-optimizing energy and data placement has been reported to cut data-center operating cost by around 20% and reduce renewable curtailment, showing the value of coordinating compute scheduling with grid conditions. (Source: Spatio-temporal load shifting, ScienceDirect / arXiv:2405.00036)",

    "Limitations of carbon/cost-aware shifting: temporal shifting helps little when a location's carbon intensity or price is nearly flat over time, and spatial shifting helps little when the rank order of regions by carbon/price never changes. The benefit of shifting is proportional to the variability of grid conditions across time and space. (Source: 'On the Limitations of Carbon-Aware ... Shifting', arXiv:2306.06502)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eccs_knowledge.jsonl")
    ap.add_argument("--source", default="domain-knowledge")
    args = ap.parse_args()

    entries = [("eccs-rule", t) for t in ECCS_RULES] + \
              [("grid-knowledge", t) for t in GRID_KNOWLEDGE]

    with open(args.out, "w", encoding="utf-8") as f:
        for _tag, text in entries:
            rec = {"corpus_text": text.strip(),
                   "corpus_images": [],
                   "corpus_source": args.source}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] {len(entries)} knowledge docs "
          f"({len(ECCS_RULES)} ECCS-rules + {len(GRID_KNOWLEDGE)} grid-knowledge) "
          f"-> {args.out}  (corpus_source={args.source})")


if __name__ == "__main__":
    main()
