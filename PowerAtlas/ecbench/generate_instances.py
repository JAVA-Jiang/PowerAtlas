"""
ECCS dataset generator (100 instances).
- Grid layer: real RTS-GMLC. DC/task/coupling: frozen DATASET.md/FORMULATION.md v1.
- Solver: Gurobi if available (pip install gurobipy + academic license), else CBC.
- Resumable: skips instances already on disk. Re-run after a crash to continue.

SETUP (local):
  1) pip install pulp pandas numpy gurobipy
  2) Unzip RTS-GMLC-master.zip somewhere, set RTS_ROOT below to that folder.
  3) python eccs_dataset_gen.py
Outputs go to ./dataset_out/ : instance_XXX_*.json, solution_XXX_*.json, manifest.{csv,json}
"""
import os, json, csv, time, random, sys, subprocess
import pandas as pd, numpy as np
import pulp

# ============================ CONFIG ============================
RTS_ROOT = os.environ.get("RTS_ROOT", "RTS-GMLC-master")          # <-- set to your unzipped RTS-GMLC-master folder
OUT_DIR  = os.environ.get("ECBENCH_OUT", "dataset")         # v3: DC 负载 ÷5(新目录,保留旧 dataset_v2_pow 不覆盖)
T        = 24
MIG_RATIO= 0.5
TIME_LIMIT = 150                       # seconds per instance
MIP_GAP    = 0.02                      # relative gap target

# ---- Liaoning calibration (real DC statistics, 11 racks x 70 days) ----
# CALIBRATED from real data:  racks-per-DC, peak-power magnitude (peak_norm).
# ASSUMED (no task-level data available): task durations / demands / arrivals /
#   deadlines / migratability / priority, AND a deliberately RELAXED load factor.
# Liaoning observed: peak_norm mean=0.74 (daily peak ~74% of full), LF mean=0.89 (very flat,
#   near-saturated -> little shift room). We keep the peak magnitude but RELAX schedulability
#   (deadline slack + migratability) to study co-scheduling potential ("assumptive relaxation").
RACKS_PER_DC    = 11        # CALIBRATED to Liaoning (n_racks=11 per DC)
PEAK_NORM_MEAN  = 0.74      # CALIBRATED: target aggregate DC peak ~74% of full capacity
TARGET_UTIL     = 0.35      # ASSUMED avg rack-utilization sizing the task pool (<0.89 real LF, to keep shift room)
# N_TASKS sized so total rack-hours ~= TARGET_UTIL * (4 DCs * RACKS * T); E[demand]~2.5, E[dur]~4
N_TASKS = max(1, round(TARGET_UTIL * 4 * RACKS_PER_DC * T / (2.5*4.0)))

# 200-instance grid: 5 day-points x 4 penetrations x 10 seeds = 200
DAYS  = [(m,d) for m in range(1,12) for d in (5,12,19,26)] + [(12,5)]  # 45 new (mo,da) for 2000-expansion   # winter / spring / summer / autumn / extra summer
RHOS  = [0.01, 0.02, 0.04, 0.06]   # v3: DC 负载调小 5 倍(原 0.05/0.10/0.20/0.30 ÷5)
SEEDS = list(range(1, 11))   # 10 seeds -> 5 days x 4 rhos x 10 = 200 instances
# IID 160/40 split BY SEED: 四个 penetration + 五个代表日在 train/test 都出现。
# seed 1-8 -> train(8*5*4=160)；seed 9-10 -> test(2*5*4=40，全新任务池=同分布泛化)。
def split_of(seed): return "train" if seed <= 8 else "test"

SRC = f"{RTS_ROOT}/RTS_Data/SourceData"
TS  = f"{RTS_ROOT}/RTS_Data/timeseries_data_files"
THERMAL={"CT","CC","STEAM","NUCLEAR"}
DC_BUS_IDX=[3,7,28,70]
ALPHA=0.25; C_RACKS=RACKS_PER_DC; R_FRAC=0.03; C_LOL=9000.0

# ---- v2: adjustable generators (LLM decides these; the other 67 thermal units are the fixed
#           background, held at their full-optimal dispatch). Data-driven selection = top-6 thermal
#           units by hourly dispatch variance (see scratchpad/pick_gens.py). ----
ADJUSTABLE_GENS = ["123_STEAM_3","218_CC_1","118_CC_1","321_CC_1","223_STEAM_3","221_CC_1"]
SWING_GEN       = "223_STEAM_3"   # absorbs residual so hourly power balance holds by construction
PCT_STEP        = 5               # LLM dispatch = integer % of [Pmin,Pmax], step 5 -> 21 levels (0=off)

# ============================ LOAD STATIC ============================
bus_df=pd.read_csv(f"{SRC}/bus.csv").sort_values("Bus ID").reset_index(drop=True)
br_df =pd.read_csv(f"{SRC}/branch.csv")
gen_df=pd.read_csv(f"{SRC}/gen.csv")
buses=bus_df["Bus ID"].tolist()
bus_area=dict(zip(bus_df["Bus ID"],bus_df["Area"]))
ref_bus=buses[0]
dc_buses={k+1:int(bus_df.iloc[idx-1]["Bus ID"]) for k,idx in enumerate(DC_BUS_IDX)}
area_total={a:bus_df[bus_df["Area"]==a]["MW Load"].sum() for a in bus_df["Area"].unique()}
pf={b:(bus_df.set_index("Bus ID").loc[b,"MW Load"]/area_total[bus_area[b]] if area_total[bus_area[b]]>0 else 0) for b in buses}

LARGE=1e4
lines=[]
for _,r in br_df.iterrows():
    x=float(r["X"]); F=float(r["Cont Rating"]); F=F if F>0 else LARGE
    lines.append((int(r["From Bus"]),int(r["To Bus"]),x,F))

def gcost(r):
    fp=float(r["Fuel Price $/MMBTU"]); vom=float(r.get("VOM",0) or 0)
    hr0=float(r["HR_avg_0"]); hr1=float(r["HR_incr_1"]) if not pd.isna(r["HR_incr_1"]) else hr0
    cP=hr1/1000.0*fp+vom
    cNL=max(0.0,float(r["PMin MW"])*(hr0-hr1)/1000.0*fp)
    sh=float(r["Start Heat Cold MBTU"]) if not pd.isna(r["Start Heat Cold MBTU"]) else 0.0
    nf=float(r["Non Fuel Start Cost $"]) if not pd.isna(r["Non Fuel Start Cost $"]) else 0.0
    return round(cP,3),round(cNL,2),round(nf+sh*fp,1)

gens={}
for _,r in gen_df.iterrows():
    if r["Unit Type"] not in THERMAL or float(r["PMax MW"])<=0: continue
    cP,cNL,cSU=gcost(r); ramp=float(r["Ramp Rate MW/Min"])*60.0
    pmin=float(r["PMin MW"]); pmax=float(r["PMax MW"])
    gens[r["GEN UID"]]=dict(bus=int(r["Bus ID"]),Pmin=pmin,Pmax=pmax,cP=cP,cNL=cNL,cSU=cSU,
        RU=ramp,RD=ramp,SU=max(pmin,ramp),SD=max(pmin,ramp),
        UT=int(max(1,r["Min Up Time Hr"])),DT=int(max(1,r["Min Down Time Hr"])))

# system peak across whole year (for DC capacity base)
_load=pd.read_csv(f"{TS}/Load/DAY_AHEAD_regional_Load.csv")
_load["total"]=_load[["1","2","3"]].sum(axis=1)
SYS_PEAK=float(_load["total"].max())

def parse_bus(c):
    try: return int(c.split("_")[0])
    except: return None
def renew_by_bus(fname,month,day):
    df=pd.read_csv(fname); d=df[(df["Month"]==month)&(df["Day"]==day)].sort_values("Period")
    out={}
    for col in df.columns:
        if col in ["Year","Month","Day","Period"]: continue
        b=parse_bus(col)
        if b not in bus_area: continue
        for p,val in zip(d["Period"],d[col]): out[(b,int(p))]=out.get((b,int(p)),0.0)+float(val)
    return out
def build_grid(month,day):
    load=pd.read_csv(f"{TS}/Load/DAY_AHEAD_regional_Load.csv")
    d=load[(load["Month"]==month)&(load["Day"]==day)].sort_values("Period").head(T)
    area_load={int(a):dict(zip(d["Period"],d[str(a)])) for a in [1,2,3]}
    D={(b,t):pf[b]*area_load[bus_area[b]][t] for b in buses for t in range(1,T+1)}
    Pre={(b,t):0.0 for b in buses for t in range(1,T+1)}
    for fn in [f"{TS}/WIND/DAY_AHEAD_wind.csv",f"{TS}/PV/DAY_AHEAD_pv.csv",
               f"{TS}/RTPV/DAY_AHEAD_rtpv.csv",f"{TS}/Hydro/DAY_AHEAD_hydro.csv"]:
        for (b,t),v in renew_by_bus(fn,month,day).items():
            if 1<=t<=T: Pre[(b,t)]=Pre.get((b,t),0.0)+v
    syspeak_day=max(sum(D[(b,t)] for b in buses) for t in range(1,T+1))
    return D,Pre,syspeak_day

def build_tasks(pi,seed):
    rnd=random.Random(seed); tasks=[]
    for i in range(N_TASKS):
        dur=rnd.choice([2,3,4,5,6]); r=rnd.choice([1,2,3,4])   # r = racks (of RACKS_PER_DC=11)
        rel=rnd.randint(1,T-dur); slack=rnd.choice([3,5,7,9]); dl=min(T,rel+dur-1+slack)
        o=rnd.choice([1,2,3,4]); mig=rnd.random()<MIG_RATIO; prio=rnd.choice(["high","high","mid","low"])
        p=r*pi; VoC={"high":200.0,"mid":70.0,"low":25.0}[prio]
        tasks.append(dict(task_id=i,origin_dc=o,release=rel,deadline=dl,duration=dur,demand=r,
            power=round(p,3),migratable=mig,priority=prio,drop_penalty=round(VoC*p*dur,1),mig_cost=round(8.0*r,1)))
    return tasks

# ============================ SOLVER SELECT ============================
def make_solver():
    try:
        _gt=os.environ.get("GRB_THREADS")
        _kw={"threads":int(_gt)} if _gt else {}
        s=pulp.GUROBI(msg=0,timeLimit=TIME_LIMIT,gapRel=MIP_GAP,**_kw)
        if s.available(): return s,"gurobi"
    except Exception: pass
    return pulp.PULP_CBC_CMD(msg=0,timeLimit=TIME_LIMIT,gapRel=MIP_GAP,threads=4),"cbc"
SOLVER,SOLVER_NAME=make_solver()

# ============================ MILP ORACLE ============================
def solve(month,day,rho,seed):
    D,Pre,syspeak_day=build_grid(month,day)
    Pmax_dc=rho*SYS_PEAK/4.0; Pidle=ALPHA*Pmax_dc; C=C_RACKS; pi=(Pmax_dc-Pidle)/C
    tasks=build_tasks(pi,seed); J=[1,2,3,4]
    def Ji(t):return J if t["migratable"] else [t["origin_dc"]]
    def Si(t):return list(range(t["release"],t["deadline"]-t["duration"]+2))
    TT=range(1,T+1); m=pulp.LpProblem("ECCS",pulp.LpMinimize)
    u={(g,t):pulp.LpVariable(f"u_{g}_{t}",cat="Binary") for g in gens for t in TT}
    vv={(g,t):pulp.LpVariable(f"v_{g}_{t}",cat="Binary") for g in gens for t in TT}
    w={(g,t):pulp.LpVariable(f"w_{g}_{t}",cat="Binary") for g in gens for t in TT}
    P={(g,t):pulp.LpVariable(f"P_{g}_{t}",lowBound=0) for g in gens for t in TT}
    th={(n,t):pulp.LpVariable(f"th_{n}_{t}") for n in buses for t in TT}
    fl={(li,t):pulp.LpVariable(f"f_{li}_{t}") for li in range(len(lines)) for t in TT}
    pc={(n,t):pulp.LpVariable(f"pc_{n}_{t}",lowBound=0) for n in buses for t in TT}
    ls={(n,t):pulp.LpVariable(f"ls_{n}_{t}",lowBound=0) for n in buses for t in TT}
    s={}
    for tk in tasks:
        for j in Ji(tk):
            for t in Si(tk): s[(tk["task_id"],j,t)]=pulp.LpVariable(f"s_{tk['task_id']}_{j}_{t}",cat="Binary")
    def z(i,j,t):
        tk=tasks[i]; return pulp.lpSum(s[(i,j,tau)] for tau in Si(tk) if (i,j,tau) in s and tau<=t<=tau+tk["duration"]-1)
    m += (pulp.lpSum(gens[g]["cP"]*P[(g,t)]+gens[g]["cNL"]*u[(g,t)]+gens[g]["cSU"]*vv[(g,t)] for g in gens for t in TT)
          + pulp.lpSum(C_LOL*ls[(n,t)] for n in buses for t in TT)
          + pulp.lpSum(tasks[i]["drop_penalty"]*(1-pulp.lpSum(s[(i,j,t)] for j in Ji(tasks[i]) for t in Si(tasks[i]))) for i in range(len(tasks)))
          + pulp.lpSum(tasks[i]["mig_cost"]*s[(i,j,t)] for i in range(len(tasks)) for j in Ji(tasks[i]) if j!=tasks[i]["origin_dc"] for t in Si(tasks[i])))
    for g in gens:
        G=gens[g]
        for t in TT:
            up=u[(g,t-1)] if t>1 else 0; Pp=P[(g,t-1)] if t>1 else 0
            m += vv[(g,t)]-w[(g,t)]==u[(g,t)]-up; m += vv[(g,t)]+w[(g,t)]<=1
            m += P[(g,t)]>=G["Pmin"]*u[(g,t)]; m += P[(g,t)]<=G["Pmax"]*u[(g,t)]
            m += P[(g,t)]-Pp<=G["RU"]*up+G["SU"]*vv[(g,t)]; m += Pp-P[(g,t)]<=G["RD"]*u[(g,t)]+G["SD"]*w[(g,t)]
        for t in TT:
            if t>=G["UT"]: m += pulp.lpSum(vv[(g,tau)] for tau in range(t-G["UT"]+1,t+1))<=u[(g,t)]
            if t>=G["DT"]: m += pulp.lpSum(w[(g,tau)] for tau in range(t-G["DT"]+1,t+1))<=1-u[(g,t)]
    for t in TT: m += pulp.lpSum(gens[g]["Pmax"]*u[(g,t)]-P[(g,t)] for g in gens)>=R_FRAC*sum(D[(b,t)] for b in buses)
    for n in buses:
        for t in TT: m += pc[(n,t)]<=Pre[(n,t)]
    for li,(a,b,x,F) in enumerate(lines):
        for t in TT:
            m += fl[(li,t)]==(1.0/x)*(th[(a,t)]-th[(b,t)]); m += fl[(li,t)]<=F; m += fl[(li,t)]>=-F
    for t in TT: m += th[(ref_bus,t)]==0
    L={}
    for j in J:
        for t in TT:
            Lv=pulp.LpVariable(f"L_{j}_{t}",lowBound=0)
            m += Lv==Pidle+pulp.lpSum(tasks[i]["power"]*z(i,j,t) for i in range(len(tasks)) if j in Ji(tasks[i]))
            m += Lv<=Pmax_dc; L[(j,t)]=Lv
            m += pulp.lpSum(tasks[i]["demand"]*z(i,j,t) for i in range(len(tasks)) if j in Ji(tasks[i]))<=C
    for i in range(len(tasks)): m += pulp.lpSum(s[(i,j,t)] for j in Ji(tasks[i]) for t in Si(tasks[i]))<=1
    lf={n:[] for n in buses}; lt={n:[] for n in buses}
    for li,(a,b,x,F) in enumerate(lines): lf[a].append(li); lt[b].append(li)
    for n in buses:
        for t in TT:
            gen_n=pulp.lpSum(P[(g,t)] for g in gens if gens[g]["bus"]==n)
            dc_n =pulp.lpSum(L[(j,t)] for j in J if dc_buses[j]==n)
            m += (gen_n+(Pre[(n,t)]-pc[(n,t)])+ls[(n,t)]-D[(n,t)]-dc_n
                  == pulp.lpSum(fl[(li,t)] for li in lf[n])-pulp.lpSum(fl[(li,t)] for li in lt[n]))
    t0=time.time(); m.solve(SOLVER); st=time.time()-t0
    val=lambda x:(0.0 if pulp.value(x) is None else pulp.value(x))
    placements=[]
    for tk in tasks:
        i=tk["task_id"]; placed=None
        for j in Ji(tk):
            for t in Si(tk):
                if (i,j,t) in s and round(val(s[(i,j,t)]))==1:
                    placed=dict(task_id=i,start=t,dc=j,shifted=(t!=tk["release"]),migrated=(j!=tk["origin_dc"]))
        placements.append(placed or dict(task_id=i,dropped=True))
    grid_v=sum(gens[g]["cP"]*val(P[(g,t)])+gens[g]["cNL"]*val(u[(g,t)])+gens[g]["cSU"]*val(vv[(g,t)]) for g in gens for t in TT)
    drop_v=sum(tk["drop_penalty"] for tk,p in zip(tasks,placements) if p.get("dropped"))
    mig_v =sum(tk["mig_cost"] for tk,p in zip(tasks,placements) if p.get("migrated"))
    lol_v =sum(C_LOL*val(ls[(n,t)]) for n in buses for t in TT)

    # ============================ v2: dispatch extraction ============================
    # Per-gen optimal dispatch (MW) from the FULL solve. The 6 adjustable gens' values become
    # the SFT label; the other 67 are the fixed background exposed to the evaluator.
    P_opt={g:[round(val(P[(g,t)]),3) for t in TT] for g in gens}          # MW, all thermal gens
    def to_pct(g,mw):
        # Encoding: "off" = shut down (0 MW); float 0..100 = % of [Pmin,Pmax] when ON (0=Pmin,100=Pmax).
        # CONTINUOUS (no quantization) so the label can reach the exact optimum.
        # A valid MILP dispatch is either 0 (u=0) or >=Pmin (u=1), so the split is clean.
        # Reconstruction (evaluator): "off"->0 ; p -> Pmin + p/100*(Pmax-Pmin).
        G=gens[g]; rng=G["Pmax"]-G["Pmin"]
        if mw<G["Pmin"]-1e-6: return "off"                                # shut down
        if rng<=0: return 0.0
        frac=(mw-G["Pmin"])/rng*100.0
        return round(min(100.0,max(0.0,frac)),1)                          # continuous % (1 decimal)
    adj_set=set(ADJUSTABLE_GENS)
    adj_params=[dict(uid=g,bus=gens[g]["bus"],Pmin=gens[g]["Pmin"],Pmax=gens[g]["Pmax"],
                     cP=gens[g]["cP"],cNL=gens[g]["cNL"],cSU=gens[g]["cSU"],
                     is_swing=(g==SWING_GEN)) for g in ADJUSTABLE_GENS]
    fixed_gens=[g for g in gens if g not in adj_set]
    # per-bus fixed-gen injection (evaluator needs it for DC power flow; load/renewable are
    # recomputable from build_grid, so we only store the solve-dependent part here)
    bus_fixed_gen={}
    for g in fixed_gens:
        b=gens[g]["bus"]; bus_fixed_gen.setdefault(b,[0.0]*T)
        for ti in range(T): bus_fixed_gen[b][ti]+=P_opt[g][ti]
    bus_fixed_gen={b:[round(v,3) for v in arr] for b,arr in bus_fixed_gen.items()}
    # system-level grid signal (goes into the prompt)
    load_t =[round(sum(D[(b,t)]   for b in buses),1) for t in TT]
    renew_t=[round(sum(Pre[(b,t)] for b in buses),1) for t in TT]
    net_t  =[round(load_t[t-1]-renew_t[t-1],1) for t in TT]
    fixed_agg=[round(sum(bus_fixed_gen[b][t-1] for b in bus_fixed_gen),1) for t in TT]
    # adjustable optimal dispatch: MW + integer-% label
    adj_mw ={g:P_opt[g] for g in ADJUSTABLE_GENS}
    adj_pct={g:[to_pct(g,P_opt[g][t-1]) for t in TT] for g in ADJUSTABLE_GENS}

    iid=f"ECCS-RTS-m{month:02d}d{day:02d}-rho{int(rho*100):02d}-s{seed}"
    inst=dict(instance_id=iid,source="RTS-GMLC",month=month,day=day,horizon=T,penetration=rho,seed=seed,
        system_peak_day=round(syspeak_day,1),n_buses=len(buses),n_lines=len(lines),n_thermal=len(gens),
        dc_buses=dc_buses,dc=dict(Pmax=round(Pmax_dc,2),Pidle=round(Pidle,2),C=C,pi=round(pi,3)),
        n_tasks=len(tasks),tasks=tasks,split_tag=split_of(seed),
        # ---- v2 power co-design ----
        n_adjustable=len(ADJUSTABLE_GENS), swing_gen=SWING_GEN, pct_step=PCT_STEP,
        adjustable_gens=adj_params,                                  # 6 gen params (for prompt + eval)
        grid_signal=dict(load=load_t, renewable=renew_t, net=net_t,  # system-level (for prompt)
                         fixed_gen_agg=fixed_agg),
        bus_fixed_gen=bus_fixed_gen)                                 # per-bus fixed injection (for eval)
    sol=dict(instance_id=iid,status=pulp.LpStatus[m.status],solver=SOLVER_NAME,solve_time_s=round(st,1),
        mip_gap_target=MIP_GAP,objective=round(pulp.value(m.objective),1),
        cost_breakdown=dict(grid=round(grid_v,1),drop=round(drop_v,1),migration=round(mig_v,1),lost_load=round(lol_v,1)),
        n_placed=sum(1 for p in placements if not p.get("dropped")),n_dropped=sum(1 for p in placements if p.get("dropped")),
        n_shifted=sum(1 for p in placements if p.get("shifted")),n_migrated=sum(1 for p in placements if p.get("migrated")),
        task_placement=placements,
        # ---- v2 power co-design: dispatch label + gap baseline ----
        grid_cost_full=round(grid_v,1),                             # full-model grid cost = same-口径 gap baseline
        adjustable_dispatch_mw=adj_mw,                              # 6 gens optimal MW
        adjustable_dispatch_pct=adj_pct)                            # 6 gens optimal integer-% (SFT label)
    return inst,sol

# ============================ DRIVER (resumable) ============================
def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    # v2: dump the STATIC grid once (evaluator needs lines/ref_bus to compute DC power flow).
    json.dump(dict(ref_bus=ref_bus, buses=buses,
                   lines=[dict(idx=i,frm=a,to=b,x=x,limit=(F if F<LARGE else None))
                          for i,(a,b,x,F) in enumerate(lines)],
                   gen_bus={g:gens[g]["bus"] for g in gens}),
              open(f"{OUT_DIR}/network.json","w"), indent=1)
    grid=[(mo,da,rho,sd) for (mo,da) in DAYS for rho in RHOS for sd in SEEDS]
    N=len(grid)
    _ck=os.environ.get("ECCS_CHUNK"); _nck=os.environ.get("ECCS_NCHUNKS")
    if _ck is not None and _nck is not None:
        grid=grid[int(_ck)::int(_nck)]; N=len(grid)
        print(f"[chunk {_ck}/{_nck}] handling {N} instances")
    print(f"Solver: {SOLVER_NAME} | total instances: {N} | out: {OUT_DIR}/")
    # RESUME BY INSTANCE IDENTITY (iid), not by index -> safe when seeds/days change.
    # Recognizes BOTH old naming (instance_NNN__IID.json) and new (instance_IID.json).
    def iid_from_fname(f):
        base=f[:-5] if f.endswith(".json") else f          # strip .json
        base=base.split("__",1)[1] if "__" in base else base.split("_",1)[1]
        return base
    done_ids=set()
    for f in os.listdir(OUT_DIR):
        if f.startswith("instance_") and f.endswith(".json"):
            try: done_ids.add(iid_from_fname(f))
            except Exception: pass
    def make_iid(mo,da,rho,sd): return f"ECCS-RTS-m{mo:02d}d{da:02d}-rho{int(rho*100):02d}-s{sd}"
    t_all=time.time()
    for idx,(mo,da,rho,sd) in enumerate(grid,1):
        iid=make_iid(mo,da,rho,sd)
        if iid in done_ids:
            print(f"[{idx:04d}/{N}] skip {iid} (exists)"); continue
        # Solve each instance in a FRESH child process -> OS reclaims all memory after each,
        # so solve time stays flat instead of creeping up (no Gurobi model accumulation).
        try:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                            str(idx), str(mo), str(da), str(rho), str(sd), str(N)], check=False)
        except Exception as e:
            print(f"[{idx:04d}/{N}] DRIVER ERROR {iid}: {e}")
    # rebuild manifest from whatever is on disk (join instance+solution by instance_id)
    rows=[]
    insts=[json.load(open(f"{OUT_DIR}/{f}")) for f in os.listdir(OUT_DIR) if f.startswith("instance_") and f.endswith(".json")]
    sols={}
    for f in os.listdir(OUT_DIR):
        if f.startswith("solution_") and f.endswith(".json"):
            S=json.load(open(f"{OUT_DIR}/{f}")); sols[S["instance_id"]]=S
    for I in sorted(insts,key=lambda I:I["instance_id"]):
        S=sols.get(I["instance_id"],{})
        rows.append(dict(instance_id=I["instance_id"],month=I["month"],day=I["day"],penetration=I["penetration"],
            seed=I["seed"],split=I["split_tag"],status=S.get("status"),solve_s=S.get("solve_time_s"),
            objective=S.get("objective"),placed=S.get("n_placed"),dropped=S.get("n_dropped"),
            shifted=S.get("n_shifted"),migrated=S.get("n_migrated")))
    json.dump(rows,open(f"{OUT_DIR}/manifest.json","w"),indent=2)
    if rows:
        with open(f"{OUT_DIR}/manifest.csv","w",newline="") as f:
            wri=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
    print(f"\nDONE. {len(rows)} instances on disk in {time.time()-t_all:.0f}s. Manifest written.")

def _solve_one(idx,mo,da,rho,sd,N):
    iid=f"ECCS-RTS-m{mo:02d}d{da:02d}-rho{int(rho*100):02d}-s{sd}"
    try:
        inst,sol=solve(mo,da,rho,sd)
    except Exception as e:
        print(f"[{idx:04d}/{N}] FAILED {iid}: {e}"); return
    iid=inst["instance_id"]
    json.dump(inst,open(f"{OUT_DIR}/instance_{idx:04d}__{iid}.json","w"),indent=2)
    json.dump(sol ,open(f"{OUT_DIR}/solution_{idx:04d}__{iid}.json","w"),indent=2)
    print(f"[{idx:04d}/{N}] {iid} | {sol['status']} {sol['solve_time_s']}s | obj {sol['objective']:.0f} | "
          f"placed {sol['n_placed']} drop {sol['n_dropped']} shift {sol['n_shifted']} mig {sol['n_migrated']}",flush=True)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--one":
        # worker: solve ONE instance in this fresh process, then exit (memory reclaimed by OS)
        _,_,idx,mo,da,rho,sd,N=sys.argv
        os.makedirs(OUT_DIR,exist_ok=True)
        _solve_one(int(idx),int(mo),int(da),float(rho),int(sd),int(N))
    else:
        main()
