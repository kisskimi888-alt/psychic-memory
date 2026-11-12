"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""

from training import (
    GA_GENERATIONS,
    GA_POPULATION,
    LOADS,
    POLICY_CASES_PER_PASS,
    POLICY_PASSES,
    PPO_GATE_MARGIN,
    Path,
    RUN_DDQN_DIAGNOSTIC,
    fast_learned,
    fit,
    json,
    measures,
    np,
    pd,
    rule,
    synthetic,
    time,
)

from scheduling import (
    Patch,
    audit,
    genetic,
    plt,
    product,
    queue_metrics,
    schedule_frame,
)

import argparse
import matplotlib
matplotlib.use("Agg")
import training
from IPython.display import display
from replay import export_replay

def main():
    parser = argparse.ArgumentParser(description="门诊双智能体训练、留出集对照及互动回放")
    parser.add_argument("--scope", choices=["small", "6x6", "10x6", "all"], default="small")
    parser.add_argument("--profile", choices=["pilot", "paper_budget"], default="pilot")
    args = parser.parse_args()
    training.RUN_SCOPE = args.scope
    training.RUN_PROFILE = args.profile
    training.PPO_ROUNDS = 180 if args.profile == "pilot" else 10_000
    training.DDQN_EPISODES = 300 if args.profile == "pilot" else 3_000
    training.VALIDATION_CASES = 45 if args.profile == "pilot" else 100
    training.VALIDATE_EVERY = 30 if args.profile == "pilot" else 250
    training.SAVE_EVERY = 30 if args.profile == "pilot" else 100
    training.TEST_REPEATS = 3 if args.scope == "all" else 12
    RUN_SCOPE = training.RUN_SCOPE
    RUN_PROFILE = training.RUN_PROFILE
    PPO_ROUNDS = training.PPO_ROUNDS
    DDQN_EPISODES = training.DDQN_EPISODES
    VALIDATION_CASES = training.VALIDATION_CASES
    VALIDATE_EVERY = training.VALIDATE_EVERY
    SAVE_EVERY = training.SAVE_EVERY
    TEST_REPEATS = training.TEST_REPEATS
    import hashlib

    config=dict(scope=RUN_SCOPE,profile=RUN_PROFILE,ppo=PPO_ROUNDS,ddqn=DDQN_EPISODES,
                val=VALIDATION_CASES,validate_every=VALIDATE_EVERY,policy_passes=POLICY_PASSES,
                policy_cases=POLICY_CASES_PER_PASS,margin=PPO_GATE_MARGIN,
                diagnostic=RUN_DDQN_DIAGNOSTIC,ga=[GA_POPULATION,GA_GENERATIONS],repeats=TEST_REPEATS)
    run_key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:10]
    OUT=Path("hospital_latest_results")/f"{RUN_SCOPE}_{RUN_PROFILE}_{run_key}"
    OUT.mkdir(parents=True,exist_ok=True)
    RESULTS=OUT/"report"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"settings.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf8")
    sizes = {"6x6": (6, 6), "10x6": (10, 6), "15x15": (15, 15),
             "20x10": (20, 10), "20x15": (20, 15)}
    seeds = {"6x6": 314, "10x6": 318, "15x15": 315, "20x10": 316, "20x15": 317}
    small_models = ("6x6", "10x6")
    if RUN_SCOPE == "all":
        model_names = tuple(sizes)
    elif RUN_SCOPE == "small":
        model_names = small_models
    else:
        model_names = (RUN_SCOPE,)
    training.OUT = OUT
    models,history,decisions={},[],[]
    started=time.perf_counter()
    for label in model_names:
        n, m = sizes[label]
        ppo,ddqn,logs,decision=fit(label,n,m,seeds[label],PPO_ROUNDS,DDQN_EPISODES,
                                   VALIDATION_CASES,VALIDATE_EVERY,SAVE_EVERY,"window_aware")
        models[label]=(ppo,ddqn)
        history.append(logs)
        decisions.append(decision)
    history=pd.concat(history,ignore_index=True)
    selection=pd.DataFrame(decisions)
    history.to_csv(RESULTS/"training_log.csv",index=False)
    selection.to_csv(RESULTS/"validation_selection.csv",index=False)
    print(f"训练单元耗时 {time.perf_counter()-started:.1f} 秒（含检查点与验证）")
    display(selection.round(3))

    scales = [(n, m, k) for label in model_names if label in small_models
              for n, m in [sizes[label]] for k in (1, 2, 3)]
    if RUN_SCOPE == "all":
        scales += list(product((10, 15, 20), (6, 10, 15), (1, 2, 3)))
        scales += list(product((30, 50, 100), (20,), (1, 2, 3)))
    scales = list(dict.fromkeys(scales))
    transfer = {30: "15x15", 50: "20x10", 100: "20x15"}
    records=[]
    for n,m,k in scales:
        scale_label = f"{n}x{m}"
        model_name = scale_label if scale_label in small_models else transfer.get(n, "6x6")
        agent,window_agent=models[model_name]
        method_status=selection.set_index("model").loc[model_name,"ddqn_mode"]
        for load,rep in product(LOADS,range(TEST_REPEATS)):
            seed=5_200_000_000+(n*100+m)*100+rep
            problem=synthetic(n,m,k,seed,load)
            methods={name:lambda name=name:rule(problem,name) for name in ("READY","SPT","MWKR","MOPNR")}
            methods.update({"GA":lambda:genetic(problem,seed+77,GA_POPULATION,GA_GENERATIONS),
                            "PPO+ECT":lambda:fast_learned(problem,agent),
                            "PPO+ECT+压力平局":lambda:fast_learned(problem,agent,"pressure_tie")})
            if window_agent is not None:
                name="PPO+DDQN" if method_status=="正式" else "PPO+DDQN（诊断）"
                methods[name]=lambda:fast_learned(problem,agent,window_agent)
            order=list(methods) if rep%2 else list(methods)[::-1]
            for name in order:
                t0=time.perf_counter_ns()
                rows=methods[name]()
                ms=(time.perf_counter_ns()-t0)/1e6
                result=measures(problem,rows)
                assert result["feasible"], (n,m,k,load,seed,name,audit(problem,rows))
                records.append(dict(patients=n,activities=m,windows=k,load=load,rep=rep,seed=seed,
                                    method=name,model=model_name,ms=ms,**result,**queue_metrics(problem,rows)))
    raw=pd.DataFrame(records)
    keys=["patients","activities","windows","load","rep"]
    assert raw.feasible.all() and not raw.duplicated(keys+["method"]).any()
    assert raw[keys].drop_duplicates().shape[0]==len(scales)*len(LOADS)*TEST_REPEATS
    raw.to_csv(RESULTS/"results_by_instance.csv",index=False)
    aggregation=dict(病例数=("mean_flow","size"),平均在院分钟每人=("mean_flow","mean"),
                     P90等待分钟=("p90_wait","mean"),排程中位毫秒=("ms","median"))
    summary=raw.groupby(["patients","activities","method"],sort=False).agg(**aggregation).reset_index()
    summary.to_csv(RESULTS/"results_by_size.csv",index=False)
    raw.groupby(["patients","activities","windows","load","method"]).agg(**aggregation).to_csv(RESULTS/"results_by_load_windows.csv")
    display(summary.round(3))
    print(f"{raw[keys].drop_duplicates().shape[0]} 批留出病例、{len(raw)} 次排程通过可行性审计。")

    paired=raw.pivot(index=keys,columns="method",values="mean_flow")
    comparisons=[]
    for scope, view in [("全部",paired),("多窗口",paired[paired.index.get_level_values("windows")>1])]:
        for (n,m),g in view.groupby(level=["patients","activities"]):
            for chosen,ref in [("PPO+ECT","READY"),("PPO+ECT","GA"),
                               ("PPO+ECT+压力平局","PPO+ECT"),("PPO+DDQN","PPO+ECT"),
                               ("PPO+DDQN","GA"),("PPO+DDQN（诊断）","PPO+ECT"),
                               ("PPO+DDQN（诊断）","READY"),("PPO+DDQN（诊断）","GA")]:
                if chosen not in g or ref not in g:
                    continue
                d=(g[chosen]-g[ref]).dropna()
                if d.empty:
                    continue
                comparisons.append(dict(规模=f"{n}×{m}",范围=scope,方案=chosen,基线=ref,配对病例=len(d),
                                        平均差分钟每人=d.mean(),胜=int((d<-1e-9).sum()),
                                        平=int((d.abs()<=1e-9).sum()),负=int((d>1e-9).sum())))
    paired_summary=pd.DataFrame(comparisons)
    paired_summary.to_csv(RESULTS/"paired_comparisons.csv",index=False)
    display(paired_summary.round(3))

    demo_replays = {}
    for label in small_models:
        if label not in models:
            continue
        folder = RESULTS if label == "6x6" else RESULTS / label
        folder.mkdir(exist_ok=True)
        n, m = sizes[label]
        demo_seed = 5_209_001_001 + n - 6
        demo = synthetic(n, m, 2, demo_seed, "near_capacity")
        agent, desk_agent = models[label]
        print(f"{n}×{m} 排程示例")
        solutions = {"READY":rule(demo,"READY"), "PPO+ECT":fast_learned(demo,agent)}
        focus="PPO+ECT"
        if desk_agent is not None:
            focus="PPO+DDQN" if selection.set_index("model").loc[label,"ddqn_mode"]=="正式" else "PPO+DDQN（诊断）"
            solutions[focus]=fast_learned(demo,agent,desk_agent)
        display(pd.DataFrame({label:measures(demo,rows) for label,rows in solutions.items()}).T)
        for method, rows in solutions.items():
            schedule_frame(demo,rows).to_csv(folder/f"demo_{method}.csv",index=False)
        display(schedule_frame(demo,solutions[focus]).head(12))

        palette = plt.get_cmap("tab10")
        fig, axes = plt.subplots(len(solutions), 1, figsize=(13, 11), sharex=True, layout="constrained")
        desk_pos = {name: i for i, name in enumerate(demo.desks)}
        for ax, (method, rows) in zip(axes, solutions.items()):
            for b in rows:
                patient = demo.tasks[b.task].patient
                ax.barh(desk_pos[b.desk], b.end-b.start, left=b.start, height=.75,
                        color=palette(patient), edgecolor="white", linewidth=.5)
            ax.set_yticks(range(len(demo.desks)), demo.desks, fontsize=8)
            ax.set_title(method.replace("（诊断）", " (diagnostic)"), loc="left", fontweight="bold")
            ax.grid(axis="x", alpha=.25)
        axes[0].legend(handles=[Patch(color=palette(i), label=f"P{i+1}") for i in range(len(demo.case.arrival))],
                       loc="upper right", ncol=min(10, n), fontsize=8)
        axes[-1].set_xlabel("Minutes from opening")
        fig.suptitle(f"{n} x {m}: same demo patients, matched scheduling policies", fontweight="bold")
        fig.savefig(folder / "demo_gantt.png", dpi=160, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        demo_replays[label] = demo, solutions[focus], folder


    def replay_states(p, rows):
        finish = {i: max(b.end for b in rows if p.tasks[b.task].patient == i)
                  for i in range(len(p.case.arrival))}
        times = sorted(set(p.case.arrival) | {b.start for b in rows} | {b.end for b in rows})
        timeline = []
        for t in times:
            arrived = sum(a <= t for a in p.case.arrival)
            busy = [b for b in rows if b.start <= t < b.end]
            done = sum(e <= t for e in finish.values())
            waiting = arrived-len(busy)-done
            assert waiting >= 0
            timeline.append((t, arrived, len(busy), waiting, done,
                             ", ".join(f"P{p.tasks[b.task].patient+1}@{b.desk}" for b in busy)))
        return pd.DataFrame(timeline, columns=["Minute", "Arrived", "In service", "Waiting",
                                               "Completed", "Active bookings"])

    for label, (problem, rows, folder) in demo_replays.items():
        print(f"{label} 事件回放")
        timeline = replay_states(problem, rows)
        timeline.to_csv(folder / "demo_event_replay.csv", index=False)
        assert timeline.iloc[-1]["Completed"] == len(problem.case.arrival)
        display(timeline.iloc[np.unique(np.linspace(0, len(timeline)-1, 9).astype(int))])

    stages=[name for name in ("PPO","自身排程排序","DDQN") if name in set(history.stage)]
    fig,axes=plt.subplots(1,len(stages),figsize=(5.4*len(stages),3.6),layout="constrained",squeeze=False)
    for ax,stage in zip(axes[0],stages):
        for label,rows in history[history.stage.eq(stage)].groupby("model",sort=False):
            ax.plot(rows.step,rows.validation_flow,"o-",label=label)
        ax.set(xlabel="Training step",ylabel="Validation mean flow (min)",
               title={"PPO":"PPO","自身排程排序":"On-policy action ranking","DDQN":"DDQN"}[stage])
        ax.grid(alpha=.25)
        ax.legend(title="Training scale")
    fig.savefig(RESULTS/"validation_curves.png",dpi=150,bbox_inches="tight")
    display(fig)
    plt.close(fig)
    print("表格、图和逐单结果目录",RESULTS)
    export_replay(models, selection, raw, sizes, seeds, small_models, OUT, RESULTS)
    print("实验与互动回放已导出", RESULTS)


if __name__ == "__main__":
    main()
