"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""

from training import (
    DDQN_EPISODES,
    GA_GENERATIONS,
    GA_POPULATION,
    PPO_ROUNDS,
    SAVE_EVERY,
    VALIDATE_EVERY,
    VALIDATION_CASES,
    fast_learned,
    json,
    measures,
    np,
    pd,
    rule,
    synthetic,
    time,
)

from scheduling import (
    audit,
    genetic,
    replay_states,
)

import training
from IPython.display import display

def export_replay(models, selection, raw, sizes, seeds, small_models, output_dir, results_dir):
    """生成离线 HTML 回放，并逐事件校验导出状态。"""
    OUT, RESULTS = output_dir, results_dir
    RUN_DDQN_DIAGNOSTIC = training.RUN_DDQN_DIAGNOSTIC

    def fit(*args, **kwargs):
        old_out, old_mode = training.OUT, training.RUN_DDQN_DIAGNOSTIC
        try:
            training.OUT = OUT
            training.RUN_DDQN_DIAGNOSTIC = RUN_DDQN_DIAGNOSTIC
            return training.fit(*args, **kwargs)
        finally:
            training.OUT, training.RUN_DDQN_DIAGNOSTIC = old_out, old_mode

    # 展示用病例与训练集、验证集和正式留出集分开。
    # 原来的默认门槛照常生效。若正式结果没有 DDQN 权重，单独训练展示用诊断权重。
    replay_models, replay_modes = {}, {}
    mode_by_model = selection.set_index("model")["ddqn_mode"]
    for model_label in small_models:
        if model_label not in models:
            continue
        replay_ppo, replay_ddqn = models[model_label]
        replay_mode = mode_by_model.loc[model_label]
        if replay_ddqn is None:
            experiment_out, experiment_diagnostic = OUT, RUN_DDQN_DIAGNOSTIC
            try:
                OUT = experiment_out / "replay_diagnostics"
                OUT.mkdir(exist_ok=True)
                RUN_DDQN_DIAGNOSTIC = True
                n, m = sizes[model_label]
                diagnostic_ppo, replay_ddqn, _, replay_fit = fit(
                    model_label, n, m, seeds[model_label], PPO_ROUNDS, DDQN_EPISODES,
                    VALIDATION_CASES, VALIDATE_EVERY, SAVE_EVERY, "window_aware"
                )
            finally:
                OUT, RUN_DDQN_DIAGNOSTIC = experiment_out, experiment_diagnostic
            assert all(np.array_equal(a, b) for a, b in zip(
                replay_ppo.parameters, diagnostic_ppo.parameters, strict=True
            )), "诊断应使用与正文相同的患者调度权重"
            assert replay_fit["ddqn_mode"] == "诊断"
            replay_mode = "诊断"
        assert replay_ddqn is not None
        replay_models[model_label] = (replay_ppo, replay_ddqn)
        replay_modes[model_label] = replay_mode

    def encode_replay(problem, rows, elapsed_ms):
        errors = audit(problem, rows)
        assert not errors, errors
        by_task = {booking.task: booking for booking in rows}
        patients, events = [], []
        for index, arrival in enumerate(problem.case.arrival):
            patient_id = f"P{index+1:02d}"
            ready = arrival
            events.append(dict(t=arrival, type="arrival", patient=patient_id))
            steps = []
            for booking in sorted((by_task[i] for i in problem.by_patient[index]),
                                  key=lambda b: b.start):
                activity = problem.tasks[booking.task].kind
                step = dict(activity=activity, desk=booking.desk, ready=ready,
                            start=booking.start, end=booking.end)
                assert ready <= booking.start < booking.end
                steps.append(step)
                fields = dict(patient=patient_id, activity=activity, desk=booking.desk)
                if ready < booking.start:
                    events.append(dict(t=ready, type="queue", **fields))
                events.append(dict(t=booking.start, type="start", **fields))
                events.append(dict(t=booking.end, type="end", **fields))
                ready = booking.end
            flow = ready-arrival
            wait = sum(s["start"]-s["ready"] for s in steps)
            service = sum(s["end"]-s["start"] for s in steps)
            assert wait + service == flow
            patients.append(dict(id=patient_id, arrival=arrival, finish=ready,
                                 flow=flow, wait=wait, service=service, steps=steps))
            events.append(dict(t=ready, type="departure", patient=patient_id))

        event_order = {event: i for i, event in enumerate(
            ("end", "departure", "arrival", "queue", "start")
        )}
        events.sort(key=lambda event: (event["t"], event_order[event["type"]], event["patient"]))
        stats = measures(problem, rows)
        assert stats["total_flow"] == sum(patient["flow"] for patient in patients)
        assert np.isclose(stats["p90_wait"], np.quantile([p["wait"] for p in patients], 0.9))
        metrics = dict(flow=stats["mean_flow"], wait=sum(p["wait"] for p in patients)/len(patients),
                       p90=stats["p90_wait"], ms=elapsed_ms, finish=max(p["finish"] for p in patients))

        # 核对导出的患者状态与原回放函数在每个事件时刻的统计。
        for minute, arrived, busy, waiting, completed, _ in replay_states(problem, rows).itertuples(index=False, name=None):
            current = [
                "future" if minute < p["arrival"] else "completed" if minute >= p["finish"]
                else "busy" if any(s["start"] <= minute < s["end"] for s in p["steps"])
                else "waiting"
                for p in patients
            ]
            assert arrived == len(patients) - current.count("future")
            assert (busy, waiting, completed) == tuple(current.count(s) for s in ("busy", "waiting", "completed"))
        return dict(patients=patients, events=events, metrics=metrics)

    class AuditedDeskPolicy:
        """记录实际调用分台策略时的候选数，并检查输出是否合法。"""
        def __init__(self, model):
            self.model = model
            self.candidate_counts = []

        def choose(self, state):
            count = int(state[1])
            choice = int(self.model.choose(state))
            assert count > 1 and 0 <= choice < count
            self.candidate_counts.append(count)
            return choice

    replay_cases, replay_summary = {}, []
    for size, case_seed in ((6, 5_209_001_001), (10, 5_209_001_005)):
        model_label = f"{size}x6"
        if model_label not in replay_models:
            continue
        label = f"{size}×6"
        replay_ppo, replay_ddqn = replay_models[model_label]
        replay_dual_label = "PPO+DDQN（诊断）" if replay_modes[model_label] == "诊断" else "PPO+DDQN"
        variants = {}
        reference = synthetic(size, 6, 1, case_seed, "near_capacity")
        for windows in (1, 2, 3):
            problem = synthetic(size, 6, windows, case_seed, "near_capacity")
            assert problem.case.arrival == reference.case.arrival
            assert [(t.patient,t.kind,t.before,t.options[0]) for t in problem.tasks] == [
                (t.patient,t.kind,t.before,t.options[0]) for t in reference.tasks
            ]
            assert all(len(t.options) == (1 if t.kind in ("Consult", "Follow") else windows)
                       for t in problem.tasks)
            desk_policy = AuditedDeskPolicy(replay_ddqn)
            schedulers = {
                "READY": lambda: rule(problem, "READY"),
                "GA": lambda: genetic(problem, case_seed+77, GA_POPULATION, GA_GENERATIONS),
                "PPO+ECT": lambda: fast_learned(problem, replay_ppo),
                "PPO+DDQN": lambda: fast_learned(problem, replay_ppo, desk_policy),
            }
            plans = {}
            for method, schedule in schedulers.items():
                started = time.perf_counter_ns()
                bookings = schedule()
                elapsed_ms = (time.perf_counter_ns()-started)/1e6
                plan = encode_replay(problem, bookings, elapsed_ms)
                plans[method] = plan
                metrics = plan["metrics"]
                replay_summary.append({
                    "规模": label, "检查窗口/类": windows,
                    "策略": replay_dual_label if method == "PPO+DDQN" else method,
                    "DDQN分台次数": len(desk_policy.candidate_counts) if method == "PPO+DDQN" else None,
                    "活动数": problem.n, "事件数": len(plan["events"]),
                    "平均等待（分钟/人）": metrics["wait"], "平均在院（分钟/人）": metrics["flow"],
                    "P90等待（分钟）": metrics["p90"], "单次排程（毫秒）": metrics["ms"],
                })
            assert sorted(desk_policy.candidate_counts) == sorted(
                len(t.options) for t in problem.tasks if len(t.options) > 1
            )
            assert all([p["arrival"] for p in plan["patients"]] == list(problem.case.arrival)
                       for plan in plans.values())
            variants[windows] = dict(
                seed=case_seed, desks=problem.desks, plans=plans, dualLabel=replay_dual_label,
                ddqnDecisions=len(desk_policy.candidate_counts),
                finish=max(plan["metrics"]["finish"] for plan in plans.values())
            )
        replay_cases[label] = variants

    replay_data = dict(cases=replay_cases)
    replay_summary_frame = pd.DataFrame(replay_summary)
    display(replay_summary_frame.round(3))
    # 正文留出集按窗口数再拆一层，单独检验 3 窗口时 DDQN 的结果。
    replay_grouped = raw[raw["method"].isin(
        ["READY", "GA", "PPO+ECT", "PPO+DDQN", "PPO+DDQN（诊断）"]
    ) & raw["patients"].isin([6, 10]) & raw["activities"].eq(6)]
    display(replay_grouped.pivot_table(index=["patients", "windows"], columns="method",
                                    values="mean_flow").round(3))
    print(f"回放验收通过：{len(replay_cases)} 个规模、{len(replay_summary)} 份排程。"
          "患者和窗口状态已逐事件核对，DDQN 的分台次数与多窗口活动数完全一致。")


    replay_template = r"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>门诊调度回放台</title>
    <style>
    :root{color-scheme:light;--ink:#15302e;--muted:#5b6c68;--line:#dbe3dc;--paper:#fff;--base:#f3f6f1;--forest:#123d34;--green:#186b52;--mint:#d9efe5;--warm:#fff0d9;--amber:#8a5114;--radius:15px;--shadow:0 3px 12px #244a3110}
    *{box-sizing:border-box}
    body{margin:0;background:var(--base);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei","PingFang SC",sans-serif}
    button,input,select{font:inherit}
    button,select{cursor:pointer}
    button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #1a875d;outline-offset:3px}
    button:disabled{cursor:default;opacity:.45}
    .wrap{max-width:1390px;margin:auto;padding:22px 24px 32px}
    .hero{background:var(--forest);color:#f7fbf5;padding:24px 26px;border-radius:19px;display:flex;align-items:center;justify-content:space-between;gap:18px}
    .eyebrow{color:#b7e5d2;font-size:12px;letter-spacing:.12em;margin:0 0 4px;font-weight:750}
    h1{font-size:clamp(26px,4vw,38px);margin:0;font-weight:770;line-height:1.18}
    .hero p:last-child{color:#d2e8dc;margin:8px 0 0}
    .hero-tag{font-size:12px;border:1px solid #76a597;border-radius:20px;padding:5px 12px;white-space:nowrap;color:#e2f6eb}
    .card{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow);min-width:0}
    .controls{margin-top:14px}
    .select-row,.control-row,.header-row,.clock-row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
    .select-row{gap:12px;justify-content:space-between;margin-bottom:12px}
    .select-group{display:flex;flex-wrap:wrap;gap:10px}
    .field{display:flex;gap:7px;align-items:center;color:var(--muted);font-weight:650}
    select{border:1px solid #bdcfc3;background:white;color:var(--ink);border-radius:8px;padding:7px 9px;max-width:100%}
    .hint{color:var(--muted);font-size:12px;margin:0}
    .control-row{gap:7px}
    .ctrl{border:1px solid #c2d3ca;background:#fff;color:var(--ink);border-radius:8px;padding:7px 10px;font-weight:660}
    .ctrl.primary{background:var(--green);color:white;border-color:var(--green);min-width:70px}
    .clock-row{margin-left:auto;gap:9px;color:var(--muted)}
    #clock{font:750 28px/1.15 ui-monospace,monospace;color:var(--forest);font-variant-numeric:tabular-nums}
    .seek-row{display:flex;gap:10px;align-items:center;margin-top:8px;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
    #seek{flex:1;min-width:60px;accent-color:var(--green)}
    #next-event{padding-top:4px;min-height:22px}
    .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
    .stat{padding:10px 15px;background:#fff;border:1px solid var(--line);border-radius:11px}
    .stat strong{display:block;font:760 24px/1.2 ui-monospace,monospace;font-variant-numeric:tabular-nums}
    .stat span{color:var(--muted);font-size:12px}
    .stat.wait{background:var(--warm);border-color:#ecd9ba}
    .stat.busy{background:#e7f2ec}
    .main-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:12px;align-items:start}
    .stack{display:grid;gap:12px;min-width:0}
    .header-row{justify-content:space-between;margin-bottom:9px}
    h2{font-size:18px;margin:0;line-height:1.3}
    h3{font-size:14px;margin:0}
    .patient-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}
    .patient,.person-link{color:var(--ink);background:var(--pc,#edf4ef);border:1px solid #acbdb5;border-radius:8px}
    .patient{padding:7px 5px;text-align:left;min-width:0}
    .patient b,.patient small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .patient small{color:#3f554e;font-size:11px}
    .patient[aria-pressed=true]{box-shadow:inset 0 0 0 2px #1a664f;border-color:#1a664f}
    .person-link{font-weight:700;font-size:12px;padding:1px 5px}
    .windows{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
    .desk{border:1px solid var(--line);border-radius:9px;padding:8px;min-width:0;background:#fafcf9}
    .desk.active{background:#edf8f2;border-color:#b9daca}
    .desk-title{display:flex;flex-wrap:wrap;gap:3px 6px;justify-content:space-between;align-items:center;font-weight:750}
    .desk-current,.desk-queue{color:var(--muted);font-size:12px;margin-top:4px;min-height:19px;overflow-wrap:anywhere}
    .desk-queue button{margin:0 2px 2px 0}
    .badge{color:var(--green);background:var(--mint);border-radius:12px;padding:2px 7px;font-size:11px;font-weight:700;white-space:nowrap}
    .badge.wait{background:var(--warm);color:var(--amber)}
    .badge.off{background:#edf1ed;color:#617069}
    .patient-info{display:flex;flex-wrap:wrap;gap:7px 14px;color:var(--muted);margin:0 0 8px;font-size:12px}
    .patient-info b{color:var(--ink)}
    .table-box{overflow:auto}
    table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
    th,td{text-align:left;white-space:nowrap;padding:7px 6px;border-bottom:1px solid #e5eae4}
    th{color:var(--muted);font-weight:730;background:#f7faf6}
    tr.now{background:#e9f7ee}
    tr.plan{color:var(--muted)}
    tr.chosen{background:#e8f5ed;font-weight:700}
    tr.baseline td:first-child{border-left:3px solid #d5a45c}
    td.num,th.num{text-align:right}
    .event-options{font-size:12px;color:var(--muted)}
    .event-list{list-style:none;padding:0;margin:0;max-height:264px;overflow:auto}
    .event-list li{display:grid;grid-template-columns:43px 48px minmax(0,1fr);gap:6px;align-items:start;padding:5px 1px;border-bottom:1px solid #e8ede7}
    .event-list li.empty{display:block;color:var(--muted)}
    .time-link{background:transparent;border:0;color:var(--green);font:700 12px ui-monospace,monospace;padding:4px 0;text-align:left}
    .event-list li.current{background:#edf7ee}
    .event-list li span{padding-top:1px}
    .wide{margin-top:12px}
    .gantt-scroll{overflow:auto;padding-top:5px}
    .gantt-grid{display:grid;grid-template-columns:126px minmax(660px,1fr);min-width:790px;align-items:stretch}
    .gantt-label{height:28px;display:flex;align-items:center;border-bottom:1px solid #e7ece6;color:var(--muted);font-size:12px}
    .gantt-body{position:relative}
    .lane{height:28px;position:relative;border-bottom:1px solid #e7ece6;background:linear-gradient(to right,#e6ede6 1px,transparent 1px) left top/var(--tick-step) 100%}
    .gantt-task{position:absolute;height:19px;top:4px;border:1px solid #58786b;border-radius:4px;min-width:3px;padding:0;background:var(--pc);color:#183931;font-size:10px;font-weight:740;white-space:nowrap;overflow:hidden}
    .gantt-task.other{opacity:.65}
    .gantt-task.selected{z-index:1;border:2px solid #123d34}
    .gantt-task:focus-visible{z-index:5}
    .playhead{pointer-events:none;position:absolute;top:0;bottom:0;width:2px;background:#b45120;z-index:2}
    .gantt-axis{height:23px;position:relative}
    .gantt-axis span{font:11px ui-monospace,monospace;color:var(--muted);position:absolute;transform:translateX(-35%);top:5px;white-space:nowrap}
    .metric-note{margin-top:6px}
    footer{color:var(--muted);font-size:12px;margin-top:15px}
    @media(max-width:1020px){.windows{grid-template-columns:repeat(2,minmax(0,1fr))}.patient-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}
    @media(max-width:760px){.wrap{padding:10px 10px 22px}.hero{padding:17px;align-items:flex-start;flex-direction:column;gap:10px}.main-grid{grid-template-columns:1fr}.select-row{gap:7px}.stats{gap:6px}.stat{padding:8px}.clock-row{margin-left:0}.card{padding:12px}}
    @media(max-width:420px){.field{flex-direction:column;align-items:flex-start;gap:2px}.patient-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.stat strong{font-size:20px}}
    </style>
    </head>
    <body>
    <div class="wrap" id="app">
    <header class="hero">
    <div><p class="eyebrow">门诊多窗口调度仿真</p><h1>门诊调度回放台</h1>
    <p>沿着同一批患者的时间线，查看候诊、分台和离院。</p></div>
    <div class="hero-tag">合成病例 · 离线演示</div>
    </header>
    <section class="card controls" aria-label="回放控制">
      <div class="select-row">
        <div class="select-group">
          <label class="field">病例规模 <select id="case"></select></label>
          <label class="field">每类检查窗口 <select id="window-count"><option value="1">1 个</option><option value="2">2 个</option><option value="3" selected>3 个</option></select></label>
          <label class="field">回放策略 <select id="method"></select></label>
          <label class="field">对照策略 <select id="baseline"></select></label>
        </div>
        <p class="hint" id="case-note"></p>
      </div>
      <div class="control-row">
        <button type="button" class="ctrl primary" id="play">播放</button>
        <button type="button" class="ctrl" id="previous">上一事件</button>
        <button type="button" class="ctrl" id="next">下一事件</button>
        <button type="button" class="ctrl" id="reset">回到开诊</button>
        <label class="field">倍速 <select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
        <div class="clock-row"><span>模拟时钟</span><strong id="clock">09:00</strong></div>
      </div>
      <div class="seek-row"><span>09:00</span><input id="seek" aria-label="模拟时刻" type="range" min="0" step="1"><span id="end-time"></span></div>
      <p id="next-event" class="hint"></p>
    </section>
    <section class="stats" aria-label="患者状态统计">
      <div class="stat"><strong id="arrived">0</strong><span>累计到达</span></div>
      <div class="stat wait"><strong id="waiting">0</strong><span>正在候诊</span></div>
      <div class="stat busy"><strong id="busy">0</strong><span>正在服务</span></div>
      <div class="stat"><strong id="completed">0</strong><span>已离院</span></div>
    </section>
    <div class="main-grid">
    <div class="stack">
    <section class="card">
    <div class="header-row"><h2>患者列表</h2><p class="hint">点击查看完整轨迹</p></div>
    <div id="patient-list" class="patient-grid"></div>
    </section>
    <section class="card">
    <div class="header-row"><h2>窗口状态</h2><p class="hint">候诊按该策略已分配的窗口统计</p></div>
    <div id="windows" class="windows"></div>
    </section>
    </div>
    <div class="stack">
    <section class="card">
    <div class="header-row"><h2 id="patient-title">患者轨迹</h2><span class="badge" id="patient-status"></span></div>
    <p id="patient-info" class="patient-info"></p>
    <div class="table-box"><table aria-label="患者活动时间线"><thead><tr><th>活动</th><th>窗口</th><th>就绪</th><th>开始–结束</th><th class="num">等待</th><th class="num">服务</th></tr></thead><tbody id="patient-steps"></tbody></table></div>
    <p id="patient-compare" class="hint metric-note"></p>
    </section>
    <section class="card">
    <div class="header-row"><h2>事件回放</h2><label class="event-options"><input id="filter-patient" type="checkbox"> 只看所选患者</label></div>
    <ul id="events" class="event-list" aria-label="截至当前时刻的事件"></ul>
    </section>
    </div>
    </div>
    <section class="card wide">
    <div class="header-row"><h2>窗口甘特图</h2><p class="hint">色块代表服务时段，竖线标记当前时刻。点击色块选中患者。</p></div>
    <div id="gantt" class="gantt-scroll"></div>
    </section>
    <section class="card wide">
    <div class="header-row"><h2>同一批患者的策略对照</h2><p class="hint" id="comparison-note"></p></div>
    <div class="table-box"><table aria-label="策略对照"><thead><tr><th>策略</th><th class="num">平均等待（分/人）</th><th class="num">平均在院（分/人）</th><th class="num">P90 等待（分）</th><th class="num">排程（毫秒）</th></tr></thead><tbody id="comparison"></tbody></table></div>
    <p class="hint metric-note">P90 表示约九成患者的累计等待时间在这个数值以内。排程耗时是这一个病例的一次计算时间。</p>
    </section>
    <footer>回放使用离线仿真排程。跨病例结论以 Notebook 的留出集评测为准。</footer>
    </div>
    <script id="replay-data" type="application/json">__REPLAY_PAYLOAD__</script>
    <script>
    "use strict";
    const data = JSON.parse(document.getElementById("replay-data").textContent);
    const cases = data.cases;
    const $ = id => document.getElementById(id);
    const state = {scenario: cases["10×6"] ? "10×6" : Object.keys(cases)[0],
      windows:3, method:"READY", baseline:"READY", patient:"P01", minute:0, timer:null};
    const colors = ["#bfe1d1","#ffdbb6","#c8daf9","#e0c8f2","#f4ccc7",
      "#f0e1a8","#c7e4a9","#c7e8e8","#d5d7f6","#ecd3ae"];
    const activities = {Reg:"挂号", Consult:"初诊", Follow:"回诊", Rx:"取药"};
    const statusLabels = {future:"未到达", waiting:"候诊", busy:"服务中", completed:"已离院"};
    const eventLabels = {arrival:"到达",queue:"开始等候",start:"开始服务",end:"完成服务",departure:"离院"};
    const scenario = () => cases[state.scenario][state.windows];
    const plan = () => scenario().plans[state.method];
    const methodName = key => key === "PPO+DDQN" ? scenario().dualLabel : key;
    const color = id => colors[(Number(id.slice(1))-1) % colors.length];
    const activityName = key => activities[key] || key.replace("Exam-", "检查 ");
    function deskName(key) {
      const [kind, index] = key.split("/");
      return activityName(kind) + " " + index + " 号窗";
    }
    function timeText(minute) {
      const total = 540 + minute;
      return [Math.floor(total/60), total%60].map(v => String(v).padStart(2,"0")).join(":");
    }
    function patientState(patient, minute) {
      if (minute < patient.arrival) return {type:"future"};
      if (minute >= patient.finish) return {type:"completed"};
      const step = patient.steps.find(s => minute < s.end);
      return {type:minute < step.start ? "waiting" : "busy", step};
    }
    function personButton(id) {
      return `<button type="button" class="person-link" data-patient="${id}" style="--pc:${color(id)}">${id}</button>`;
    }
    function badgeClass(type) { return type === "waiting" ? " wait" : (type === "future" ? " off" : ""); }
    function renderPatients(live) {
      $("patient-list").innerHTML = live.map(({patient:p, current:s}) =>
        `<button type="button" class="patient" data-patient="${p.id}" style="--pc:${color(p.id)}" aria-pressed="${p.id === state.patient}">
        <b>${p.id}</b><small>${statusLabels[s.type]}${s.step ? " · "+activityName(s.step.activity) : ""}</small></button>`).join("");
    }
    function renderWindows(live) {
      const byDesk = new Map(scenario().desks.map(key => [key, {busy:null, queue:[]}]));
      for (const {patient, current} of live) {
        if (!current.step) continue;
        const desk = byDesk.get(current.step.desk);
        if (current.type === "busy") desk.busy = patient.id;
        else desk.queue.push({id:patient.id, ready:current.step.ready});
      }
      $("windows").innerHTML = Array.from(byDesk, ([desk, status]) => {
        const queue = status.queue.sort((a,b) => a.ready-b.ready || a.id.localeCompare(b.id));
        return `<article class="desk ${status.busy ? "active" : ""}" data-desk="${desk}">
          <div class="desk-title">${deskName(desk)} <span class="badge ${status.busy ? "" : "off"}">${status.busy ? "服务中" : "空闲"}</span></div>
          <div class="desk-current">${status.busy ? "当前 "+personButton(status.busy) : "当前 · —"}</div>
          <div class="desk-queue">候诊 ${queue.length} 人 ${queue.map(x => personButton(x.id)).join("")}</div>
          </article>`;
      }).join("");
    }
    function signed(value) { return value > 0 ? "+" + value : String(value); }
    function renderPatient(live) {
      const {patient:p, current:s} = live.find(x => x.patient.id === state.patient);
      const base = scenario().plans[state.baseline].patients.find(x => x.id === p.id);
      $("patient-title").textContent = p.id + " 患者轨迹";
      $("patient-status").textContent = statusLabels[s.type];
      $("patient-status").className = "badge" + badgeClass(s.type);
      $("patient-info").innerHTML =
        `<span>到达 <b>${timeText(p.arrival)}</b></span><span>离院 <b>${timeText(p.finish)}</b></span>
        <span>累计等待 <b>${p.wait} 分</b></span><span>在院 <b>${p.flow} 分</b></span>`;
      $("patient-steps").innerHTML = p.steps.map(step => {
        const row = state.minute >= step.ready && state.minute < step.end ? "now" : (state.minute < step.ready ? "plan" : "");
        return `<tr class="${row}"><td>${activityName(step.activity)}</td><td>${deskName(step.desk)}</td><td>${timeText(step.ready)}</td>
          <td>${timeText(step.start)}–${timeText(step.end)}</td><td class="num">${step.start-step.ready} 分</td><td class="num">${step.end-step.start} 分</td></tr>`;
      }).join("");
      $("patient-compare").textContent = `与 ${methodName(state.baseline)} 相比，这位患者等待 ${signed(p.wait-base.wait)} 分钟，在院 ${signed(p.flow-base.flow)} 分钟。`;
    }
    function eventText(event) {
      return eventLabels[event.type] + (event.desk ? " · " + deskName(event.desk) : "");
    }
    function renderEvents() {
      const events = plan().events;
      const next = events.find(event => event.t > state.minute);
      $("next-event").textContent = next ? `下一事件 ${timeText(next.t)} · ${next.patient} ${eventText(next)}` : "当前策略的全部事件已播放";
      const filtered = events.filter(e => e.t <= state.minute && (!$("filter-patient").checked || e.patient === state.patient));
      $("events").innerHTML = filtered.slice(-30).reverse().map(event =>
        `<li class="${event.t === state.minute ? "current" : ""}"><button type="button" class="time-link" data-minute="${event.t}">${timeText(event.t)}</button>
         ${personButton(event.patient)}<span>${eventText(event)}</span></li>`).join("") ||
        '<li class="empty">这个时刻还没有可显示的事件。</li>';
    }
    function renderGantt() {
      const end = scenario().finish;
      const tick = end > 135 ? 30 : 20;
      const step = tick/end*100;
      const lanes = new Map(scenario().desks.map(d => [d, []]));
      for (const p of plan().patients) for (const s of p.steps) lanes.get(s.desk).push({p,s});
      const labels = Array.from(lanes.keys(), d => `<div class="gantt-label">${deskName(d)}</div>`).join("");
      const slots = Array.from(lanes.values(), bookings => `<div class="lane">${bookings.map(({p,s}) =>
        `<button type="button" class="gantt-task ${p.id === state.patient ? "selected" : "other"}" data-patient="${p.id}"
          style="left:${s.start/end*100}%;width:${(s.end-s.start)/end*100}%;--pc:${color(p.id)}"
          title="${p.id} · ${deskName(s.desk)} · ${timeText(s.start)}–${timeText(s.end)}">${p.id}</button>`).join("")}</div>`).join("");
      const ticks = Array.from({length:Math.floor(end/tick)+1}, (_,i) =>
        `<span style="left:${i*tick/end*100}%">${timeText(i*tick)}</span>`).join("");
      $("gantt").innerHTML = `<div class="gantt-grid"><div>${labels}</div><div class="gantt-body" style="--tick-step:${step}%">
        ${slots}<div id="playhead" class="playhead"></div><div class="gantt-axis">${ticks}</div></div></div>`;
    }
    function renderComparison() {
      const c = scenario();
      const desks = c.ddqnDecisions
        ? " · DDQN 分台 " + c.ddqnDecisions + " 次"
        : " · 单窗口，无分台选择";
      $("comparison-note").textContent = "当前回放 · " + methodName(state.method)
        + "　对照 · " + methodName(state.baseline) + desks;
      $("comparison").innerHTML = Object.entries(scenario().plans).map(([key,{metrics:m}]) =>
        `<tr class="${key === state.method ? "chosen " : ""}${key === state.baseline ? "baseline" : ""}">
          <td>${methodName(key)}</td><td class="num">${m.wait.toFixed(2)}</td><td class="num">${m.flow.toFixed(2)}</td>
          <td class="num">${m.p90.toFixed(2)}</td><td class="num">${m.ms.toFixed(2)}</td></tr>`).join("");
    }
    function renderMinute() {
      const live = plan().patients.map(patient => ({patient,current:patientState(patient,state.minute)}));
      const count = key => live.filter(({current}) => current.type === key).length;
      $("arrived").textContent = live.length-count("future");
      for (const key of ["waiting","busy","completed"]) $(key).textContent = count(key);
      $("clock").textContent = timeText(state.minute);
      $("seek").value = state.minute;
      $("seek").setAttribute("aria-valuetext", timeText(state.minute));
      $("playhead").style.left = Math.min(99.9, 100*state.minute/scenario().finish) + "%";
      const events = plan().events;
      $("previous").disabled = !events.some(e => e.t < state.minute);
      $("next").disabled = !events.some(e => e.t > state.minute);
      renderPatients(live);
      renderWindows(live);
      renderPatient(live);
      renderEvents();
    }
    function fillOptions(select, values, labelFor, active) {
      select.replaceChildren(...values.map(key => new Option(labelFor(key),key,false,key === active)));
    }
    function renderScenario() {
      const methods = Object.keys(scenario().plans);
      fillOptions($("method"), methods, methodName, state.method);
      fillOptions($("baseline"), methods, methodName, state.baseline);
      $("seek").max = scenario().finish;
      $("end-time").textContent = timeText(scenario().finish);
      $("case-note").textContent = state.scenario + " · 近满载 · 初诊和复诊各 1 个窗口";
      renderGantt();
      renderComparison();
      renderMinute();
    }
    function pause() {
      clearInterval(state.timer);
      state.timer = null;
      $("play").textContent = "播放";
    }
    function seek(minute) {
      state.minute = Math.max(0, Math.min(minute, scenario().finish));
      renderMinute();
    }
    function play() {
      if (state.minute >= scenario().finish) seek(0);
      $("play").textContent = "暂停";
      state.timer = setInterval(() => {
        seek(state.minute + 1);
        if (state.minute >= scenario().finish) pause();
      }, 220/Number($("speed").value));
    }
    function jumpEvent(direction) {
      pause();
      const times = plan().events.map(e => e.t);
      if (direction === -1) times.reverse();
      const minute = times.find(t => direction*(t-state.minute) > 0);
      if (minute !== undefined) seek(minute);
    }

    fillOptions($("case"), Object.keys(cases), key => key+"（"+key.split("×")[0]+" 人）", state.scenario);
    $("case").addEventListener("change", () => {
      pause();
      state.scenario = $("case").value;
      state.minute = 0;
      if (!plan().patients.some(p => p.id === state.patient)) state.patient = "P01";
      renderScenario();
    });
    $("window-count").addEventListener("change", () => {
      pause();
      state.windows = Number($("window-count").value);
      state.minute = Math.min(state.minute, scenario().finish);
      renderScenario();
    });
    $("method").addEventListener("change", () => {
      state.method = $("method").value;
      renderGantt();
      renderComparison();
      renderMinute();
    });
    $("baseline").addEventListener("change", () => {
      state.baseline = $("baseline").value;
      renderComparison();
      renderMinute();
    });
    $("play").addEventListener("click", () => state.timer === null ? play() : pause());
    $("previous").addEventListener("click", () => jumpEvent(-1));
    $("next").addEventListener("click", () => jumpEvent(1));
    $("reset").addEventListener("click", () => { pause(); seek(0); });
    $("speed").addEventListener("change", () => { if (state.timer !== null) { pause(); play(); } });
    $("seek").addEventListener("input", () => { pause(); seek(Number($("seek").value)); });
    $("filter-patient").addEventListener("change", renderEvents);
    $("app").addEventListener("click", event => {
      const btn = event.target.closest("button[data-patient],button[data-minute]");
      if (!btn) return;
      if (btn.dataset.patient) {
        state.patient = btn.dataset.patient;
        renderGantt();
        renderMinute();
        const selected = $("patient-list").querySelector("[aria-pressed=true]");
        if (selected) selected.focus({preventScroll:true});
      } else {
        pause();
        seek(Number(btn.dataset.minute));
      }
    });
    renderScenario();
    </script>
    </body>
    </html>

    """

    from html import escape
    from urllib.parse import quote

    payload = json.dumps(replay_data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # 禁止 JSON 文本中的尖括号结束脚本数据标签。
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    assert replay_template.count("__REPLAY_PAYLOAD__") == 1
    replay_page = replay_template.replace("__REPLAY_PAYLOAD__", payload)
    replay_html_path = RESULTS / "outpatient_replay.html"
    replay_html_path.write_text(replay_page, encoding="utf-8")
    print("互动回放页面已导出", replay_html_path)

    class OutpatientReplay:
        def _repr_html_(self):
            source_url = quote(replay_html_path.as_posix())
            return (
                f'<p><a href="{source_url}" target="_blank" rel="noopener">单独打开门诊调度回放台</a></p>'
                f'<iframe title="门诊调度动态回放台" sandbox="allow-scripts" '
                f'style="width:100%;height:920px;border:1px solid #dbe3dc;border-radius:12px" '
                f'srcdoc="{escape(replay_page, quote=True)}"></iframe>'
            )

        def __repr__(self):
            return f"门诊调度动态回放台 · {replay_html_path}"

    display(OutpatientReplay())

    return replay_html_path
