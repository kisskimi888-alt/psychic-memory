"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""

from scheduling import (
    Booking,
    Case,
    LOADS,
    Problem,
    Schedule,
    TIME_UNIT,
    Task,
    audit,
    genetic,
    math,
    measures,
    np,
    permutations,
    product,
    replace,
    rule,
    schedule_frame,
    synthetic,
)

from agents import (
    DDQN,
    FastDDQN,
    FastPPO,
    FastSchedule,
    PPO,
    fast_learned,
    learned,
)

from IPython.display import display
names = ("CT", "CBC", "ECG")
durations = ((2, 7, 2), (4, 4, 1), (7, 4, 6))
toy = Problem(Case((0, 0, 0), tuple(
    Task(j, name, (), ((name, durations[j][i]),))
    for j in range(3) for i, name in enumerate(names)
)))

def exact_table4(p):
    groups = [list(range(3*j, 3*j+3)) for j in range(3)]
    groups += [list(range(i, 9, 3)) for i in range(3)]
    best, best_order, tested, feasible = math.inf, None, 0, 0
    for orders in product(*(list(permutations(g)) for g in groups)):
        tested += 1
        succ, indeg = [[] for _ in range(9)], [0]*9
        for a, b, c in orders:
            succ[a].append(b)
            succ[b].append(c)
            indeg[b] += 1
            indeg[c] += 1
        queue = [i for i in range(9) if indeg[i] == 0]
        starts = [0]*9
        for i in queue:
            finish = starts[i] + p.minimum[i]
            for j in succ[i]:
                starts[j] = max(starts[j], finish)
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
        if len(queue) != 9:
            continue
        feasible += 1
        score = sum(max(starts[i] + p.minimum[i] for i in group) for group in groups[:3])
        if score < best:
            best, best_order = score, queue.copy()
    env = Schedule(p)
    for i in best_order:
        env.place(i, p.tasks[i].kind)
    return tested, feasible, best, env.rows

tested, feasible, optimum, exact = exact_table4(toy)
assert (tested, feasible, optimum) == (46656, 19164, 39)
assert measures(toy, exact)["total_flow"] == optimum
print(f"枚举 {tested:,} 种顺序，可行 {feasible:,} 种。最优总在院时长 {optimum} 患者·分钟。")
display(schedule_frame(toy, exact))

audit_case = Problem(Case((0, 0), (
    Task(0, "reg", (), (("R", 2),)), Task(0, "exam", (0,), (("X", 3),)),
    Task(1, "exam", (), (("X", 3),)),
)))
valid = [Booking(0, "R", 0, 2), Booking(1, "X", 2, 5), Booking(2, "X", 5, 8)]
faults = {
    "有效排程": (valid, set()),
    "同一个患者做两项 / 前置未完成": ([valid[0], Booking(1, "X", 1, 4), valid[2]],
                                     {"患者撞单", "前置未完成"}),
    "同一个窗口做两项": ([*valid[:2], Booking(2, "X", 3, 6)], {"窗口撞单"}),
    "窗口没资格": ([replace(valid[0], desk="Z"), *valid[1:]], {"窗口无资格", "时长错误"}),
    "漏排": (valid[:2], {"任务漏排"}),
}
for label, (rows, expected) in faults.items():
    got = audit(audit_case, rows)
    assert set(got) == expected, (label, got, expected)
    print(f"{label} → {', '.join(got) if got else '通过'}")

# 先锁定 [0, 10) 和 [30, 40)，再提交第三个项目，预期插入中间。
gap_case = Problem(Case((0, 30, 12), tuple(
    Task(i, "CT", (), (("CT/1", length),)) for i, length in enumerate((10, 10, 8))
)))
gap = Schedule(gap_case, graph=True)
for task in (0, 1, 2):
    gap.place(task, "CT/1")
assert gap.rows[-1] == Booking(2, "CT/1", 12, 20)
assert set(gap.edges) == {(0, 2), (2, 1)} and not audit(gap_case, gap.rows)
print("插空验算：第三个项目排在 12–20 分钟，窗口边从 0→1 更新为 0→2→1。")

for method in ("READY", "SPT", "MWKR", "MOPNR", "GA"):
    rows = genetic(toy, 100) if method == "GA" else rule(toy, method)
    result = measures(toy, rows)
    assert result["feasible"] and result["total_flow"] >= optimum
    print(f"{method:5s} → 总在院 {result['total_flow']} 患者·分钟")

paired_low = synthetic(15, 10, 1, 600, "low")
paired_peak = synthetic(15, 10, 3, 600, "peak")
assert all(a == replace(b, options=b.options[:1]) for a, b in zip(paired_low.tasks, paired_peak.tasks))
assert paired_low.case.arrival != paired_peak.case.arrival
print("病例生成检查通过：改变客流和窗口数，患者项目、前置关系和第一个窗口的时长一致。")

for i, desk in enumerate(("X","Y")):
    p = Problem(Case((0,0), (Task(0,"flex",(),(("X",5),("Y",5))),
                            Task(1,"fixed",(),((desk,5),)))))
    expected = np.zeros(2)
    expected[i] = 5
    assert np.allclose(Schedule(p,True).window_state(0)[0][:,2]*TIME_UNIT,expected)
p=Problem(Case((0,0,0),(Task(0,"previous",(),(("X",10),("Y",10))),
                       Task(1,"x",(),(("X",5),)), Task(2,"y",(),(("Y",5),)))))
states=[]
for desk in ("X","Y"):
    env=FastSchedule(p,True)
    env.place(0,desk)
    states.append(env.node_state("window_aware")[0])
assert not np.array_equal(*states)
for seed in range(12):
    p=synthetic(6,6,1+seed%3,29_200+seed,LOADS[seed%3])
    original_actor, actor=PPO(seed,"window_aware"),FastPPO(seed,"window_aware")
    original_desk, desk=DDQN(seed),FastDDQN(seed)
    slow=learned(p,original_actor,original_desk)
    quick=fast_learned(p,actor,desk)
    assert slow==quick and not audit(p,quick)
for n in (50,100):
    p=synthetic(n,20,2,29_220+n,"near_capacity")
    assert not FastSchedule(p,True).dense
    assert not audit(p,fast_learned(p,FastPPO(1,"window_aware"),FastDDQN(2)))
print("窗口占用反例、12 批逐单一致性、50/100 人大图可行性验收均通过。")