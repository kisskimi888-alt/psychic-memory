"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""


from dataclasses import dataclass, replace
from collections import defaultdict, deque
from bisect import bisect_right
from itertools import permutations, product
from pathlib import Path
import heapq
import json
import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TIME_UNIT = 60.0

@dataclass(frozen=True)
class Task:
    patient: int
    kind: str
    before: tuple
    options: tuple  # ((窗口名, 服务分钟), ...)

@dataclass(frozen=True)
class Case:
    arrival: tuple
    tasks: tuple

@dataclass(frozen=True)
class Booking:
    task: int
    desk: str
    start: int
    end: int

class Problem:
    """验证并整理输入。每个策略获得同一个 Problem，排程状态独立。"""
    def __init__(self, case):
        self.case = case
        self.tasks = case.tasks
        self.n = len(case.tasks)
        p = len(case.arrival)
        if not p or not self.n or any(not isinstance(a, int) or a < 0 for a in case.arrival):
            raise ValueError("到达时间须是非负的整数分钟，患者和任务不能为空")
        self.succ = [[] for _ in self.tasks]
        self.by_patient = [[] for _ in case.arrival]
        self.lookup = []
        self.edges = set()
        for i, t in enumerate(self.tasks):
            if not 0 <= t.patient < p:
                raise ValueError(f"任务 {i} 的患者编号无效")
            if not t.options or len(dict(t.options)) != len(t.options) or any(
                not isinstance(d, str) or not d or not isinstance(v, int) or v <= 0
                for d, v in t.options
            ):
                raise ValueError(f"任务 {i} 的窗口或时长无效")
            if len(set(t.before)) != len(t.before):
                raise ValueError(f"任务 {i} 的前置任务重复")
            for pred in t.before:
                if not 0 <= pred < self.n or self.tasks[pred].patient != t.patient or pred == i:
                    raise ValueError(f"任务 {i} 的前置任务无效")
                self.succ[pred].append(i)
                self.edges.add((pred, i))
            self.by_patient[t.patient].append(i)
            self.lookup.append(dict(t.options))
        if any(not tasks for tasks in self.by_patient):
            raise ValueError("每个患者至少要有一个项目")
        indegree = [len(t.before) for t in self.tasks]
        self.ready = [i for i, d in enumerate(indegree) if d == 0]
        queue = deque(self.ready)
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for nxt in self.succ[i]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != self.n:
            raise ValueError("前置关系有环")
        self.by_patient = [[] for _ in case.arrival]
        for i in order:
            self.by_patient[self.tasks[i].patient].append(i)
        self.minimum = [min(d.values()) for d in self.lookup]
        self.desks = sorted({desk for d in self.lookup for desk in d})
        self.tasks_by_desk = {desk: set() for desk in self.desks}
        for task, options in enumerate(self.lookup):
            for desk in options:
                self.tasks_by_desk[desk].add(task)
        self.width = max(map(len, self.lookup))
        self.work = [sum(self.minimum[i] for i in ids) for ids in self.by_patient]

class Schedule:
    def __init__(self, problem, graph=False):
        self.p = problem
        self.rows = []
        self.patient_end = list(problem.case.arrival)
        self.calendar = {desk: [] for desk in problem.desks}
        self.ends = [-1] * problem.n
        self.todo = [len(t.before) for t in problem.tasks]
        self.ready = set(problem.ready)
        self.work = problem.work.copy()
        self.left = list(map(len, problem.by_patient))
        self.graph = graph
        if graph:
            # 一条边可以同时来自固定约束、患者顺序和窗口顺序。
            self.edges = dict.fromkeys(sorted(problem.edges), 1)
            self.last_patient = {}
            self.features = np.zeros((problem.n, 2), np.float32)
            self.clb = [0] * problem.n
            for patient in range(len(self.patient_end)):
                self.update_nodes(patient)

    def update_nodes(self, patient):
        for i in self.p.by_patient[patient]:
            t = self.p.tasks[i]
            done = self.ends[i] >= 0
            self.clb[i] = self.ends[i] if done else max(
                self.patient_end[patient], max((self.clb[j] for j in t.before), default=0)
            ) + self.p.minimum[i]
            self.features[i] = (float(done), self.clb[i] / TIME_UNIT)

    def link(self, edge):
        self.edges[edge] = self.edges.get(edge, 0) + 1

    def unlink(self, edge):
        self.edges[edge] -= 1
        if self.edges[edge] == 0:
            del self.edges[edge]

    def node_state(self, observation="paper"):
        edges = np.asarray(tuple(self.edges), dtype=np.intp).reshape(-1, 2)
        legal = np.array(sorted(self.ready), dtype=np.intp)
        if observation == "paper":
            return self.features.copy(), edges, legal
        # 扩展对照：就绪时间 + 合格窗口造成的最短开工等待。
        extra = np.zeros_like(self.features)
        for i in legal:
            patient = self.p.tasks[i].patient
            ready = self.patient_end[patient]
            start = min(self.first_gap(desk, ready, duration)[1]
                        for desk, duration in self.p.tasks[i].options)
            extra[i] = ready / TIME_UNIT, (start-ready) / TIME_UNIT
        return np.concatenate((self.features, extra), axis=1), edges, legal

    def first_gap(self, desk, ready, duration):
        """返回第一个可行空档的插入位置和开工分钟，预订区间采用 [开始, 结束)。"""
        return self.gap_in(self.calendar[desk], ready, duration)

    @staticmethod
    def gap_in(booked, ready, duration):
        pos = bisect_right(booked, (ready, math.inf, math.inf))
        start = max(ready, booked[pos-1][1]) if pos else ready
        while pos < len(booked) and start + duration > booked[pos][0]:
            start = max(start, booked[pos][1])
            pos += 1
        return pos, start

    def window_state(self, task):
        patient = self.p.tasks[task].patient
        ready = self.patient_end[patient]
        options = self.p.tasks[task].options
        x = np.zeros((self.p.width, 3), np.float32)
        for idx, (desk, duration) in enumerate(options):
            pos, start = self.first_gap(desk, ready, duration)
            end = start + duration
            pressure = 0
            if len(options) > 1:
                calendar = self.calendar[desk]
                shadow = calendar[:pos] + [(start, end, task)] + calendar[pos:]
                for other in sorted(self.ready & self.p.tasks_by_desk[desk]):
                    other_patient = self.p.tasks[other].patient
                    if other_patient == patient:
                        continue
                    other_ready = self.patient_end[other_patient]
                    other_duration = self.p.lookup[other][desk]
                    _, old_start = self.first_gap(desk, other_ready, other_duration)
                    old_end = old_start + other_duration
                    if old_end <= start or old_start >= end:
                        continue
                    alternatives = [self.first_gap(d, other_ready, length)[1] + length
                                    for d, length in self.p.tasks[other].options if d != desk]
                    best_alt = min(alternatives, default=math.inf)
                    if best_alt <= old_end:
                        continue
                    new_end = self.gap_in(shadow, other_ready, other_duration)[1] + other_duration
                    pressure += min(best_alt, new_end) - old_end
            x[idx] = ((start-ready)/TIME_UNIT, (end-ready)/TIME_UNIT, pressure/TIME_UNIT)
        return x, len(options)

    def earliest(self, task):
        free = self.patient_end[self.p.tasks[task].patient]
        return min(self.p.tasks[task].options, key=lambda option:
                   (self.first_gap(option[0], free, option[1])[1] + option[1], option[0]))[0]

    def place(self, i, desk):
        if i not in self.ready or desk not in self.p.lookup[i]:
            raise ValueError("该项目暂不可排，或所选窗口无资格")
        patient = self.p.tasks[i].patient
        old_end = self.patient_end[patient]
        duration = self.p.lookup[i][desk]
        position, start = self.first_gap(desk, old_end, duration)
        end = start + duration
        bookings = self.calendar[desk]
        previous = bookings[position-1][2] if position else None
        following = bookings[position][2] if position < len(bookings) else None
        bookings.insert(position, (start, end, i))
        self.rows.append(Booking(i, desk, start, end))
        self.ends[i] = end
        self.patient_end[patient] = end
        self.ready.remove(i)
        self.work[patient] -= self.p.minimum[i]
        self.left[patient] -= 1
        for nxt in self.p.succ[i]:
            self.todo[nxt] -= 1
            if self.todo[nxt] == 0:
                self.ready.add(nxt)
        if self.graph:
            if patient in self.last_patient:
                self.link((self.last_patient[patient], i))
            if previous is not None and following is not None:
                self.unlink((previous, following))
            if previous is not None:
                self.link((previous, i))
            if following is not None:
                self.link((i, following))
            self.last_patient[patient] = i
            self.update_nodes(patient)
        # 变化只来自这个患者。潜势是已排到的末尾 + 剩余最短耗时 - 到达时间。
        return (old_end - end + self.p.minimum[i]) / TIME_UNIT

def audit(p, rows):
    errors, by_task = set(), {}
    desk_rows, patient_rows = defaultdict(list), defaultdict(list)
    for b in rows:
        if not 0 <= b.task < p.n:
            errors.add("未知任务")
            continue
        t = p.tasks[b.task]
        if b.task in by_task:
            errors.add("任务重复")
        by_task[b.task] = b
        if b.desk not in p.lookup[b.task]:
            errors.add("窗口无资格")
        if b.start >= b.end or b.end - b.start != p.lookup[b.task].get(b.desk):
            errors.add("时长错误")
        if b.start < p.case.arrival[t.patient]:
            errors.add("到达前开工")
        desk_rows[b.desk].append(b)
        patient_rows[t.patient].append(b)
    if len(by_task) != p.n:
        errors.add("任务漏排")
    for i, b in by_task.items():
        if any(k not in by_task or by_task[k].end > b.start for k in p.tasks[i].before):
            errors.add("前置未完成")
    for label, groups in (("窗口撞单", desk_rows), ("患者撞单", patient_rows)):
        for bookings in groups.values():
            end = -math.inf
            for b in sorted(bookings, key=lambda x: x.start):
                if b.start < end:
                    errors.add(label)
                end = max(end, b.end)
    return sorted(errors)

def measures(p, rows):
    errors = audit(p, rows)
    if errors:
        return dict(feasible=False, total_flow=np.nan, mean_flow=np.nan, p90_wait=np.nan)
    finish = [0] * len(p.case.arrival)
    service = finish.copy()
    for b in rows:
        j = p.tasks[b.task].patient
        finish[j] = max(finish[j], b.end)
        service[j] += b.end - b.start
    flow = np.subtract(finish, p.case.arrival)
    wait = flow - service
    return dict(feasible=True, total_flow=int(flow.sum()), mean_flow=float(flow.mean()),
                p90_wait=float(np.quantile(wait, 0.9)))

def schedule_frame(p, rows):
    return pd.DataFrame([
        (b.task, p.tasks[b.task].patient + 1, p.tasks[b.task].kind, b.desk, b.start, b.end)
        for b in sorted(rows, key=lambda row: (row.start, row.desk))
    ], columns=["Task", "Patient", "Activity", "Desk", "Start (min)", "End (min)"])

def queue_metrics(p, rows):
    """按患者拆等待；复诊窗口空着且有人已就绪时，累计窗口分钟。"""
    by_task = {b.task: b for b in rows}
    follow_events = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    waits = defaultdict(int)
    for patient, ids in enumerate(p.by_patient):
        ready = p.case.arrival[patient]
        for b in sorted((by_task[i] for i in ids), key=lambda b: b.start):
            kind = p.tasks[b.task].kind
            waits[kind] += b.start-ready
            if kind == "Follow" and len(p.tasks[b.task].options) == 1 and ready < b.start:
                follow_events[b.desk][ready][0] += 1
                follow_events[b.desk][b.start][0] -= 1
            ready = b.end
    for b in rows:
        if b.desk in follow_events:
            follow_events[b.desk][b.start][1] += 1
            follow_events[b.desk][b.end][1] -= 1
    idle = 0
    for timeline in follow_events.values():
        waiting, busy, last = 0, 0, 0
        for minute, (delta_wait, delta_busy) in sorted(timeline.items()):
            if waiting > 0 and busy == 0:
                idle += minute-last
            waiting += delta_wait
            busy += delta_busy
            last = minute
    n = len(p.case.arrival)
    return dict(consult_wait=waits["Consult"]/n, follow_wait=waits["Follow"]/n,
                follow_idle_with_waiting=idle)

LOADS = ("low", "near_capacity", "peak")

def arrival_times(n, mean_consult, rng, load):
    nominal = {"low": .65, "near_capacity": .95, "peak": .80}[load]
    gaps = rng.gamma(4, .25, n-1)
    targets = np.full(n-1, mean_consult / nominal)
    if load == "peak":
        targets[:max(1, (n-1)//3)] = mean_consult / 2.0
    return tuple(map(int, np.rint(np.r_[0, (gaps*targets).cumsum()])))

def synthetic(n, m, k, seed, load="near_capacity"):
    if n < 1 or m < 5 or not 1 <= k <= 3:
        raise ValueError("需要至少一位患者、五类项目、1–3 个可变窗口")
    parts = np.random.SeedSequence(seed).spawn(3)
    rng, service_rng, arrival_rng = map(np.random.default_rng, parts)
    exams = [f"Exam-{i+1:02}" for i in range(m-4)]
    base = {"Reg": 4, "Consult": 10, "Follow": 7, "Rx": 4}
    base.update(zip(exams, map(int, rng.integers(7, 22, len(exams)))))
    pace = {a: service_rng.uniform(.82, 1.18, 1 if a in ("Consult", "Follow") else 3)
            for a in base}
    tasks = []
    def add(patient, kind, predecessors):
        idx = len(tasks)
        length = base[kind] * rng.uniform(.70, 1.32)
        options = tuple((f"{kind}/{d+1}", max(1, round(length*f + service_rng.uniform(-2, 2))))
                        for d, f in enumerate(pace[kind]))
        tasks.append(Task(patient, kind, predecessors, options[:k]))
        return idx
    for j in range(n):
        reg = add(j, "Reg", ())
        consult = add(j, "Consult", (reg,))
        count = rng.integers(min(2, len(exams)), min(5, len(exams)) + 1)
        selected = sorted(rng.choice(exams, count, replace=False))
        exams_done = tuple(add(j, e, (consult,)) for e in selected)
        follow = add(j, "Follow", exams_done)
        add(j, "Rx", (follow,))
    mean_consult = np.mean([t.options[0][1] for t in tasks if t.kind == "Consult"])
    return Problem(Case(arrival_times(n, mean_consult, arrival_rng, load), tuple(tasks)))

def rule(p, mode):
    env = Schedule(p)
    while env.ready:
        def rank(i):
            j = p.tasks[i].patient
            if mode == "READY":
                return (env.patient_end[j], p.case.arrival[j], i)
            if mode == "SPT":
                return (p.minimum[i], env.patient_end[j], i)
            if mode == "MWKR":
                return (-env.work[j], env.patient_end[j], i)
            if mode == "MOPNR":
                return (-env.left[j], env.patient_end[j], i)
            raise ValueError(mode)
        i = min(env.ready, key=rank)
        env.place(i, env.earliest(i))
    return env.rows

def genetic(p, seed, population=12, generations=10):
    if population < 6:
        raise ValueError("种群至少要容纳四个规则种子和随机个体")
    rng = np.random.default_rng(seed)
    option_counts = np.asarray([len(t.options) for t in p.tasks])
    def decode(gene):
        env = Schedule(p)
        # 固定基因优先级可用堆保存，只把已经就绪的项目推入堆。
        priority = gene[0].tolist()
        slots = np.floor(gene[1] * option_counts).astype(int)
        queue = [(priority[i], i) for i in p.ready]
        heapq.heapify(queue)
        while queue:
            _, i = heapq.heappop(queue)
            env.place(i, p.tasks[i].options[slots[i]][0])
            for j in p.succ[i]:
                if env.todo[j] == 0:
                    heapq.heappush(queue, (priority[j], j))
        return sum(e-a for e, a in zip(env.patient_end, p.case.arrival)), env.rows
    genes = rng.random((population, 2, p.n))
    for g, method in enumerate(("READY", "SPT", "MWKR", "MOPNR")):
        for rank, booking in enumerate(rule(p, method)):
            opts = list(p.lookup[booking.task])
            genes[g, :, booking.task] = (rank/p.n, (opts.index(booking.desk)+.5)/len(opts))
    scores = [decode(g) for g in genes]
    for _ in range(generations):
        ranked = sorted(range(population), key=lambda i: scores[i][0])
        next_genes = [genes[i] for i in ranked[:2]]
        next_scores = [scores[i] for i in ranked[:2]]
        while len(next_genes) < population:
            a, b = rng.choice(ranked[:population//2], 2, replace=False)
            child = np.where(rng.random((2, p.n)) < .5, genes[a], genes[b])
            change = rng.random((2, p.n)) < .07
            child[change] = rng.random(change.sum())
            next_genes.append(child)
            next_scores.append(decode(child))
        genes, scores = np.asarray(next_genes), next_scores
    return min(scores, key=lambda result: result[0])[1]

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