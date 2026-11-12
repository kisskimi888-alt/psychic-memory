"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""

from scheduling import (
    LOADS,
    Path,
    Schedule,
    TIME_UNIT,
    json,
    math,
    measures,
    np,
    os,
    pd,
    rule,
    synthetic,
    time,
)

from agents import (
    Adam,
    FastDDQN,
    FastPPO,
    FastSchedule,
    Replay,
    batch_ppo_update,
    fast_learned,
    load_weights,
    mlp_back,
    save_weights,
)

RUN_SCOPE = "small"               # "small"（6×6 和 10×6）、"6x6"、"10x6" 或 "all"
RUN_PROFILE = "pilot"             # "pilot" 或 "paper_budget"
RUN_DDQN_DIAGNOSTIC = True         # PPO 未过门槛时，是否仍训练 DDQN 作诊断
PPO_ROUNDS = 180 if RUN_PROFILE == "pilot" else 10_000
DDQN_EPISODES = 300 if RUN_PROFILE == "pilot" else 3_000
VALIDATION_CASES = 45 if RUN_PROFILE == "pilot" else 100
VALIDATE_EVERY = 30 if RUN_PROFILE == "pilot" else 250
SAVE_EVERY = 30 if RUN_PROFILE == "pilot" else 100
POLICY_PASSES = 10
POLICY_CASES_PER_PASS = 36
PPO_GATE_MARGIN = 0.05           # 至少快 0.05 分钟／人，才正式进入 DDQN 阶段
GA_POPULATION, GA_GENERATIONS = 20, 15
TEST_REPEATS = 3 if RUN_SCOPE == "all" else 12
assert RUN_SCOPE in ("small", "6x6", "10x6", "all") and RUN_PROFILE in ("pilot", "paper_budget")
OUT = Path("hospital_latest_results")
def ready_priority(env):
    return min(env.ready,key=lambda i:(env.patient_end[env.p.tasks[i].patient],
                                       env.p.case.arrival[env.p.tasks[i].patient],i))


def fork_without_graph(env):
    clone=Schedule(env.p)
    for key in ('rows','patient_end','ends','todo','work','left'):
        setattr(clone,key,getattr(env,key).copy())
    clone.ready=env.ready.copy()
    clone.calendar={k:v.copy() for k,v in env.calendar.items()}
    return clone


def complete_after(env,task):
    clone=fork_without_graph(env)
    clone.place(task,clone.earliest(task))
    while clone.ready:
        nxt=ready_priority(clone)
        clone.place(nxt,clone.earliest(nxt))
    return sum(end-arr for end,arr in zip(clone.patient_end,clone.p.case.arrival))


def on_policy_examples(model, n, m, rng, cases):
    examples=[]
    for i in range(cases):
        case=synthetic(n,m,1+(i//3)%3,int(rng.integers(1_000_000_000,1_700_000_000)),LOADS[i%3])
        env=FastSchedule(case,True)
        for step in range(case.n):
            state=env.node_state(model.observation)
            legal=state[2]
            if step % 4 == i % 4 and len(legal)>1:
                scores=np.array([complete_after(env,int(a)) for a in legal],float)
                if scores.max()>scores.min():
                    examples.append((state,scores-scores.min()))
            task=int(rng.choice(legal)) if len(legal)>1 and rng.random()<.12 else model.choose(state)
            env.place(task,env.earliest(task))
    return examples

def train_ranking_round(model, examples, optimizer, rng):
    if not examples:
        return 0.0
    losses=[]
    for batch in np.array_split(rng.permutation(len(examples)), max(1,len(examples)//32)):
        gradient=[np.zeros_like(w) for w in model.parameters]
        for i in batch:
            state,gap=examples[i]
            prob,_,cache=model.forward(state)
            expected=np.exp(-gap/3)
            expected/=expected.sum()
            losses.append(-float(expected@np.log(prob+1e-12)))
            for dst,src in zip(gradient,model.backward(prob-expected,0,cache)):
                dst+=src
        optimizer.step([g/len(batch) for g in gradient])
    return float(np.mean(losses))

def ppo_trajectory(p, agent, rng):
    env = FastSchedule(p, graph=True)
    rollout = []
    while env.ready:
        state = env.node_state(agent.observation)
        prob, value, _ = agent.forward(state)
        action = int(rng.choice(len(prob), p=prob.astype(float)/prob.sum(dtype=float)))
        task = int(state[2][action])
        reward = env.place(task, env.earliest(task))
        rollout.append((state, action, float(np.log(prob[action]+1e-12)), value, reward))
    assert np.isclose(sum(row[-1] for row in rollout),
                      (sum(p.work) - sum(e-a for e, a in zip(env.patient_end, p.case.arrival))) / TIME_UNIT)
    records = []
    advantage = next_value = 0.0
    for state, action, old_log, value, reward in reversed(rollout):
        advantage = reward + next_value - value + .95*advantage
        records.append((state, action, old_log, advantage, value + advantage))
        next_value = value
    return records, sum(e-a for e, a in zip(env.patient_end, p.case.arrival)) / len(p.case.arrival)



def ddqn_update(replay, model, target, optimizer, rng):
    xs, counts, action, reward, next_x, next_count, done = replay.sample(rng)
    mask = np.arange(xs.shape[1]) < counts[:, None]
    next_mask = np.arange(xs.shape[1]) < next_count[:, None]
    online, _ = model.forward(next_x, next_mask)
    next_action = np.argmax(np.where(next_mask, online, -np.inf), axis=1)
    target_q, _ = target.forward(next_x, next_mask)
    i = np.arange(len(xs))
    y = reward + .95*target_q[i, next_action]*(~done)
    q, cache = model.forward(xs, mask)
    error = q[i, action] - y
    g = np.zeros_like(q)
    g[i, action] = np.clip(error, -1, 1)/len(xs)
    _, gradients = mlp_back(g.reshape(-1, 1), cache, model.parameters)
    optimizer.step(gradients)
    return float(np.mean(np.where(np.abs(error) <= 1, .5*error**2, np.abs(error)-.5)))

def validation(problems, ppo, ddqn=None):
    return float(np.mean([measures(p, fast_learned(p, ppo, ddqn))["mean_flow"] for p in problems]))

def copy_parameters(dst, src):
    for a, b in zip(dst, src, strict=True):
        a[...] = b

class Stage:
    """维护一阶段的最新权重、验证最优权重和可续跑的完整状态。"""
    def __init__(self, path, identity, model, optimizer, rng, baseline, target=None, replay=None):
        self.path, self.identity = path, identity
        self.model, self.optimizer, self.rng = model, optimizer, rng
        self.target, self.replay = target, replay
        self.step, self.best_step, self.log = 0, 0, []
        self.best = [w.copy() for w in model.parameters]
        if path.exists():
            self.restore()
        else:
            self.record(0, baseline())
            self.save(0)

    def record(self, step, value, flow=None, loss=None, epsilon=None):
        assert np.isfinite(value)
        old_best = min((row["validation_flow"] for row in self.log), default=math.inf)
        self.log.append(dict(step=step, validation_flow=float(value), updates=self.optimizer.t,
                             training_flow=None if flow is None else float(flow),
                             loss=None if loss is None else float(loss), epsilon=epsilon))
        if value < old_best - 1e-9:
            self.best_step = step
            copy_parameters(self.best, self.model.parameters)

    def arrays(self):
        groups = {"model": self.model.parameters, "best": self.best,
                  "m": self.optimizer.m, "v": self.optimizer.v}
        if self.target is not None:
            groups["target"] = self.target.parameters
        return {f"{name}_{i}": a for name, values in groups.items() for i, a in enumerate(values)}

    def save(self, step):
        meta = dict(identity=self.identity, step=step, best_step=self.best_step, log=self.log,
                    updates=self.optimizer.t, rng=self.rng.bit_generator.state)
        arrays = self.arrays()
        if self.replay:
            arrays.update({f"replay_{i}": np.asarray(values)
                           for i, values in enumerate(zip(*self.replay))})
        partial = self.path.with_suffix(".tmp")
        with partial.open("wb") as handle:
            np.savez(handle, metadata=json.dumps(meta, allow_nan=False), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(self.path)
        self.step = step

    def restore(self):
        with np.load(self.path, allow_pickle=False) as saved:
            meta = json.loads(saved["metadata"].item())
            if meta["identity"] != self.identity:
                raise ValueError(f"检查点配置不一致：{self.path}")
            for key, dest in self.arrays().items():
                assert saved[key].shape == dest.shape and np.isfinite(saved[key]).all()
                dest[...] = saved[key]
            self.step, self.best_step, self.log = meta["step"], meta["best_step"], meta["log"]
            self.optimizer.t = meta["updates"]
            self.rng.bit_generator.state = meta["rng"]
            if self.replay is not None and "replay_0" in saved:
                self.replay.extend(zip(*(saved[f"replay_{i}"] for i in range(7))))
        print(f"恢复 {self.path.name}，从第 {self.step + 1} 轮继续。", flush=True)

def fit(label, n, m, seed, ppo_rounds, ddqn_episodes, val_count, validate_every,
        save_every, observation):
    started = time.perf_counter()
    directory = OUT / label
    directory.mkdir(exist_ok=True)
    identity = dict(version="gap-v5-onpolicy", n=n, m=m, seed=seed, ppo_rounds=ppo_rounds,
                    ddqn_episodes=ddqn_episodes, val_count=val_count,
                    validate_every=validate_every, observation=observation,
                    rank_passes=POLICY_PASSES, rank_cases=POLICY_CASES_PER_PASS,
                    gate_margin=PPO_GATE_MARGIN, diagnostic=RUN_DDQN_DIAGNOSTIC)
    val = [synthetic(n, m, 1+(i//3) % 3, 2_000_000_000 + seed*1000 + i, LOADS[i % 3])
           for i in range(val_count)]
    ready = float(np.mean([measures(p, rule(p, "READY"))["mean_flow"] for p in val]))

    def training_case(rng, k, load):
        return synthetic(n, m, k, int(rng.integers(0, 1_000_000_000)), load)

    def persist(stage, limit, name, step, check):
        if step % save_every == 0 or check:
            stage.save(step)
            if check or step % (save_every*4) == 0:
                last = stage.log[-1]["validation_flow"]
                print(f"{label} {name} {step:,}/{limit:,}｜最近验证 {last:.2f} 分钟/人"
                      f"｜已保存，累计 {time.perf_counter()-started:.0f} 秒", flush=True)

    ppo = FastPPO(seed, observation)
    rng = np.random.default_rng(seed)
    opt = Adam(ppo.parameters, 2e-5)
    patient_stage = Stage(directory / "ppo_state.npz", identity, ppo, opt, rng,
                          lambda: validation(val, ppo))
    for step in range(patient_stage.step+1, ppo_rounds+1):
        buffer, flow = [], []
        for j in range(4):
            counter = 4*(step-1) + j
            p = training_case(rng, 1+(counter//3) % 3, LOADS[counter % 3])
            records, average = ppo_trajectory(p, ppo, rng)
            buffer.extend(records)
            flow.append(average)
        loss = batch_ppo_update(buffer, ppo, opt, rng)
        check = step % validate_every == 0 or step == ppo_rounds
        if check:
            patient_stage.record(step, validation(val, ppo), np.mean(flow), loss)
        persist(patient_stage, ppo_rounds, "PPO", step, check)
    save_weights(directory / "ppo_pre_policy.npz", patient_stage.best)
    ppo = FastPPO(seed+800, observation)
    load_weights(directory / "ppo_pre_policy.npz", ppo.parameters)
    ppo_validation = validation(val,ppo)
    rank_stage=None
    rank_examples=[]
    if (n, m) in ((6, 6), (10, 6)):
        rank_rng=np.random.default_rng(seed+9851)
        rank_opt=Adam(ppo.parameters,.0003)
        rank_stage=Stage(directory/"ranking_state.npz",dict(identity,ppo_step=patient_stage.best_step),
                         ppo,rank_opt,rank_rng,lambda: validation(val,ppo))
        for iteration in range(rank_stage.step+1,POLICY_PASSES+1):
            examples=on_policy_examples(ppo,n,m,rank_rng,POLICY_CASES_PER_PASS)
            loss=train_ranking_round(ppo,examples,rank_opt,rank_rng)
            score=validation(val,ppo)
            rank_stage.record(iteration,score,loss=loss)
            rank_stage.log[-1]["informative_states"]=len(examples)
            rank_stage.save(iteration)
            print(f"{label} 自身排程排序 {iteration}/{POLICY_PASSES}｜有效状态 {len(examples)}"
                  f"｜验证 {score:.3f} 分钟/人",flush=True)
        copy_parameters(ppo.parameters,rank_stage.best)
        rank_examples=rank_stage.log
    save_weights(directory / "ppo.npz",ppo.parameters)
    ect=validation(val,ppo)
    ranking_step=rank_stage.best_step if rank_stage else 0
    has_trained_actor=bool(patient_stage.best_step or ranking_step)
    passed=has_trained_actor and ect<=ready-PPO_GATE_MARGIN
    ddqn_mode="正式" if passed else ("诊断" if RUN_DDQN_DIAGNOSTIC else "未启训")
    selection=dict(model=label,ready=ready,ppo_before_ranking=ppo_validation,
                   ppo_ect=ect,ppo_ddqn=None,ppo_step=patient_stage.best_step,
                   ranking_step=ranking_step,ddqn_step=None,ddqn_mode=ddqn_mode)
    history=[dict(row,model=label,stage=stage) for stage,rows in
             (("PPO",patient_stage.log),("自身排程排序",rank_examples)) for row in rows]
    print(f"{label} PPO+ECT {ect:.3f}｜READY {ready:.3f}｜DDQN {ddqn_mode}",flush=True)
    if ddqn_mode=="未启训":
        logs=pd.DataFrame(history)
        logs.to_csv(directory/"training_log.csv",index=False)
        return ppo,None,logs,selection

    ddqn, target = FastDDQN(seed+1), FastDDQN(seed+2)
    copy_parameters(target.parameters, ddqn.parameters)
    rng = np.random.default_rng(seed+41)
    opt = Adam(ddqn.parameters, .00025)
    replay = Replay(2000)
    window_stage = Stage(directory / "ddqn_state.npz", dict(identity, ppo_step=patient_stage.best_step, ranking_step=ranking_step),
                         ddqn, opt, rng, lambda: validation(val, ppo, ddqn), target, replay)

    def next_choice(env):
        task = next(iter(env.ready)) if len(env.ready) == 1 else ppo.choose(env.node_state(ppo.observation))
        return task, env.window_state(task)

    for episode in range(window_stage.step+1, ddqn_episodes+1):
        # 窗口轴固定补齐到 3，单窗口项目仍严格屏蔽不合法的动作。
        p = training_case(rng, 2+((episode-1)//3)%2, LOADS[(episode-1) % 3])
        env = FastSchedule(p, graph=True)
        epsilon = 1 - .9*(episode-1)/max(1, ddqn_episodes-1)
        losses = []
        task, state = next_choice(env)
        while True:
            x, count = state
            action = int(rng.integers(count)) if rng.random() < epsilon else ddqn.choose(state)
            reward = env.place(task, p.tasks[task].options[action][0])
            done = not env.ready
            next_task, nxt = (None, (np.zeros_like(x), 1)) if done else next_choice(env)
            replay.append((x, count, action, reward, *nxt, done))
            if len(replay) >= 100:
                losses.append(ddqn_update(replay, ddqn, target, opt, rng))
                if opt.t % 100 == 0:
                    copy_parameters(target.parameters, ddqn.parameters)
            if done:
                break
            task, state = next_task, nxt
        check = episode % validate_every == 0 or episode == ddqn_episodes
        if check:
            train = sum(e-a for e, a in zip(env.patient_end, p.case.arrival)) / n
            window_stage.record(episode, validation(val, ppo, ddqn), train,
                                np.mean(losses) if losses else None, epsilon)
        persist(window_stage, ddqn_episodes, "DDQN", episode, check)
    save_weights(directory / "ddqn.npz", window_stage.best)
    ddqn = FastDDQN(seed+801)
    load_weights(directory / "ddqn.npz", ddqn.parameters)
    dual = min(row["validation_flow"] for row in window_stage.log)
    history.extend(dict(row,model=label,stage="DDQN") for row in window_stage.log)
    logs=pd.DataFrame(history)
    logs.to_csv(directory/"training_log.csv",index=False)
    selection.update(ppo_ddqn=dual,ddqn_step=window_stage.best_step)
    print(f"{label} DDQN 选中轮次 {window_stage.best_step}｜验证 {dual:.3f}",flush=True)
    return ppo,ddqn,logs,selection