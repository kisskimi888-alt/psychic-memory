"""门诊双智能体调度实验。基于随附 Notebook 的可复现实现。"""

from scheduling import (
    Schedule,
    TIME_UNIT,
    math,
    np,
)

def linear(rng, n_in, n_out, scale=1.0):
    return [rng.normal(0, scale/np.sqrt(n_in), (n_in, n_out)).astype(np.float32),
            np.zeros(n_out, np.float32)]

def affine(x, parameters):
    w, b = parameters
    return x @ w + b

def mlp(x, parameters, activation=True):
    hidden = np.tanh(affine(x, parameters[:2]))
    raw = affine(hidden, parameters[2:])
    y = np.tanh(raw) if activation else raw
    return y, (x, hidden, y, activation)

def mlp_back(dy, cache, parameters):
    x, hidden, y, activated = cache
    dz = dy * (1-y*y) if activated else dy
    w1, _, w2, _ = parameters
    dh = (dz @ w2.T) * (1-hidden*hidden)
    return dh @ w1.T, [x.T @ dh, dh.sum(0), hidden.T @ dz, dz.sum(0)]

def save_weights(path, parameters):
    np.savez_compressed(path, *parameters)

def load_weights(path, parameters):
    with np.load(path) as file:
        assert len(file.files) == len(parameters)
        for i, value in enumerate(parameters):
            array = file[f"arr_{i}"]
            assert value.shape == array.shape and np.isfinite(array).all()
            value[...] = array

class Adam:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(w) for w in parameters]
        self.v = [np.zeros_like(w) for w in parameters]

    def step(self, gradients):
        norm = math.sqrt(sum(float(np.square(g).sum()) for g in gradients))
        scale = min(1.0, 0.5/(norm + 1e-9))
        self.t += 1
        for w, m, v, g in zip(self.parameters, self.m, self.v, gradients):
            g = g * scale
            m *= .9
            m += .1*g
            v *= .999
            v += .001*g*g
            w -= self.lr * (m/(1-.9**self.t)) / (np.sqrt(v/(1-.999**self.t)) + 1e-8)

def aggregate(x, edges, reverse=False):
    src, dst = edges.T
    if reverse:
        src, dst = dst, src
    out = x.copy()
    np.add.at(out, dst, x[src])
    return out

class PPO:
    def __init__(self, seed, observation="paper"):
        if observation not in ("paper", "window_aware"):
            raise ValueError(observation)
        self.observation = observation
        rng = np.random.default_rng(seed)
        self.g1 = linear(rng, 2 if observation == "paper" else 4, 64) + linear(rng, 64, 64)
        self.g2 = linear(rng, 64, 64) + linear(rng, 64, 64)
        self.actor = linear(rng, 128, 32) + linear(rng, 32, 1, .05)
        self.critic = linear(rng, 64, 32) + linear(rng, 32, 1, .1)
        self.parameters = self.g1 + self.g2 + self.actor + self.critic

    def forward(self, state):
        x, edges, legal = state
        h1, c1 = mlp(aggregate(x, edges), self.g1)
        h2, c2 = mlp(aggregate(h1, edges), self.g2)
        pool = h2.mean(axis=0, keepdims=True)
        local = np.concatenate((h2[legal], np.broadcast_to(pool, (len(legal), 64))), axis=1)
        logits, ca = mlp(local, self.actor, activation=False)
        value, cv = mlp(pool, self.critic, activation=False)
        z = logits[:, 0] - logits.max()
        prob = np.exp(z)
        prob /= prob.sum()
        return prob, float(value.item()), (edges, legal, c1, c2, ca, cv, len(x))

    def choose(self, state):
        x, edges, legal = state
        if len(legal) == 1:
            return int(legal[0])
        h = np.tanh(affine(np.tanh(affine(aggregate(x, edges), self.g1[:2])), self.g1[2:]))
        h = np.tanh(affine(np.tanh(affine(aggregate(h, edges), self.g2[:2])), self.g2[2:]))
        pooled = h.mean(axis=0)
        # Same affine map, split at the local/global concatenation.
        w, b, v, c = self.actor
        hidden = np.tanh(h[legal] @ w[:64] + pooled @ w[64:] + b)
        return int(legal[np.argmax(hidden @ v + c)])

    def backward(self, logits_grad, value_grad, cache):
        edges, legal, c1, c2, ca, cv, n = cache
        da, ga = mlp_back(logits_grad[:, None], ca, self.actor)
        dv, gv = mlp_back(np.asarray([[value_grad]], np.float32), cv, self.critic)
        dh = np.broadcast_to((da[:, 64:].sum(0) + dv[0]) / n, (n, 64)).copy()
        dh[legal] += da[:, :64]
        dh1, g2 = mlp_back(dh, c2, self.g2)
        _, g1 = mlp_back(aggregate(dh1, edges, reverse=True), c1, self.g1)
        return g1 + g2 + ga + gv

class DDQN:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.parameters = linear(rng, 6, 32) + linear(rng, 32, 1, .1)

    def forward(self, x, mask):
        # 共享打分器使 1、2、3 个候选窗口可用同一套权重。
        mean = (x * mask[..., None]).sum(-2, keepdims=True) / mask.sum(-1)[..., None, None]
        features = np.concatenate([x, np.broadcast_to(mean, x.shape)], axis=-1)
        values, cache = mlp(features.reshape(-1, 6).astype(np.float32),
                            self.parameters, activation=False)
        return values.reshape(mask.shape), cache

    def choose(self, state):
        x, count = state
        mask = np.arange(len(x)) < count
        q, _ = self.forward(x, mask)
        return int(np.argmax(q[:count]))

def learned(p, agent, window=None):
    env = Schedule(p, graph=True)
    while env.ready:
        task = next(iter(env.ready)) if len(env.ready) == 1 else agent.choose(env.node_state(agent.observation))
        options = p.tasks[task].options
        if len(options) == 1:
            desk = options[0][0]
        elif window is None:
            desk = env.earliest(task)
        elif window == "pressure_tie":
            features, count = env.window_state(task)
            best = min(range(count), key=lambda i: (features[i,1], features[i,2], options[i][0]))
            desk = options[best][0]
        else:
            desk = options[window.choose(env.window_state(task))][0]
        env.place(task, desk)
    return env.rows

class FastSchedule(Schedule):
    """Dense adjacency for a small graph, maintained as its edges change."""
    def __init__(self,p,graph=False):
        super().__init__(p,graph)
        self.dense = graph and p.n <= 64
        if self.dense:
            self.adj=np.eye(p.n,dtype=np.float32)
            self.wait_cache={}
            for src,dst in self.edges:
                self.adj[dst,src]=1

    def link(self,edge):
        super().link(edge)
        src,dst=edge
        if self.dense:
            self.adj[dst,src]=1

    def unlink(self,edge):
        super().unlink(edge)
        if self.dense and edge not in self.edges:
            src,dst=edge
            self.adj[dst,src]=0

    def node_state(self,observation="paper",snapshot=True):
        if not self.dense:
            return super().node_state(observation)
        legal=np.asarray(sorted(self.ready),dtype=np.intp)
        if observation=="paper":
            features=self.features.copy() if snapshot else self.features
        else:
            features=np.zeros((self.p.n,4),np.float32)
            features[:,:2]=self.features
            for i in legal:
                t=self.p.tasks[i]
                ready=self.patient_end[t.patient]
                if i not in self.wait_cache:
                    start=min(self.first_gap(d,ready,length)[1] for d,length in t.options)
                    self.wait_cache[i]=start-ready
                features[i,2:]=(ready/TIME_UNIT,self.wait_cache[i]/TIME_UNIT)
        return features,self.adj.copy() if snapshot else self.adj,legal

    def place(self,task,desk):
        reward=super().place(task,desk)
        if self.dense:
            self.wait_cache.pop(task,None)
            patient=self.p.tasks[task].patient
            changed=self.p.tasks_by_desk[desk].intersection(self.ready)
            changed.update(self.p.by_patient[patient])
            for i in changed:
                self.wait_cache.pop(i,None)
        return reward

    def window_state(self,task):
        x,count=super().window_state(task)
        if len(x)<3:
            fixed=np.zeros((3,3),dtype=x.dtype)
            fixed[:len(x)]=x
            x=fixed
        return x,count


class FastPPO(PPO):
    def forward(self,state):
        x,adj,legal=state
        if adj.dtype.kind in "iu":
            prob,value,cache=super().forward(state)
            return prob,value,("sparse",cache)
        h1,c1=mlp(adj@x,self.g1)
        h2,c2=mlp(adj@h1,self.g2)
        pool=h2.mean(axis=0,keepdims=True)
        local=np.concatenate((h2[legal],np.broadcast_to(pool,(len(legal),64))),axis=1)
        logits,ca=mlp(local,self.actor,activation=False)
        value,cv=mlp(pool,self.critic,activation=False)
        prob=np.exp(logits[:,0]-logits.max())
        prob/=prob.sum()
        return prob,float(value.item()),(adj,legal,c1,c2,ca,cv,len(x))

    def choose(self,state,original_edges=None):
        x,adj,legal=state
        if adj.dtype.kind in "iu":
            return super().choose(state)
        if len(legal)==1:
            return int(legal[0])
        h1,_=mlp(adj@x,self.g1)
        h2,_=mlp(adj@h1,self.g2)
        w,b,v,c=self.actor
        hidden=np.tanh(h2[legal]@w[:64]+h2.mean(axis=0)@w[64:]+b)
        score=(hidden@v+c)[:,0]
        best=int(np.argmax(score))
        if original_edges is not None and np.count_nonzero(score[best]-score<=1e-6)>1:
            return super().choose((x,original_edges(),legal))
        return int(legal[best])

    def backward(self,logits_grad,value_grad,cache):
        if len(cache)==2:
            return super().backward(logits_grad,value_grad,cache[1])
        adj,legal,c1,c2,ca,cv,n=cache
        da,ga=mlp_back(logits_grad[:,None],ca,self.actor)
        dv,gv=mlp_back(np.asarray([[value_grad]],np.float32),cv,self.critic)
        dh=np.broadcast_to((da[:,64:].sum(0)+dv[0])/n,(n,64)).copy()
        dh[legal]+=da[:,:64]
        dh1,g2=mlp_back(dh,c2,self.g2)
        _,g1=mlp_back(adj.T@dh1,c1,self.g1)
        return g1+g2+ga+gv


class FastDDQN(DDQN):
    def choose(self,state):
        x,count=state
        if count==1:
            return 0
        active=x[:count]
        mean=(active.sum(axis=0)/float(count)).astype(np.float32)
        w,b,v,c=self.parameters
        h=np.tanh(active@w[:3]+mean@w[3:]+b)
        score=(h@v+c)[:,0]
        best=int(np.argmax(score))
        # Preserve the old decision for ties and floating-point near ties.
        if np.count_nonzero(score[best]-score<=1e-6)>1:
            return super().choose(state)
        return best


class Replay:
    def __init__(self,capacity):
        self.capacity=capacity
        self.size=0
        self.head=0
        self.arrays=(np.empty((capacity,3,3),np.float32),np.empty(capacity,np.intp),
                     np.empty(capacity,np.intp),np.empty(capacity,float),
                     np.empty((capacity,3,3),np.float32),np.empty(capacity,np.intp),
                     np.empty(capacity,bool))

    def __len__(self):
        return self.size

    def append(self,row):
        for a,v in zip(self.arrays,row):
            a[self.head]=v
        self.head=(self.head+1)%self.capacity
        self.size=min(self.size+1,self.capacity)

    def extend(self,rows):
        for row in rows:
            self.append(row)

    def __iter__(self):
        indices=(np.arange(self.size)+self.head-self.size)%self.capacity
        return iter(zip(*(a[indices] for a in self.arrays)))

    def sample(self,rng):
        order=rng.choice(self.size,32,replace=False)
        index=(order+self.head-self.size)%self.capacity
        return tuple(a[index] for a in self.arrays)


def batch_ppo_update(records,model,optimizer,rng):
    first_shape=records[0][0][0].shape
    if any(state[0].shape!=first_shape or state[1].dtype.kind in "iu"
           for state,*_ in records):
        return sequential_ppo_update(records,model,optimizer,rng)
    adv=np.asarray([r[3] for r in records])
    norm=(adv-adv.mean())/(adv.std()+1e-8)
    losses=[]
    for indices in np.array_split(rng.permutation(len(records)),4):
        group=[records[i] for i in indices]
        x,adj=np.stack([r[0][0] for r in group]),np.stack([r[0][1] for r in group])
        b,n,_=x.shape
        mask=np.zeros((b,n),bool)
        act=np.empty(b,np.intp)
        for j,(state,choice,*_) in enumerate(group):
            mask[j,state[2]]=True
            act[j]=state[2][choice]
        old=np.array([r[2] for r in group])
        target=np.array([r[4] for r in group])
        h1,c1=mlp((adj@x).reshape(b*n,-1),model.g1)
        h2,c2=mlp((adj@h1.reshape(b,n,64)).reshape(b*n,64),model.g2)
        h=h2.reshape(b,n,64)
        pool=h.mean(axis=1)
        inp=np.concatenate((h,np.broadcast_to(pool[:,None,:],h.shape)),axis=-1)
        raw,ca=mlp(inp.reshape(b*n,128),model.actor,activation=False)
        value,cv=mlp(pool,model.critic,activation=False)
        value=value[:,0]
        z=np.where(mask,raw.reshape(b,n),-np.inf)
        prob=np.exp(z-z.max(axis=1,keepdims=True))
        prob/=prob.sum(axis=1,keepdims=True)
        logp=np.log(prob+1e-12)
        entropy=-(prob*logp).sum(axis=1)
        row=np.arange(b)
        ratio=np.exp(logp[row,act].astype(float)-old)
        a=norm[indices]
        clipped=((a>0)&(ratio>1.2))|((a<0)&(ratio<.8))
        grad=.01*prob*(logp+entropy[:,None])
        score=-prob.copy()
        score[row,act]+=1
        grad-=np.where(clipped,0,2*a*ratio)[:,None]*score
        da,ga=mlp_back(grad.reshape(-1,1),ca,model.actor)
        dv,gv=mlp_back((2*(value-target))[:,None].astype(np.float32),cv,model.critic)
        da=da.reshape(b,n,128)
        dh=da[:,:,:64]+(da[:,:,64:].sum(axis=1)+dv)[:,None,:]/n
        dh1,g2=mlp_back(dh.reshape(b*n,64),c2,model.g2)
        _,g1=mlp_back((adj.transpose(0,2,1)@dh1.reshape(b,n,64)).reshape(b*n,64),c1,model.g1)
        optimizer.step([g/b for g in g1+g2+ga+gv])
        losses.extend((-2*np.minimum(a*ratio,a*np.clip(ratio,.8,1.2))+(value-target)**2-.01*entropy).tolist())
    return float(np.mean(losses))


def fast_learned(p,agent,window=None):
    env=FastSchedule(p,graph=True)
    while env.ready:
        task=next(iter(env.ready)) if len(env.ready)==1 else agent.choose(env.node_state(agent.observation,False),
                         lambda: np.asarray(tuple(env.edges),dtype=np.intp).reshape(-1,2))
        options=p.tasks[task].options
        if len(options)==1:
            desk=options[0][0]
        elif window is None:
            desk=env.earliest(task)
        elif window=="pressure_tie":
            x,count=env.window_state(task)
            best=min(range(count),key=lambda i:(x[i,1],x[i,2],options[i][0]))
            desk=options[best][0]
        else:
            desk=options[window.choose(Schedule.window_state(env,task))][0]
        env.place(task,desk)
    return env.rows

def sequential_ppo_update(records, model, optimizer, rng):
    advantages = np.asarray([item[3] for item in records])
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    losses = []
    for minibatch in np.array_split(rng.permutation(len(records)), 4):
        gradients = [np.zeros_like(w) for w in model.parameters]
        for row in minibatch:
            state, action, old_log, _, target = records[row]
            a = normalized[row]
            p, value, cache = model.forward(state)
            ratio = np.exp(float(np.log(p[action]+1e-12)) - old_log)
            logp = np.log(p+1e-12)
            entropy = -float(p @ logp)
            gradient = .01*p*(logp + entropy)
            if not ((a > 0 and ratio > 1.2) or (a < 0 and ratio < .8)):
                score = -p.copy()
                score[action] += 1
                gradient -= 2*a*ratio*score
            dvalue = 2*(value - target)
            loss = -2*min(a*ratio, a*np.clip(ratio, .8, 1.2)) + (value-target)**2 - .01*entropy
            losses.append(loss)
            for buf, g in zip(gradients, model.backward(gradient, dvalue, cache)):
                buf += g
        optimizer.step([g/len(minibatch) for g in gradients])
    return float(np.mean(losses))