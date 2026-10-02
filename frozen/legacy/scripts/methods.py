"""Masked prediction, amortized acquisition, and bounded one-action refinement.
Only training donors and observed values are accepted by Planner.refine.
"""
import os, sys, time, copy, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'env_packages'))
sys.path.insert(0,str(ROOT/'env_core'))
sys.path.append(r'D:\ptenv\Lib\site-packages')
os.environ.setdefault('OMP_NUM_THREADS',os.environ.get('ASC_THREADS','1'))
os.environ.setdefault('MKL_NUM_THREADS',os.environ.get('ASC_THREADS','1'))
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
torch.set_num_threads(int(os.environ.get('ASC_THREADS','1')))
torch.set_num_interop_threads(1)

def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)

def brier(p,y):
    return .5*((p*p).sum(1)-2*p[np.arange(len(y)),y]+1)

def entropy(p):
    return -(p*np.log(np.clip(p,1e-8,1))).sum(1)

class MLP(nn.Module):
    def __init__(self,d,o,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,o))
    def forward(self,x): return self.net(x)

def train_net(net, X, y, kind, seed, epochs=70, batch=256, lr=1e-3):
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long if kind=='ce' else torch.float32)
    net.train()
    for ep in range(epochs):
        ids=rng.permutation(len(X))
        for beg in range(0,len(X),batch):
            ix=ids[beg:beg+batch]
            z=net(xt[ix])
            loss=nn.functional.cross_entropy(z,yt[ix]) if kind=='ce' else (nn.functional.mse_loss(z,yt[ix]) if kind=='mean' else nn.functional.smooth_l1_loss(z,yt[ix],beta=.05))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
    net.eval()
    return net

class Predictor:
    def __init__(self,d,groups,nclass,seed=17,width=128):
        self.group_index=np.asarray(groups,int); self.G=int(self.group_index.max())+1; self.K=nclass; self.d=d
        self.net=MLP(d+self.G,nclass,width); self.seed=seed; self.temperature=1.; self.prediction_rows=0; self.model_calls=0
    def fit(self,X,y,Xv,yv,epochs=100):
        rng=np.random.default_rng(self.seed); seed_all(self.seed)
        opt=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-4)
        x=torch.tensor(X,dtype=torch.float32); yy=torch.tensor(y,dtype=torch.long)
        # Frozen validation masks: no tuning on test records.
        vv=np.repeat(Xv,4,axis=0); vy=np.repeat(yv,4)
        vm=(rng.random((len(vv),self.G))<np.tile(np.array([.15,.35,.65,1.]),len(Xv))[:,None]).astype(np.float32)
        vin=np.concatenate([vv*vm[:,self.group_index],vm],1)
        best=np.inf; bestw=None; stale=0
        for ep in range(epochs):
            self.net.train()
            for beg in range(0,len(x),256):
                ids=rng.choice(len(x),min(256,len(x)),replace=False)
                q=rng.uniform(0.,1.,(len(ids),1))
                mm=(rng.random((len(ids),self.G))<q).astype(np.float32)
                mm[rng.random(len(ids))<.10]=1
                m=torch.tensor(mm)
                inp=torch.cat((x[ids]*m[:,self.group_index],m),1)
                loss=nn.functional.cross_entropy(self.net(inp),yy[ids])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(),5); opt.step()
            self.net.eval()
            with torch.no_grad():
                logit=self.net(torch.tensor(vin,dtype=torch.float32))
                vl=float(nn.functional.cross_entropy(logit,torch.tensor(vy)))
            if vl<best-1e-4: best=vl; bestw=copy.deepcopy(self.net.state_dict()); stale=0
            else: stale+=1
            if stale>=15 and ep>=30: break
        self.net.load_state_dict(bestw); self.net.eval()
        # Temperature calibrated over observed-mask distribution on validation only.
        with torch.no_grad(): logits=self.net(torch.tensor(vin,dtype=torch.float32))
        options=np.array([.65,.8,1.,1.25,1.6,2.])
        losses=[float(nn.functional.cross_entropy(logits/t,torch.tensor(vy))) for t in options]
        self.temperature=float(options[np.argmin(losses)])
        self.fit_info={'epochs':ep+1,'validation_ce':best,'temperature':self.temperature}
        return self
    def predict(self,X,masks):
        self.prediction_rows+=len(X); self.model_calls+=1
        inp=np.concatenate([np.asarray(X,np.float32)*masks[:,self.group_index],masks],1).astype(np.float32)
        out=[]
        with torch.no_grad():
            for beg in range(0,len(inp),8192):
                z=self.net(torch.from_numpy(inp[beg:beg+8192]))/self.temperature
                out.append(torch.softmax(z,1).numpy())
        return np.concatenate(out)
    def encode(self,X,masks): return np.concatenate([X*masks[:,self.group_index],masks,self.predict(X,masks)],1)

class GainPolicy:
    """Expected immediate Brier improvement; costs/availability condition action."""
    def __init__(self,pred,seed):
        self.pred=pred; self.seed=seed; self.net=MLP(pred.d+pred.G+pred.K,pred.G,128)
    def fit(self,X,y,nstates=6000):
        rng=np.random.default_rng(self.seed); n=min(nstates,max(2500,len(X)*3))
        ix=rng.integers(len(X),size=n); xx=X[ix]; yy=y[ix]
        m=(rng.random((n,self.pred.G))<rng.uniform(0.,.85,(n,1))).astype(np.float32)
        base=brier(self.pred.predict(xx,m),yy)
        target=np.zeros((n,self.pred.G),np.float32)
        for j in range(self.pred.G):
            mj=m.copy(); mj[:,j]=1
            target[:,j]=base-brier(self.pred.predict(xx,mj),yy)
        inp=self.pred.encode(xx,m)
        seed_all(self.seed+11); train_net(self.net,inp,target,'reg',self.seed+11,epochs=70)
        self.static=target.mean(0)
        return self
    def gains(self,x,m):
        inp=self.pred.encode(x,m)
        with torch.no_grad(): return self.net(torch.tensor(inp,dtype=torch.float32)).numpy()
    def action(self,x,m,cost,available,budget,stop=True):
        gain=self.gains(x,m)
        legal=(m<.5)&available&(cost<=budget[:,None]+1e-7)
        scores=gain/np.maximum(cost,1e-6)
        scores[~legal]=-np.inf
        a=scores.argmax(1); bad=~legal.any(1)
        if stop: bad|=scores[np.arange(len(a)),a]<=0
        a[bad]=-1
        return a

def reveal(obs,mask,hidden,actions,group_index):
    out=obs.copy(); mm=mask.copy(); rows=np.flatnonzero(actions>=0)
    for j in np.unique(actions[rows]):
        rr=rows[actions[rows]==j]; cc=np.flatnonzero(group_index==j)
        out[np.ix_(rr,cc)]=hidden[np.ix_(rr,cc)]; mm[rr,j]=1
    return out,mm

def base_rollout(policy,hidden,obs,m,cost,avail,budget,maxsteps=None):
    G=policy.pred.G; spent=np.zeros(len(obs))
    for _ in range(maxsteps or G):
        a=policy.action(obs,m,cost,avail,budget-spent)
        rows=np.flatnonzero(a>=0)
        if not len(rows): break
        obs,m=reveal(obs,m,hidden,a,policy.pred.group_index)
        spent[rows]+=cost[rows,a[rows]]
    return obs,m,spent

class Planner:
    def __init__(self,policy,donor_x,donor_y,seed=17,k=16,candidates=8):
        self.policy=policy; self.pred=policy.pred
        self.donor_x=np.asarray(donor_x,np.float32); self.donor_y=np.asarray(donor_y,int)
        self.k=k; self.candidates=candidates; self.seed=seed
        self.group_dims=np.bincount(self.pred.group_index,minlength=self.pred.G)
    def neighbors(self,obs,m):
        ids=[]
        # Distance respects only acquired groups, each group has equal weight.
        for x,mask in zip(obs,m):
            w=mask[self.pred.group_index]/self.group_dims[self.pred.group_index]
            dist=((self.donor_x-x)**2*w).sum(1)
            if mask.sum()==0:
                rng=np.random.default_rng(self.seed); ix=rng.choice(len(dist),self.k,replace=len(dist)<self.k)
            else:
                kk=min(self.k,len(dist)); ix=np.argpartition(dist,kk-1)[:kk]
                if kk<self.k: ix=np.resize(ix,self.k)
            ids.append(ix)
        return np.asarray(ids)
    def refine(self,obs,m,cost,avail,budget):
        n=len(obs); G=self.pred.G
        legal=(m<.5)&avail&(cost<=budget[:,None]+1e-7)
        scores=self.policy.gains(obs,m)/np.maximum(cost,1e-6); scores[~legal]=-np.inf
        base=self.policy.action(obs,m,cost,avail,budget)
        C=min(self.candidates,G)
        cand=np.full((n,C+1),-1,int)
        # Base action plus high score and deterministic exploratory candidates.
        for i in range(n):
            choices=np.flatnonzero(legal[i])
            top=choices[np.argsort(-scores[i,choices])[:max(1,C-2)]]
            remaining=np.setdiff1d(choices,top)
            # Exploration depends only on observed state, never hidden values.
            h=int(abs(float(np.sum(obs[i])))*100003+int(m[i].sum())*313+self.seed)%2**32
            rng=np.random.default_rng(h)
            extra=rng.choice(remaining,min(2,len(remaining)),replace=False) if len(remaining) else []
            arr=list(dict.fromkeys([int(base[i])]+list(map(int,top))+list(map(int,extra))))
            arr=[j for j in arr if j>=0][:C]
            cand[i,:len(arr)]=arr
        ix=self.neighbors(obs,m)
        dx=self.donor_x[ix]; dy=self.donor_y[ix]
        # Common conditional pseudo-records for every candidate.
        pseudo=np.where(m[:,None,self.pred.group_index]>.5,obs[:,None,:],dx)
        hidden=np.repeat(pseudo[:,None,:,:],C+1,axis=1).reshape(-1,self.pred.d)
        labels=np.repeat(dy[:,None,:],C+1,axis=1).reshape(-1)
        oo=np.repeat(obs[:,None,None,:],(C+1)*self.k,axis=1).reshape(-1,self.pred.d)
        mm=np.repeat(m[:,None,None,:],(C+1)*self.k,axis=1).reshape(-1,G)
        cc=np.repeat(cost,(C+1)*self.k,axis=0); aa=np.repeat(avail,(C+1)*self.k,axis=0)
        bb=np.repeat(budget,(C+1)*self.k); act=np.repeat(cand,self.k,axis=1).reshape(-1)
        oo,mm=reveal(oo,mm,hidden,act,self.pred.group_index)
        active=act>=0
        bb[active]-=cc[np.arange(len(act))[active],act[active]]
        # STOP stays frozen. Others follow the exact same base continuation.
        aa[~active]=False
        oo,mm,sp=base_rollout(self.policy,hidden,oo,mm,cc,aa,bb)
        risk=brier(self.pred.predict(oo,mm),labels).reshape(n,C+1,self.k).mean(2)
        # Invalid padded actions duplicate STOP and cannot beat it differently.
        best=risk.argmin(1); chosen=cand[np.arange(n),best]
        basepos=np.array([np.flatnonzero(cand[i]==base[i])[0] if np.any(cand[i]==base[i]) else C for i in range(n)])
        improvement=risk[np.arange(n),basepos]-risk[np.arange(n),best]
        return chosen, {'estimated_improvement':improvement,'risk':risk,'candidates':cand,
                        'donor_trajectories':np.full(n,(C+1)*self.k),'rollout_max':G}

def state_features(policy,obs,m,cost,avail,budget,fullcost):
    p=policy.pred.predict(obs,m); gains=policy.gains(obs,m)
    legal=(m<.5)&avail&(cost<=budget[:,None]+1e-7)
    score=gains/np.maximum(cost,1e-6); score[~legal]=-1
    order=np.sort(score,axis=1)
    z=np.column_stack([budget/fullcost, m.mean(1),entropy(p),order[:,-1],order[:,-1]-order[:,-2] if m.shape[1]>1 else order[:,-1]])
    return np.concatenate([obs*m[:,policy.pred.group_index],m,p,cost/(fullcost/m.shape[1]),avail.astype(float),z],1).astype(np.float32)

class GateStudent:
    def __init__(self,policy,seed):
        self.policy=policy; self.seed=seed; d=policy.pred.d+3*policy.pred.G+policy.pred.K+5
        self.gate=MLP(d,1,128); self.student=MLP(d,policy.pred.G+1,256)
    def fit(self,features,gains,actions):
        seed_all(self.seed)
        train_net(self.gate,features,np.asarray(gains,np.float32)[:,None],'mean',self.seed,epochs=90)
        target=np.where(actions<0,self.policy.pred.G,actions)
        train_net(self.student,features,target,'ce',self.seed+2,epochs=90)
    def values(self,obs,m,cost,avail,budget,fullcost):
        inp=state_features(self.policy,obs,m,cost,avail,budget,fullcost)
        with torch.no_grad(): return self.gate(torch.tensor(inp)).numpy().ravel()
    def action(self,obs,m,cost,avail,budget,fullcost):
        inp=state_features(self.policy,obs,m,cost,avail,budget,fullcost)
        with torch.no_grad(): z=self.student(torch.tensor(inp)).numpy()
        legal=(m<.5)&avail&(cost<=budget[:,None]+1e-7)
        z[:,:self.policy.pred.G][~legal]=-np.inf
        a=z.argmax(1); a[a==self.policy.pred.G]=-1
        return a

def make_scenarios(n,basecost,seed,kind,budget_fraction):
    rng=np.random.default_rng(seed); G=len(basecost)
    if kind=='fixed': factors=np.ones((n,G)); avail=np.ones((n,G),bool)
    elif kind=='support': factors=np.exp(rng.uniform(np.log(.5),np.log(2.),(n,G))); avail=rng.random((n,G))>.1
    elif kind=='stress': factors=np.exp(rng.uniform(np.log(.25),np.log(4.),(n,G))); avail=rng.random((n,G))>.3
    else: raise ValueError(kind)
    cost=basecost[None,:]*factors
    budget=np.full(n,budget_fraction*sum(basecost))
    return cost,avail,budget

def sample_training_states(policy,X,y,basecost,seed,n=900):
    rng=np.random.default_rng(seed); idx=rng.integers(len(X),size=n); hidden=X[idx]; labels=y[idx]
    cost,avail,budget=make_scenarios(n,basecost,seed+7,'support',.30)
    budget=rng.choice([.15,.30,.5],n)*sum(basecost)
    obs=np.zeros_like(hidden); m=np.zeros((n,policy.pred.G),np.float32); spent=np.zeros(n)
    # Sample actual prefixes of the base policy; includes early and late states.
    depths=rng.integers(0,min(7,policy.pred.G),n)
    for t in range(depths.max(initial=0)):
        a=policy.action(obs,m,cost,avail,budget-spent)
        a[depths<=t]=-1
        rr=np.flatnonzero(a>=0); obs,m=reveal(obs,m,hidden,a,policy.pred.group_index)
        spent[rr]+=cost[rr,a[rr]]
    return hidden,labels,obs,m,cost,avail,budget-spent

def label_states(policy,planner,X,y,basecost,seed,n=900,batch=64):
    hidden,labels,obs,m,cost,avail,budget=sample_training_states(policy,X,y,basecost,seed,n)
    oo,mm,_=base_rollout(policy,hidden,obs.copy(),m.copy(),cost,avail,budget)
    baseline=brier(policy.pred.predict(oo,mm),labels)
    actions=[]; refined=[]
    for start in range(0,n,batch):
        sl=slice(start,min(start+batch,n))
        ac,_=planner.refine(obs[sl],m[sl],cost[sl],avail[sl],budget[sl]); actions.append(ac)
        ro,rm=reveal(obs[sl],m[sl],hidden[sl],ac,policy.pred.group_index)
        rb=budget[sl].copy(); rr=np.flatnonzero(ac>=0); rb[rr]-=cost[sl][rr,ac[rr]]
        av=avail[sl].copy(); av[ac<0]=False
        ro,rm,_=base_rollout(policy,hidden[sl],ro,rm,cost[sl],av,rb)
        refined.append(brier(policy.pred.predict(ro,rm),labels[sl]))
    target=baseline-np.concatenate(refined)
    feat=state_features(policy,obs,m,cost,avail,budget,float(sum(basecost)))
    return feat,target,np.concatenate(actions), {'mean_gain':float(target.mean()),'positive_fraction':float((target>0).mean()),'n':n}

def train_system(X,y,tr,va,groups,basecost,seed,out,nstates=700,pred_epochs=90,split_groups=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); seed_all(seed)
    info={'seed':seed,'gate_loss':'MSE','version':'mean_v2','started':time.time()}
    pred=Predictor(X.shape[1],groups,len(np.unique(y)),seed).fit(X[tr],y[tr],X[va],y[va],epochs=pred_epochs)
    policy=GainPolicy(pred,seed+100).fit(X[tr],y[tr]); policy.fullcost=float(sum(basecost))
    folds=StratifiedKFold(2,shuffle=True,random_state=seed) if split_groups is None else StratifiedGroupKFold(2,shuffle=True,random_state=seed)
    fs=[]; ts=[]; acts=[]; cross=[]
    for fi,(fit,hold) in enumerate(folds.split(X[tr],y[tr], None if split_groups is None else split_groups[tr])):
        fx=X[tr][fit]; fy=y[tr][fit]; hx=X[tr][hold]; hy=y[tr][hold]
        # Validation for fold classifier is carved from the fitting fold only.
        if split_groups is None:
            tt,vv=next(StratifiedKFold(5,shuffle=True,random_state=seed+fi).split(fx,fy))
        else:
            fg=split_groups[tr][fit]
            tt,vv=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed+fi).split(fx,fy,fg))
            assert not np.intersect1d(fg[tt],fg[vv]).size
        fcenter=fx[tt].mean(0); fscale=fx[tt].std(0); fscale[fscale<1e-6]=1
        fx=(fx-fcenter)/fscale; hx=(hx-fcenter)/fscale
        fp=Predictor(X.shape[1],groups,pred.K,seed+fi+500).fit(fx[tt],fy[tt],fx[vv],fy[vv],epochs=pred_epochs)
        bp=GainPolicy(fp,seed+fi+600).fit(fx[tt],fy[tt],nstates=4500)
        pl=Planner(bp,fx[tt],fy[tt],seed+fi+700)
        f,t,a,i=label_states(bp,pl,hx,hy,basecost,seed+fi+800,nstates)
        # Return raw observed coordinates to the common full-training coordinate system.
        fm=f[:,X.shape[1]:X.shape[1]+pred.G]
        f[:,:X.shape[1]]=(f[:,:X.shape[1]]*fscale+fcenter)*fm[:,groups]
        fs.append(f);ts.append(t);acts.append(a);cross.append(i)
    np.savez_compressed(out/'crossfit_labels.npz',features=np.concatenate(fs),gains=np.concatenate(ts),actions=np.concatenate(acts))
    gate=GateStudent(policy,seed+900); gate.fit(np.concatenate(fs),np.concatenate(ts),np.concatenate(acts))
    planner=Planner(policy,X[tr],y[tr],seed+700)
    info.update({'seconds':time.time()-info['started'],'predictor':pred.fit_info,'crossfit':cross})
    torch.save({'pred':pred.net.state_dict(),'policy':policy.net.state_dict(),'gate':gate.gate.state_dict(),
                'student':gate.student.state_dict(),'temperature':pred.temperature,'static':policy.static,
                'info':info},out/'models.pt')
    (out/'training.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    return pred,policy,planner,gate,info

def run_episodes(policy,planner,gate,hidden,cost,avail,budget,method,threshold=.0,entropy_threshold=.65,random_rate=.25,seed=1,chunk=64):
    """Environment owns hidden. Planner/gate receive only observed values."""
    G=policy.pred.G; fullcost=getattr(policy,'fullcost',float(np.mean(cost.sum(1))))
    allp=[]; allspent=[]; allcount=[]; allsearched=[]; alltime=[]; allm=[]; allnt=[]
    for start in range(0,len(hidden),chunk):
        sl=slice(start,min(start+chunk,len(hidden))); hh=hidden[sl]; cc=cost[sl]; av=avail[sl]; bb=budget[sl]
        n=len(hh); obs=np.zeros_like(hh); m=np.zeros((n,G),np.float32); spent=np.zeros(n)
        used=np.zeros(n,bool); dead=np.zeros(n,bool); nt=np.zeros(n,int)
        rng=np.random.default_rng(seed+start); t0=time.perf_counter()
        for t in range(G):
            remaining=bb-spent
            if method=='random':
                legal=(m<.5)&av&(cc<=remaining[:,None]+1e-7)
                score=rng.random((n,G));score[~legal]=-1; a=score.argmax(1);a[~legal.any(1)]=-1
            elif method=='static':
                legal=(m<.5)&av&(cc<=remaining[:,None]+1e-7)
                score=policy.static[None,:]/cc;score[~legal]=-np.inf;a=score.argmax(1);a[~legal.any(1)]=-1
            elif method=='distilled': a=gate.action(obs,m,cc,av,remaining,fullcost)
            elif method=='greedy_fill': a=policy.action(obs,m,cc,av,remaining,stop=False)
            else: a=policy.action(obs,m,cc,av,remaining)
            a[dead]=-1
            if method=='distilled_once' and t>=1:
                eligible=(~used)&(~dead)&((m<.5)&av&(cc<=remaining[:,None]+1e-7)).any(1)
                rr=np.flatnonzero(eligible)
                if len(rr):
                    a[rr]=gate.action(obs[rr],m[rr],cc[rr],av[rr],remaining[rr],fullcost)
                    used[rr]=True
            if method in ('always','gain','entropy','random_gate') and t>=1:
                eligible=(~used)&(~dead)&((m<.5)&av&(cc<=remaining[:,None]+1e-7)).any(1)
                if method=='gain': eligible&=gate.values(obs,m,cc,av,remaining,fullcost)>threshold
                elif method=='entropy': eligible&=entropy(policy.pred.predict(obs,m))>entropy_threshold
                elif method=='random_gate': eligible&=rng.random(n)<random_rate
                rr=np.flatnonzero(eligible)
                if len(rr):
                    aa,diag=planner.refine(obs[rr],m[rr],cc[rr],av[rr],remaining[rr])
                    a[rr]=aa;used[rr]=True;nt[rr]=diag['donor_trajectories']
            dead|=a<0
            rows=np.flatnonzero(a>=0)
            if not len(rows): break
            obs,m=reveal(obs,m,hh,a,policy.pred.group_index)
            spent[rows]+=cc[rows,a[rows]]
        p=policy.pred.predict(obs,m);dt=time.perf_counter()-t0
        allp.append(p);allspent.append(spent);allcount.append(m.sum(1));allsearched.append(used&(nt>0));allnt.append(nt);alltime.append(np.full(n,dt/n));allm.append(m); 
    return {'p':np.concatenate(allp),'cost':np.concatenate(allspent),'n_groups':np.concatenate(allcount),
            'searched':np.concatenate(allsearched),'amortized_seconds':np.concatenate(alltime),'mask':np.concatenate(allm),'donor_trajectories':np.concatenate(allnt)}


