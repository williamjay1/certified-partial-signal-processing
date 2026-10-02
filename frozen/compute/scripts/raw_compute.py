"""Raw-signal software compute actions. Original frozen test signals are excluded.

Online episodes receive a raw window already resident in D-backed host memory.
Timing includes only software work after that handoff, never sensor purchase,
acquisition, transmission, or file transfer. Counterfactual donor signals are
cached TRAIN-only features, while real episode features are actually computed.
"""
from __future__ import annotations
import os, sys, json, hashlib, time, zipfile
from pathlib import Path

ROOT = Path(r'D:\MLWork\ASC_ComputeValue_20261002')
OLD = Path(r'D:\MLWork\ASC_SelectiveReplanning_20260930')
sys.path.insert(0, str(OLD / 'scripts'))
import methods as legacy
import numpy as np

SENSORS = ['PS1','PS2','PS3','PS4','PS5','PS6','EPS1','FS1','FS2','TS1','TS2','TS3','TS4','VS1']
STATS = ['mean','std','min','max','q25','q50','q75','first','last','slope_per_cycle','rms']
SPECTRAL = ['log_power','centroid_hz','spread_hz','spectral_entropy','dominant_hz','low_third_power_fraction']
WIDTHS = np.tile([11, 6], 14)
OFFSETS = np.r_[0, np.cumsum(WIDTHS)]
GROUP_INDEX = np.repeat(np.arange(28), WIDTHS)
SEED = 20261002

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

class RawBlocks:
    def __init__(self, raw, center=None, scale=None):
        self.raw = raw
        self.center = np.zeros(238, np.float32) if center is None else np.asarray(center, np.float32)
        self.scale = np.ones(238, np.float32) if scale is None else np.asarray(scale, np.float32)
        self.t = [np.linspace(-.5, .5, a.shape[1]) for a in raw]
        self.tt = [float(t@t) for t in self.t]
        self.freq = [np.fft.rfftfreq(a.shape[1], d=60./a.shape[1]) for a in raw]

    def values(self, case, block):
        j = block // 2
        x = self.raw[j][case]
        if block % 2 == 0:
            q = np.quantile(x, [.25,.5,.75])
            out = np.asarray([x.mean(),x.std(),x.min(),x.max(),*q,x[0],x[-1],
                              float(x@self.t[j])/self.tt[j],np.sqrt(np.mean(x*x))],np.float32)
        else:
            power = np.abs(np.fft.rfft(x-x.mean()))**2
            total = float(power.sum())
            norm = power / max(total, 1e-30)
            freq = self.freq[j]
            centroid = float(norm@freq)
            spread = np.sqrt(float(norm@((freq-centroid)**2)))
            ent = float(-(norm*np.log(np.maximum(norm,1e-30))).sum()/np.log(max(len(norm),2)))
            out = np.asarray([np.log1p(total/len(x)**2),centroid,spread,ent,
                              freq[int(np.argmax(power))],norm[freq<=freq[-1]/3].sum()],np.float32)
        return out

    def reveal(self, case, block, obs, mask):
        if mask[block] > .5: raise AssertionError('A block must not be computed twice')
        sl = slice(int(OFFSETS[block]), int(OFFSETS[block+1]))
        obs[sl] = (self.values(case,block)-self.center[sl])/self.scale[sl]
        mask[block] = 1

def grouped_assignment(y, units):
    """Outcome-free wrt model results: fixed-seed class-count group balancing.

    Labels are used only to ensure class support. No predictive outputs are read.
    Rows not selected for gate labels stay excluded from classifier fitting.
    """
    uniq, inv = np.unique(units, return_inverse=True)
    counts = np.zeros((len(uniq),4), int)
    np.add.at(counts,(inv,y),1)
    ratios = np.array([.53,.12,.19,.08,.08])
    target = ratios[:,None]*counts.sum(0)
    rng = np.random.default_rng(SEED)
    best = None
    for trial in range(1000):
        order = np.argsort(-counts.sum(1)*rng.uniform(.7,1.3,len(uniq)))
        allocation = np.zeros((5,4),float); assigned = np.full(len(uniq),-1,int)
        for group in order:
            scores = []
            for p in range(5):
                proposed = allocation.copy(); proposed[p] += counts[group]
                scores.append(float(((proposed-target)**2/np.maximum(target,1)).sum()))
            p = int(np.argmin(scores)); assigned[group] = p; allocation[p] += counts[group]
        capacities = allocation.sum(1)
        missing = int((allocation == 0).sum())
        short = np.maximum(np.array([1,1,300,100,100])-capacities,0).sum()
        score = float(((allocation-target)**2/np.maximum(target,1)).sum())+1e8*(missing+short)
        if best is None or score < best[0]: best = score, assigned.copy(), allocation.copy()
    if np.any(best[2] == 0) or np.any(best[2].sum(1)<[1,1,300,100,100]):
        raise RuntimeError('Unable to obtain class-complete, adequately sized group partitions')
    partition = best[1][inv]
    labels = ['predictor_fit','predictor_val','gate_fit','gate_cal','gate_audit']
    split = {label:np.flatnonzero(partition==p) for p,label in enumerate(labels)}
    for label, n in [('gate_fit',300),('gate_cal',100),('gate_audit',100)]:
        split[label+'_all'] = split[label].copy()
        split[label] = np.sort(rng.choice(split[label],n,replace=False))
    return split, partition

def prepare():
    out = ROOT/'cache'/'hydraulic_development_raw_v1.npz'
    if out.exists():
        with np.load(out,allow_pickle=False) as z: return {k:z[k] for k in z.files}
    old = np.load(OLD/'datasets'/'hydraulic_valve.npz', allow_pickle=False)
    rows = np.sort(np.r_[old['train_idx'],old['val_idx']])
    assert len(rows)==1770 and not np.intersect1d(rows,old['test_idx']).size
    y = old['y'][rows]; units = old['split_group'][rows]
    split, assignment = grouped_assignment(y,units)
    manifest = json.loads((OLD/'datasets'/'raw_download_manifest.json').read_text(encoding='utf-8'))
    entry = next(e for e in manifest['entries'] if e['tag']=='hydraulic_source' and e['status']=='complete')
    archive = Path(entry['path'])
    if digest(archive) != entry['sha256']: raise RuntimeError('Immutable source checksum mismatch')
    selected = set(map(int,rows)); raw = []
    with zipfile.ZipFile(archive) as z:
        for sensor in SENSORS:
            member = next(n for n in z.namelist() if Path(n).name.lower()==sensor.lower()+'.txt')
            values = []
            with z.open(member) as stream:
                for row,line in enumerate(stream):
                    if row in selected: values.append(np.fromstring(line.decode('ascii'),sep=' ',dtype=np.float64))
            array = np.stack(values)
            assert array.shape[0]==1770 and np.isfinite(array).all()
            raw.append(array)
            print('PARSED_DEVELOPMENT',sensor,array.shape,flush=True)
    blocks = RawBlocks(raw)
    features = np.empty((len(rows),238),np.float32)
    for i in range(len(rows)):
        for g in range(28): features[i,OFFSETS[g]:OFFSETS[g+1]] = blocks.values(i,g)
    center = features[split['predictor_fit']].mean(0)
    scale = features[split['predictor_fit']].std(0); scale[scale<1e-8]=1
    x = (features-center)/scale
    data = dict(y=y, units=units, original_rows=rows, assignment=assignment, features=features,
                X=x, center=center,scale=scale,**split,**{'raw_'+s: a for s,a in zip(SENSORS,raw)})
    np.savez(out, **data)
    report = {'source_file':str(archive),'source_sha256':entry['sha256'],'cache_sha256':digest(out),
              'development_rows':1770,'original_test_rows_loaded':0,'features':238,'blocks':28,
              'statistics':STATS,'spectral':SPECTRAL,'frequency_units':'Hz, 60-second cycle sampling rates',
              'scaler_fit':'predictor_fit only','donors':'predictor_fit only',
              'partition_rule':'Fixed-seed, whole-unit class-count balancing; gate samples without replacement after allocation; unused gate-unit cases remain excluded from classifier.',
              'split':{k:{'n':len(v),'n_units':len(np.unique(units[v])),
                          'class_counts':np.bincount(y[v],minlength=4).tolist(),
                          'units':np.unique(units[v]).tolist()} for k,v in split.items()},
              'signal_samples':[a.shape[1] for a in raw],
              'primary_partition_unit':'Contiguous identical-condition operating segments; proxy units, not independently certified experiments',
              'time_origin':'Raw 60-second signal window already in host memory from D cache',
              'excluded_time':['sensor acquisition','transmission','raw file transfer','offline fitting'],
              'immutable_test_note':'Original 435 test signals were not parsed or featurized'}
    dump(ROOT/'datasets'/'hydraulic_development_manifest.json',report)
    dump(ROOT/'datasets'/'frozen_split_indices.json',
         {k:{'development_indices':v.tolist(),'original_rows':rows[v].tolist()} for k,v in split.items()})
    return data

def fit_predictor(data, epochs=70):
    legacy.seed_all(17)
    pred = legacy.Predictor(238,GROUP_INDEX,4,seed=17)
    path = ROOT/'checkpoints'/'raw_masked_predictor_seed17.pt'
    if path.exists():
        obj = legacy.torch.load(path, map_location='cpu',weights_only=False)
        pred.net.load_state_dict(obj['weights']); pred.net.eval()
        pred.temperature=obj['temperature']; pred.fit_info=obj['fit_info']
    else:
        tr, va = data['predictor_fit'],data['predictor_val']
        pred.fit(data['X'][tr],data['y'][tr],data['X'][va],data['y'][va],epochs=epochs)
        legacy.torch.save({'weights':pred.net.state_dict(),'temperature':pred.temperature,
                           'fit_info':pred.fit_info,'seed':17,'epochs_cap':epochs},path)
    return pred

def estimate_costs(data, blocks):
    tr = data['predictor_fit']; rng=np.random.default_rng(SEED+20)
    cases=rng.choice(tr,min(40,len(tr)),replace=False)
    timings = np.empty((5,len(cases),28),float)
    for case in cases[:3]:
        for g in range(28):
            obs=np.zeros(238,np.float32);mask=np.zeros(28,np.float32);blocks.reveal(case,g,obs,mask)
    for rep in range(5):
        for i,case in enumerate(cases):
            for g in rng.permutation(28):
                obs=np.zeros(238,np.float32);mask=np.zeros(28,np.float32)
                start=time.perf_counter_ns();blocks.reveal(int(case),int(g),obs,mask)
                timings[rep,i,g]=(time.perf_counter_ns()-start)/1e6
    costs=np.median(timings,axis=(0,1))
    dump(ROOT/'results'/'block_cost_estimates.json', {'milliseconds':costs.tolist(),
         'measured_fit_cases':data['original_rows'][cases].tolist(),'repetitions':5,
         'estimator':'median warm serial block extraction plus training-only scaling',
         'estimated_feature_budget_ms':float(.3*costs.sum()),
         'not_hard_realtime':'These fixed estimates constrain block selection; observed full time is a separate endpoint.'})
    np.save(ROOT/'results'/'block_timing_repeats.npy',timings)
    return costs

def static_scores(pred, data, costs):
    rng=np.random.default_rng(SEED+21);tr=data['predictor_fit']
    take=rng.choice(tr,min(800,len(tr)),replace=False);x=data['X'][take];y=data['y'][take]
    mask=(rng.random((len(take),28))<rng.uniform(.05,.5,(len(take),1))).astype(np.float32)
    baseline=legacy.brier(pred.predict(x,mask),y)
    gain=np.zeros(28)
    for g in range(28):
        m=mask.copy();m[:,g]=1
        gain[g]=np.mean(baseline-legacy.brier(pred.predict(x,m),y))
    score=gain/costs
    order=np.argsort(-score,kind='stable')
    stats=np.arange(0,28,2);initial=int(stats[np.argmax(score[stats])])
    dump(ROOT/'checkpoints'/'static_policy.json', {'mean_immediate_risk_gain':gain.tolist(),
        'gain_per_estimated_ms':score.tolist(),'order':order.tolist(),'initial_stats_block':initial,
        'initial_sensor':SENSORS[initial//2],'gain_label_cases':'predictor_fit only',
        'feature_budget_fraction':.3,'positive_gain_stopping':True})
    return gain,order,initial

class Engine:
    def __init__(self,pred,data,blocks,costs,gain,order,initial):
        self.pred=pred;self.data=data;self.blocks=blocks;self.costs=costs
        self.gain=gain;self.order=order;self.initial=initial
        self.budget=float(.3*costs.sum());self.k=16;self.C=8
        self.donors=data['X'][data['predictor_fit']]
        self.donor_y=data['y'][data['predictor_fit']]
        self.student=None;self.gate=None;self.gate_lambda=None;self.entropy_threshold=None
        self.repeated_threshold=None;self.repeated_model=None;self.random_probs=None

    def eligible(self,mask,left):
        return (mask<.5)&(self.costs<=left+1e-12)&(self.gain>0)

    def static_action(self,mask,left):
        legal=self.eligible(mask,left)
        for g in self.order:
            if legal[g]:return int(g)
        return -1

    def base_finish(self,case,obs,mask,left,reveal=None):
        while True:
            a=self.static_action(mask,left)
            if a<0:break
            (self.blocks.reveal if reveal is None else reveal)(case,a,obs,mask);left-=self.costs[a]
        return left

    def cheap_state(self,obs,mask,left):
        sl=slice(int(OFFSETS[self.initial]),int(OFFSETS[self.initial+1]))
        legal=self.eligible(mask,left)
        return np.r_[obs[sl],mask,left/self.budget,mask.mean(),legal.mean(),
                     (self.costs[legal].min() if legal.any() else 0)/self.budget,
                     self.costs[legal].max(initial=0)/self.budget,
                     self.costs[legal].sum()/self.budget].astype(np.float64)

    def student_action(self,obs,mask,left):
        z=self.cheap_state(obs,mask,left)
        values=self.student.score(z)
        legal=self.eligible(mask,left)
        values[:28][~legal]=-np.inf
        a=int(values.argmax())
        return -1 if a==28 else a

    def planner(self,obs,mask,left,deep):
        legal=self.eligible(mask,left)
        choices=[int(g) for g in self.order if legal[g]][:self.C]
        candidates=choices+[-1]
        if not choices:return -1,{'candidate_count':1,'depth':0}
        w=mask[GROUP_INDEX]/WIDTHS[GROUP_INDEX]
        dist=((self.donors-obs)**2*w).sum(1)
        ids=np.argpartition(dist,min(self.k,len(dist))-1)[:self.k]
        hidden=np.where(mask[GROUP_INDEX]>.5,obs,self.donors[ids])
        C=len(candidates);K=len(ids)
        simulated=np.tile(obs,(C*K,1));masks=np.tile(mask,(C*K,1));labels=np.tile(self.donor_y[ids],C)
        remain=np.full(C*K,left);depths=np.zeros(C*K,int)
        pseudo=np.tile(hidden,(C,1))
        for ci,a in enumerate(candidates):
            if a>=0:
                rr=np.arange(ci*K,(ci+1)*K);sl=slice(OFFSETS[a],OFFSETS[a+1])
                simulated[rr,sl]=pseudo[rr,sl];masks[rr,a]=1;remain[rr]-=self.costs[a]
                depths[rr]+=1
                if deep:
                    for g in self.order:
                        if self.gain[g]<=0:continue
                        rr2=rr[(masks[rr,g]<.5)&(self.costs[g]<=remain[rr]+1e-12)]
                        if len(rr2):
                            sl=slice(OFFSETS[g],OFFSETS[g+1]);simulated[rr2,sl]=pseudo[rr2,sl]
                            masks[rr2,g]=1;remain[rr2]-=self.costs[g]
                            depths[rr2]+=1
        p=self.pred.predict(simulated,masks)
        risks=legacy.brier(p,labels).reshape(C,K).mean(1)
        a=candidates[int(risks.argmin())]
        return a,{'candidate_count':C,'requested_action':'deep' if deep else 'shallow',
                  'simulated_depth_mean':float(depths.mean()),'simulated_depth_max':int(depths.max()),
                  'estimated_risks':risks.tolist()}

    def episode(self,case,mode,profile=False):
        """One independently executable deployment episode, real feature work.

        No labels or uncomputed real features are accepted by routing/planning.
        Risk is calculated by caller after this timer stops.
        """
        start=time.perf_counter_ns();obs=np.zeros(238,np.float32);mask=np.zeros(28,np.float32)
        component={};aux={'mode':mode,'branch':mode,'calls':0,'fallback':False,'chosen_action':-1,
                          'gate_evaluations':0,'illegal_action_count':0,'action_sequence':[]}
        def reveal(c,a,o,m):
            tick=time.perf_counter_ns() if profile else 0
            self.blocks.reveal(c,a,o,m)
            if profile:component['feature_computation_ms']=component.get('feature_computation_ms',0)+(time.perf_counter_ns()-tick)/1e6
            aux['action_sequence'].append(int(a))
        if self.costs[self.initial]>self.budget:raise AssertionError('Initial feature exceeds estimated allowance')
        reveal(case,self.initial,obs,mask)
        left=self.budget-self.costs[self.initial]
        if not self.eligible(mask,left).any():mode='skip';aux['branch']='skip';aux['fallback']=True
        if mode in ['once','risk_only']:
            tick=time.perf_counter_ns() if profile else 0;z=self.cheap_state(obs,mask,left)
            risk,timing=self.gate.predict(z)
            branch=int(np.argmin(risk+(self.gate_lambda if mode=='once' else 0)*timing))
            mode=['skip','static','student','shallow','deep'][branch]
            if profile:component['gate_ms']=(time.perf_counter_ns()-tick)/1e6
            aux.update(branch=mode,gate_evaluations=1,fallback=mode in ['skip','static','student'])
        if mode=='random_mix':
            # Fixed case-keyed draw, independent of observed values or outcomes.
            tick=time.perf_counter_ns() if profile else 0
            rng=np.random.default_rng(SEED+47+int(case))
            mode=['skip','static','student','shallow','deep'][int(rng.choice(5,p=self.random_probs))]
            if profile:component['gate_ms']=(time.perf_counter_ns()-tick)/1e6
            aux.update(branch=mode,gate_evaluations=1,fallback=mode in ['skip','static','student'])
        if mode=='entropy':
            tick=time.perf_counter_ns() if profile else 0;p=self.pred.predict(obs[None],mask[None])[0]
            ent=float(legacy.entropy(p[None])[0]);mode='deep' if ent>=self.entropy_threshold else 'static'
            if profile:component['gate_ms']=(time.perf_counter_ns()-tick)/1e6
            aux.update(branch=mode,gate_evaluations=1,fallback=mode=='static')
        if mode=='repeated':
            # A newly fitted repeated terminal-value gate on this feature space;
            # this is NOT a reproduction of the old incompatible 154-d gate.
            while self.static_action(mask,left)>=0:
                tick=time.perf_counter_ns() if profile else 0;z=self.cheap_state(obs,mask,left)
                value=float(self.repeated_model.score(z)[0])
                if profile:component['gate_ms']=component.get('gate_ms',0)+(time.perf_counter_ns()-tick)/1e6
                aux['gate_evaluations']+=1
                if value>=self.repeated_threshold:
                    mode='deep';aux['branch']='deep';break
                a=self.static_action(mask,left)
                reveal(case,a,obs,mask);left-=self.costs[a]
            else:mode='skip';aux['branch']='static';aux['fallback']=True
        if mode in ['shallow','deep','student']:
            tick=time.perf_counter_ns() if profile else 0
            if mode=='student':a=self.student_action(obs,mask,left)
            else:
                a,diag=self.planner(obs,mask,left,deep=(mode=='deep'));aux['calls']=1
                aux['planner_diagnostics']=diag
            if profile:component['action_computation_ms']=(time.perf_counter_ns()-tick)/1e6
            aux['chosen_action']=a
            if a>=0:
                if not self.eligible(mask,left)[a]:aux['illegal_action_count']+=1;raise AssertionError('Illegal action')
                reveal(case,a,obs,mask);left-=self.costs[a]
                mode='static'
            else:mode='skip'
        if mode=='static':
            left=self.base_finish(case,obs,mask,left,reveal=reveal)
        tick=time.perf_counter_ns() if profile else 0;p=self.pred.predict(obs[None],mask[None])[0]
        if profile:component['final_prediction_ms']=(time.perf_counter_ns()-tick)/1e6
        full_ms=(time.perf_counter_ns()-start)/1e6
        aux.update(full_ms=full_ms,estimated_feature_spend_ms=self.budget-left,
                   block_count=int(mask.sum()),estimated_budget_violation=bool(left < -1e-10),
                   observed_mask=mask.astype(int).tolist())
        if profile:aux['components']=component
        if profile:aux['realized_feature_budget_violation']=bool(component.get('feature_computation_ms',0)>self.budget)
        return p,aux

class LinearScores:
    def __init__(self):self.classes=None
    def fit(self,x,y,nclasses=29):
        self.center=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1
        design=np.c_[np.ones(len(x)),(x-self.center)/self.scale]
        target=np.eye(nclasses)[np.where(y<0,nclasses-1,y)]
        penalty=np.eye(design.shape[1])*10;penalty[0,0]=0
        self.coef=np.linalg.solve(design.T@design+penalty,design.T@target)
        return self
    def score(self,z):return np.r_[1,(z-self.center)/self.scale]@self.coef

class ComputeGate:
    def fit(self,x,risk,timing):
        self.center=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1
        design=np.c_[np.ones(len(x)),(x-self.center)/self.scale]
        penalty=np.eye(design.shape[1])*10;penalty[0,0]=0
        # Baseline-relative targets share noise of same-case counterfactuals.
        self.riskcoef=np.linalg.solve(design.T@design+penalty,design.T@(risk-risk[:,1,None]))
        self.timecoef=np.linalg.solve(design.T@design+penalty,design.T@timing)
        return self
    def predict(self,z):
        design=np.r_[1,(z-self.center)/self.scale]
        return design@self.riskcoef,np.maximum(design@self.timecoef,0)

class LinearValue:
    def fit(self,x,y):
        self.center=x.mean(0);self.scale=x.std(0);self.scale[self.scale<1e-8]=1
        design=np.c_[np.ones(len(x)),(x-self.center)/self.scale]
        penalty=np.eye(design.shape[1])*10;penalty[0,0]=0
        self.coef=np.linalg.solve(design.T@design+penalty,design.T@y[:,None])
        return self
    def score(self,z):return np.r_[1,(z-self.center)/self.scale]@self.coef
