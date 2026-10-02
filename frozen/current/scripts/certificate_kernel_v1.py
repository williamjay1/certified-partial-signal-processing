"""Compiled frozen numeric HGB bounds and relative half-Brier certificates.

No hidden features or labels are passed to the online bounds/certificate kernels.
Numerical guards implement conservative engineering checks, not formal verified
floating-point arithmetic. The exact-arithmetic theorem is in METHOD_DERIVATION.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'cache'/'compiled')
sys.path.insert(0, str(ROOT/'cache'/'numba_env'))
import numpy as np
from numba import njit

@njit(cache=True)
def update_bounds(block, observed, known, feature, group, threshold, left,
                  right, leaf, missing_left, affected, start, stop, lower, upper):
    for k in range(start, stop):
        node = affected[k]
        if leaf[node]:
            continue
        if known[group[node]]:
            x = observed[feature[node]]
            go_left = missing_left[node] if np.isnan(x) else x <= threshold[node]
            child = left[node] if go_left else right[node]
            lower[node] = lower[child]
            upper[node] = upper[child]
        else:
            lower[node] = min(lower[left[node]], lower[right[node]])
            upper[node] = max(upper[left[node]], upper[right[node]])

@njit(cache=True)
def score_intervals(roots, classes, baseline, lower, upper, root_abs_bound):
    lo = baseline.copy()
    hi = baseline.copy()
    collapsed = True
    absolute = np.abs(baseline.copy())
    counts = np.zeros(len(baseline), np.int64)
    for t in range(len(roots)):
        root = roots[t]
        c = classes[t]
        lo[c] += lower[root]
        hi[c] += upper[root]
        absolute[c] += root_abs_bound[t]
        counts[c] += 1
        if lower[root] != upper[root]:
            collapsed = False
    for c in range(len(baseline)):
        neps = counts[c] * np.finfo(np.float64).eps
        guard = 8.0 * neps / (1.0-neps) * absolute[c] + 1e-12
        lo[c] = np.nextafter(lo[c]-guard, -np.inf)
        hi[c] = np.nextafter(hi[c]+guard, np.inf)
    return lo, hi, collapsed

@njit(cache=True)
def probability_intervals(lo, hi):
    k = len(lo)
    l = np.empty(k, np.float64)
    u = np.empty(k, np.float64)
    for j in range(k):
        shift_l = lo[j]
        shift_u = hi[j]
        for h in range(k):
            if h != j:
                shift_l = max(shift_l, hi[h])
                shift_u = max(shift_u, lo[h])
        num_l = np.exp(lo[j]-shift_l)
        num_u = np.exp(hi[j]-shift_u)
        dl = num_l
        du = num_u
        for h in range(k):
            if h != j:
                dl += np.exp(hi[h]-shift_l)
                du += np.exp(lo[h]-shift_u)
        l[j] = max(0.0, np.nextafter(num_l/dl-2e-13, -np.inf))
        u[j] = min(1.0, np.nextafter(num_u/du+2e-13, np.inf))
    if (not np.all(np.isfinite(l))) or (not np.all(np.isfinite(u))) or np.sum(l) > 1.0 or np.sum(u) < 1.0 or np.any(l > u):
        # Fail closed, rather than repairing an impossible set by normalization.
        l[:] = 0.0
        u[:] = 1.0
    return l, u

@njit(cache=True)
def simplex_projection(a):
    sorted_a = np.sort(a)[::-1]
    cum = 0.0
    theta = 0.0
    for j in range(len(a)):
        cum += sorted_a[j]
        candidate = (cum-1.0)/(j+1)
        if sorted_a[j] > candidate:
            theta = candidate
    p = np.maximum(a-theta, 0.0)
    p /= np.sum(p)
    # Emit the actual vector at which the conservative primal bound is evaluated.
    largest = np.argmax(p)
    p[largest] += 1.0-np.sum(p)
    return p

@njit(cache=True)
def distance_dual_lower(l, u):
    k = len(l)
    d = np.empty(k, np.float64)
    for y in range(k):
        low = -1.0
        high = 1.0
        for repeat in range(48):
            nu = .5*(low+high)
            total = 0.0
            for j in range(k):
                e = 1.0 if j == y else 0.0
                total += min(u[j], max(l[j], e-nu))
            if total > 1.0:
                low = nu
            else:
                high = nu
        nu = .5*(low+high)
        total = 0.0
        loss = 0.0
        for j in range(k):
            e = 1.0 if j == y else 0.0
            v = min(u[j], max(l[j], e-nu))
            total += v
            loss += .5*(v-e)*(v-e)
        dual = loss+nu*(total-1.0)
        d[y] = max(0.0, np.nextafter(dual-2e-12, -np.inf))
    return d

@njit(cache=True)
def certificate_at(a, p):
    bound = .5*np.dot(p,p)+np.max(a-p)+2e-11
    return np.nextafter(max(0.0,bound), np.inf)

@njit(cache=True)
def choose_probability(l, u, method):
    """0=tight minimax;1=coordinate relaxed;2=tight with midpoint output."""
    if method == 1:
        # a_y >= max_Q(q_y-.5||q||²), relaxing cross-coordinate coupling.
        a = np.empty(len(l), np.float64)
        sum_l2 = np.dot(l,l)
        for y in range(len(l)):
            a[y] = u[y]-.5*u[y]*u[y]-.5*(sum_l2-l[y]*l[y])+2e-12
        p = simplex_projection(a)
    else:
        a = .5-distance_dual_lower(l,u)
        if method == 2:
            p = .5*(l+u)
            p /= np.sum(p)
            p[np.argmax(p)] += 1.0-np.sum(p)
        else:
            p = simplex_projection(a)
    return p, certificate_at(a,p)

@njit(cache=True)
def exact_scores(observed, roots, classes, baseline, feature, threshold,
                 left, right, leaf, missing_left, values):
    scores = baseline.copy()
    for t in range(len(roots)):
        node = roots[t]
        while not leaf[node]:
            x = observed[feature[node]]
            go_left = missing_left[node] if np.isnan(x) else x <= threshold[node]
            node = left[node] if go_left else right[node]
        scores[classes[t]] += values[node]
    return scores

@njit(cache=True)
def stable_softmax(scores):
    p = np.exp(scores-np.max(scores))
    p /= np.sum(p)
    return p

class FrozenForest:
    def __init__(self, model, group_index, n_blocks=28):
        assert model.n_trees_per_iteration_ == len(model.classes_), 'Multiclass prototype only'
        roots=[]; classes=[]; arrays=[]; parents=[]
        offset=0
        for stage in model._predictors:
            for c, tree in enumerate(stage):
                a=tree.nodes.copy()
                assert not np.any(a['is_categorical']), 'Only numerical splits supported'
                roots.append(offset); classes.append(c)
                parent=np.full(len(a),-1,np.int64)
                for node in np.flatnonzero(a['is_leaf']==0):
                    parent[a['left'][node]]=node+offset
                    parent[a['right'][node]]=node+offset
                a['left']+=offset; a['right']+=offset
                arrays.append(a);parents.append(parent);offset+=len(a)
        a=np.concatenate(arrays)
        self.roots=np.asarray(roots,np.int64)
        self.classes=np.asarray(classes,np.int64)
        self.baseline=np.asarray(model._baseline_prediction[0],np.float64)
        self.feature=a['feature_idx'].astype(np.int64)
        self.group=np.asarray(group_index[self.feature],np.int64)
        self.threshold=a['num_threshold'].astype(np.float64)
        self.left=a['left'].astype(np.int64); self.right=a['right'].astype(np.int64)
        self.leaf=a['is_leaf'].astype(np.bool_)
        self.missing_left=a['missing_go_to_left'].astype(np.bool_)
        self.values=a['value'].astype(np.float64)
        self.parent=np.concatenate(parents)
        nonleaf=np.flatnonzero(~self.leaf)
        assert np.all(self.left[nonleaf]>nonleaf) and np.all(self.right[nonleaf]>nonleaf), 'Reverse update requires children after parents'
        lower=self.values.copy();upper=self.values.copy()
        for node in range(len(a)-1,-1,-1):
            if not self.leaf[node]:
                lower[node]=min(lower[self.left[node]],lower[self.right[node]])
                upper[node]=max(upper[self.left[node]],upper[self.right[node]])
        self.static_lower=lower;self.static_upper=upper
        self.root_abs=np.maximum(np.abs(lower[self.roots]),np.abs(upper[self.roots]))
        self.gains=np.zeros(n_blocks,np.float64)
        np.add.at(self.gains,self.group[~self.leaf],np.maximum(a['gain'][~self.leaf],0))
        self.used=np.unique(self.group[~self.leaf])
        affected=[]; offsets=[0]
        for block in range(n_blocks):
            active=set()
            for node in np.flatnonzero((self.group==block)&~self.leaf):
                while node >= 0:
                    active.add(int(node)); node=self.parent[node]
            affected.extend(sorted(active,reverse=True));offsets.append(len(affected))
        self.affected=np.asarray(affected,np.int64);self.affected_offsets=np.asarray(offsets,np.int64)

    def update(self, block, obs, known, lower, upper):
        update_bounds(block,obs,known,self.feature,self.group,self.threshold,
                      self.left,self.right,self.leaf,self.missing_left,self.affected,
                      self.affected_offsets[block],self.affected_offsets[block+1],lower,upper)

    def intervals(self, lower, upper):
        lo,hi,collapse=score_intervals(self.roots,self.classes,self.baseline,
                                     lower,upper,self.root_abs)
        l,u=probability_intervals(lo,hi)
        return lo,hi,l,u,collapse

    def exact(self, obs):
        scores=exact_scores(obs,self.roots,self.classes,self.baseline,self.feature,
                            self.threshold,self.left,self.right,self.leaf,
                            self.missing_left,self.values)
        return stable_softmax(scores)
