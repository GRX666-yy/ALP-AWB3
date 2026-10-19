"""LP boundary features and candidate ordering."""
import time
import numpy as np
def extract(inst,context,stats):
    sol=context['solution'];x=np.clip(np.asarray(sol.col_value),0.,1.)
    x[np.abs(x)<1e-7]=0.;x[np.abs(x-1)<1e-7]=1.
    prices=np.maximum(np.asarray(sol.row_dual),0.)
    upper=float(prices@inst.capacity+np.maximum(inst.values-inst.weights@prices,0).sum())
    upper=max(upper,stats['initial_lp_value']);upper+=1e-7*max(1.,abs(upper))
    result=dict(x=x,rc=np.asarray(sol.col_dual),prices=prices,upper=upper)
    stats.update(lp_price_upper=upper,lp_prices=prices.tolist(),lp_fractionals=np.flatnonzero((x>0)&(x<1)).tolist())
    return result

def prefix(inst,kind,seed,tracker,stats,limit=1000):
    context={};x,L=old.construct(inst,kind,seed,tracker,stats,context)
    clock_flag=stats['construction_complete'];load=inst.weights[tracker.selected].sum(0)
    fillable=int(np.sum((~tracker.selected)&np.all(inst.weights<=inst.capacity-load+1e-8,axis=1)))
    statuses_ok=bool(stats['lp_status']) and all(s=='HighsModelStatus.kOptimal' for s in stats['lp_status'])
    stats.update(source_clock_complete=clock_flag,construction_fillable=fillable,
        construction_complete=clock_flag and statuses_ok and fillable==0)
    features=extract(inst,context,stats) if context else None
    if stats['construction_complete'] and time.perf_counter()<tracker.deadline and limit:
        old.run_backend(inst,'e10000' if limit==10000 else 'e1000',x,L,tracker,stats,seed,tracker.deadline-tracker.started)
    stats.update(prefix_value=tracker.value,prefix_selected=attr.packed(tracker.selected),
        prefix_elapsed_s=time.perf_counter()-tracker.started)
    return features,context,L

def candidate_pool(inst,x,features,stats,k):
    rc=np.asarray(features['rc']);xx=np.asarray(features['x']);n=len(x)
    order=np.lexsort((np.arange(n),-inst.values,np.abs(rc)))
    rounded=attr.unpacked(stats['rounded_selected'],n)
    frac=(xx>0)&(xx<1);changed=x!=rounded
    result=[];seen=set()
    for ids in (order[frac[order]],order[changed[order]],order):
        for j in ids:
            j=int(j)
            if j not in seen:result.append(j);seen.add(j)
            if len(result)>=min(k,n):return np.asarray(result,dtype=int)
    return np.asarray(result,dtype=int)
