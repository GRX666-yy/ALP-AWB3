"""First-core wrapper and default half enumeration."""
import time
import numpy as np
TOL=1e-8
optimize=None
def states(profit,weight,capacity,deadline):
    p=np.zeros(1);w=np.zeros((1,len(capacity)));bits=np.zeros(1,np.uint32);generated=1
    for j in range(len(profit)):
        if time.perf_counter()>=deadline:return None
        pp=np.concatenate((p,p+profit[j]));ww=np.concatenate((w,w+weight[j]),axis=0)
        bb=np.concatenate((bits,bits|np.uint32(1<<j)));generated+=len(pp)
        feasible=np.all(ww<=capacity+TOL,axis=1);p,w,bits=pp[feasible],ww[feasible],bb[feasible]
    return p,w,bits,generated

def repair(inst,tracker,features,stats,variant):
    tick=time.perf_counter();before=tracker.selected.copy();ids=old.candidate_pool(inst,before,features,stats,20)
    fixed=before.copy();fixed[ids]=False;fixed_value=float(inst.values@fixed)
    residual=inst.capacity-inst.weights[fixed].sum(0)
    def publish(pool):
        x=fixed.copy();x[ids]=pool;tracker.offer(x)
        return tracker.value-fixed_value
    result=optimize(inst.values[ids],inst.weights[ids],residual,before[ids],features['prices'],tracker.deadline,variant,publish)
    result.update(free_ids=ids.tolist(),fixed_selected=attr.packed(fixed),before_selected=attr.packed(before),
        before_value=inst.validate(before),seconds=time.perf_counter()-tick)
    stats.update(mitm=result,lp_x=features['x'].tolist(),lp_rc=features['rc'].tolist())
