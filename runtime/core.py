"""Deadline-aware exact restricted search; no solver or model dependencies."""
import time
import numpy as np
from common import CORE
from bitmap import masks,best_in_block
MODE="bitmap"
TOL=1e-8

def states(profit,weight,capacity,deadline):
    p=np.zeros(1);w=np.zeros((1,len(capacity)));bits=np.zeros(1,np.uint32);generated=1
    for j in range(len(profit)):
        if time.perf_counter()>=deadline:return None
        pp=np.concatenate((p,p+profit[j]));ww=np.concatenate((w,w+weight[j]),axis=0)
        bb=np.concatenate((bits,bits|np.uint32(1<<j)));generated+=len(pp)
        feasible=np.all(ww<=capacity+TOL,axis=1);p,w,bits=pp[feasible],ww[feasible],bb[feasible]
    return p,w,bits,generated

def pareto_right(p,w,bits,deadline,stats):
    order=np.lexsort((bits,-p));p,w,bits=p[order],w[order],bits[order]
    keep=np.ones(len(p),bool);indices=np.arange(len(p))
    for begin in range(0,len(p),64):
        if time.perf_counter()>=deadline:return None
        target=indices[begin:begin+64]
        dominates=(indices[:,None]<target[None,:])&(p[:,None]>=p[target][None,:])
        for j in range(w.shape[1]):
            dominates &= w[:,j,None]<=w[target,j][None,:]
            if not dominates.any():break
        keep[target]=~dominates.any(0)
    stats['dominance_pruned']=int(sum(~keep))
    return p[keep],w[keep],bits[keep]

def optimize(profit,weight,capacity,initial,prices,deadline,variant,publish):
    """publish(pool_bool) returns the actually admitted pool objective value."""
    assert variant in ('plain','price','dominance') and len(profit)<=20
    tick=time.perf_counter();k=len(profit);half=k//2
    initial=np.asarray(initial,bool);best=float(profit@initial);raw_best=best
    stats=dict(variant=variant,k=k,half=half,complete=False,phase='enumeration',dominance_pruned=0,
        left_bound_pruned=0,left_rows_done=0,pair_slots=0,proposals=0,initial_pool_value=best)
    def finish(phase,complete=False):
        stats.update(phase=phase,complete=bool(complete and abs(best-raw_best)<1e-7),
            admitted_pool_value=best,raw_best_pool_value=raw_best,seconds=time.perf_counter()-tick)
        return stats
    left=CORE.states(profit[:half],weight[:half],capacity,deadline)
    right=CORE.states(profit[half:],weight[half:],capacity,deadline)
    if left is None or right is None:return finish('enumeration_deadline')
    pa,wa,ba,ga=left;pb,wb,bb,gb=right
    stats.update(left_states=len(pa),right_states_before_dominance=len(pb),generated_partial_states=ga+gb)
    if variant=='dominance':
        result=pareto_right(pb,wb,bb,deadline,stats)
        if result is None:return finish('dominance_deadline')
        pb,wb,bb=result
    stats['right_states']=len(pb)
    assert len(pa) and len(pb) # The incumbent implies the empty half is feasible.
    max_right=float(pb.max());upper=pa+max_right
    if variant!='plain':
        lam=np.maximum(np.asarray(prices),0.)
        box=float(np.maximum(profit[half:]-weight[half:]@lam,0).sum())
        upper=np.minimum(upper,pa-wa@lam+float((capacity+TOL)@lam)+box)
    upper+=1e-7*np.maximum(1.,np.abs(upper))
    order=np.lexsort((ba,-pa,-upper));pa,wa,ba,upper=pa[order],wa[order],ba[order],upper[order]
    constraints=np.argsort(capacity/np.maximum(weight.sum(0),1e-12),kind='stable')
    stats['constraint_order']=constraints.tolist()
    stats.update(pair_search_mode=MODE,index_build_s=0.,bitmap_query_s=0.,right_bound_pruned=0,
        resource_dims_skipped=0,indexed_dimensions=0,initial_query_rows=0,prefix_entries=0,
        mask_intersections=0,bitmap_rows_queried=0,candidate_validations=0,
        conservative_false_positives=0,right_prefix_slots_pruned=0)

    prep_tick=time.perf_counter()
    if MODE=='guided':
        lam=np.maximum(np.asarray(prices),0.)
        box_left=float(np.maximum(profit[:half]-weight[:half]@lam,0).sum())
        right_upper=np.minimum(pb+float(pa.max()),pb-wb@lam+float((capacity+TOL)@lam)+box_left)
        right_upper+=1e-7*np.maximum(1.,np.abs(right_upper))
        keep=right_upper>best+1e-9
        stats['right_bound_pruned']=int(sum(~keep));pb,wb,bb=pb[keep],wb[keep],bb[keep]
    stats['information_filter_s']=time.perf_counter()-prep_tick
    sort_tick=time.perf_counter()
    right_order=np.lexsort((bb,-pb));pb,wb,bb=pb[right_order],wb[right_order],bb[right_order]
    stats['right_sort_s']=time.perf_counter()-sort_tick;stats['indexed_right_states']=len(pb)
    if not len(pb):return finish('complete',time.perf_counter()<deadline)
    query_masks=None
    if MODE!='prefix':
        query_masks=masks(pa,wa,upper,pb,wb,capacity,constraints,best,deadline,MODE=='guided',stats)
        if query_masks is None:return finish('index_deadline')
    for begin in range(0,len(pa),64):
        if time.perf_counter()>=deadline:return finish('pair_deadline')
        ids=np.arange(begin,min(begin+64,len(pa)));active=upper[ids]>best+1e-9
        stats['left_bound_pruned']+=int(sum(~active));ids=ids[active]
        if not len(ids):continue
        stats['left_rows_done']+=len(ids);result=None;done=True
        if MODE=='prefix':
            # Keep original 64-left dense computation, safely trim whole right tail.
            count=int(np.count_nonzero(float(pa[ids].max())+pb>best+1e-9))
            stats['right_prefix_slots_pruned']+=len(ids)*(len(pb)-count)
            if not count:continue
            profits=pa[ids,None]+pb[None,:count];feasible=profits>best+1e-9
            for j in constraints:
                feasible &= wa[ids,j,None]+wb[None,:count,j]<=capacity[j]+TOL
                if not feasible.any():break
            stats['pair_slots']+=len(ids)*count
            if feasible.any():
                value=float(np.max(np.where(feasible,profits,-np.inf)))
                aa,bs=np.nonzero(feasible&(profits==value));q=np.lexsort((bb[bs],aa))[0]
                result=(int(ids[aa[q]]),int(bs[q]),value)
        else:
            query_tick=time.perf_counter()
            result,done=best_in_block(pa,wa,pb,wb,bb,capacity,ids,query_masks,best,deadline,stats)
            stats['bitmap_query_s']+=time.perf_counter()-query_tick
        if result is not None:
            a,b,proposed=result;raw_best=max(raw_best,proposed)
            bits=int(ba[a])|(int(bb[b])<<half);selected=(bits&(1<<np.arange(k,dtype=np.uint32)))!=0
            best=max(best,float(publish(selected)));stats['proposals']+=1
        if not done:return finish('pair_deadline')
    return finish('complete' if time.perf_counter()<deadline else 'final_deadline',time.perf_counter()<deadline)
