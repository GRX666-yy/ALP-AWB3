"""LP construction and E100 primitives; bindings supplied by common."""
import time,sys
import numpy as np
import highspy
def highs_model(inst, integer=True, seed=0):
    n, m = inst.weights.shape
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = inst.values
    lp.col_lower_, lp.col_upper_ = np.zeros(n), np.ones(n)
    lp.row_lower_, lp.row_upper_ = np.full(m, -highspy.kHighsInf), inst.capacity
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.arange(0, (n + 1) * m, m, dtype=np.int32)
    lp.a_matrix_.index_ = np.tile(np.arange(m, dtype=np.int32), n)
    lp.a_matrix_.value_ = inst.weights.ravel()
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h = highspy.Highs()
    for name, value in [('output_flag', False), ('threads', 1), ('parallel', 'off'), ('random_seed', seed)]:
        if h.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {name}={value}")
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS rejected the MDKP model")
    return h

def packed(x):return np.packbits(x).tobytes().hex()

def unpacked(s,n):return np.unpackbits(np.frombuffer(bytes.fromhex(s),dtype=np.uint8))[:n].astype(bool)

class Logged(Incumbent):
    def __init__(self,inst,budget,started):
        super().__init__(inst,budget);self.started=started;self.deadline=started+budget
        self.ledger=[[0.,0.,attr.packed(self.selected)]]
    def offer(self,x):
        prior=self.value;result=super().offer(x)
        if self.value>prior:self.ledger.append([*self.events[-1],attr.packed(self.selected)])
        return result

def ordered_fill(inst,order,tracker,initial=None):
    x=np.zeros(len(inst.values),bool) if initial is None else initial.copy()
    load=inst.weights[x].sum(0);chosen=[]
    for j in order:
        if time.perf_counter()>=tracker.deadline:break
        if not x[j] and np.all(load+inst.weights[j]<=inst.capacity+1e-8):
            x[j]=True;load+=inst.weights[j];chosen.append(int(j))
    tracker.offer(x)
    return x,chosen

def lp_order(solution):
    x=np.clip(np.asarray(solution.col_value),0.,1.)
    x[np.abs(x)<1e-7]=0.;x[np.abs(x-1.)<1e-7]=1.
    rc=np.asarray(solution.col_dual)
    return np.lexsort((np.arange(len(x)),-rc,-x)),x,rc

def run_lp(lp,tracker,stats):
    remain=tracker.deadline-time.perf_counter()
    if remain<=0:return None
    lp.setOptionValue('time_limit',remain);tick=time.perf_counter();lp.run()
    stats['lp_s']+=time.perf_counter()-tick;stats['lp_calls']+=1
    status=lp.getModelStatus();stats['lp_status'].append(str(status))
    if status!=highspy.HighsModelStatus.kOptimal or time.perf_counter()>=tracker.deadline:return None
    return lp.getSolution()

def adaptive(inst,lp,solution,tracker,stats):
    order,x,rc=lp_order(solution);selected=x==1.;L=[int(j) for j in order if selected[j]]
    load=inst.weights[selected].sum(0)
    # Check the rounded LP selection against the resource capacities.
    if np.any(load>inst.capacity+1e-8):return ordered_fill(inst,order,tracker)
    available=~selected
    tracker.offer(selected)
    while np.any(available) and time.perf_counter()<tracker.deadline:
        available &= np.all(inst.weights<=inst.capacity-load+1e-8,axis=1)
        if not np.any(available):break
        lower=selected.astype(float);upper=(selected|available).astype(float)
        lp.changeColsBounds(len(x),np.arange(len(x),dtype=np.int32),lower,upper)
        solution=run_lp(lp,tracker,stats)
        if solution is None:break
        order,_,_=lp_order(solution);j=next(int(j) for j in order if available[j])
        selected[j]=True;available[j]=False;load+=inst.weights[j];L.append(j);tracker.offer(selected)
    tracker.offer(selected)
    return selected,L

def improve_last(inst,x,L,tracker,limit,stats,il5=False):
    order=np.argsort(-inst.values,kind='stable')
    cap=min(5,len(L)) if il5 else len(L)
    additions=0
    for k in range(1,cap+1):
        if time.perf_counter()>=tracker.deadline or (not il5 and additions>=limit):break
        base=x.copy();base[L[-k:]]=False;stats['rollback_depths']+=1
        if il5:
            ordered_fill(inst,order,tracker,base);continue
        current=base.copy();load=inst.weights[base].sum(0);value=float(inst.values@base)
        ids=order[~base[order]]
        def dfs(candidates,used,profit):
            nonlocal additions
            if time.perf_counter()>=tracker.deadline or additions>=limit:return
            feasible=candidates[np.all(inst.weights[candidates]<=inst.capacity-used+1e-8,axis=1)]
            for position,j in enumerate(feasible):
                if time.perf_counter()>=tracker.deadline or additions>=limit:return
                current[j]=True;additions+=1;next_profit=profit+inst.values[j]
                if next_profit>tracker.value+1e-9:tracker.offer(current)
                dfs(feasible[position+1:],used+inst.weights[j],next_profit)
                current[j]=False
        dfs(ids,load,value)
    stats['enumeration_additions']+=additions

def stats_new():
    return dict(lp_calls=0,lp_s=0.,lp_status=[],rollback_depths=0,enumeration_additions=0,
        cardinality_attempts=0,construction_complete=False,guard_complete=False,
        wrapper_density_fallback=True,wrapper_best_single=True,solver='HiGHS simplex')

def construct(inst,kind,seed,tracker,stats,context=None):
    tick=time.perf_counter();n=len(inst.values)
    tracker.offer(greedy(inst,deadline=tracker.deadline))
    feasible=np.all(inst.weights<=inst.capacity+1e-8,axis=1)
    single=np.zeros(n,bool);ids=np.flatnonzero(feasible)
    if len(ids):single[int(ids[np.argmax(inst.values[ids])])]=True
    tracker.offer(single);stats['single_value']=inst.validate(single)
    if time.perf_counter()>=tracker.deadline:
        return tracker.selected.copy(),np.flatnonzero(tracker.selected).tolist()
    lp=highs_model(inst,False,seed);lp.setOptionValue('solver','simplex')
    if not feasible.all():
        bad=np.flatnonzero(~feasible).astype(np.int32)
        lp.changeColsBounds(len(bad),bad,np.zeros(len(bad)),np.zeros(len(bad)))
    sol=old.run_lp(lp,tracker,stats)
    if sol is None:return tracker.selected.copy(),np.flatnonzero(tracker.selected).tolist()
    if context is not None:context.update(lp=lp,solution=sol)
    order,variables,_=old.lp_order(sol)
    rounded=variables==1.
    if np.any(inst.weights.T@rounded>inst.capacity+1e-8):
        rounded=np.floor(np.clip(sol.col_value,0.,1.)).astype(bool)
        stats['rounded_tolerance_fallback']=True
    tracker.offer(rounded)
    stats.update(guard_complete=time.perf_counter()<tracker.deadline,
        initial_lp_value=float(lp.getObjectiveValue()),initial_lp_fractionals=int(np.sum((variables>0)&(variables<1))),
        rounded_value=inst.validate(rounded),rounded_selected=attr.packed(rounded))
    if kind=='adaptive':x,L=old.adaptive(inst,lp,sol,tracker,stats)
    else:x,L=old.ordered_fill(inst,order,tracker)
    stats['construction_complete']=time.perf_counter()<tracker.deadline
    # Preserve insertion order for the incumbent retained after construction.
    best=tracker.selected.copy()
    if not np.array_equal(best,x):
        L=[int(j) for j in L if best[j]];seen=set(L)
        L += [int(j) for j in order if best[j] and int(j) not in seen]
        stats['guard_replaced_lp_start']=True
    assert len(set(L))==len(L) and set(L)==set(np.flatnonzero(best))
    stats.update(construction_value=inst.validate(best),construction_selected=attr.packed(best),
        construction_order=L,construction_s=time.perf_counter()-tick,constructor_kind=kind)
    return best,L

def finish(row,tracker,started,budget,method,stats):
    row.update(method=method,budget_s=budget,value=tracker.value,selected=attr.packed(tracker.selected),
        events=tracker.ledger,stats=stats,elapsed_s=time.perf_counter()-started)
    row['overrun_s']=max(0.,row['elapsed_s']-budget)
    return row
