"""Current E100 prefix dispatch; only the frozen arm is bound."""
import time
import numpy as np
NEURAL=()
LIMITS={"e100_core":100}
def prefix(inst,limit,seed,tracker,stats):
    features,_,L=v61.prefix(inst,'adaptive',seed,tracker,stats,0)
    stats.update(before_enum_value=tracker.value,before_enum_selected=attr.packed(tracker.selected),
        before_enum_elapsed_s=time.perf_counter()-tracker.started,enum_limit=limit)
    tick=time.perf_counter();calls_before=stats['enumeration_additions']
    if limit and stats['construction_complete'] and time.perf_counter()<tracker.deadline:
        x=tracker.selected.copy();v59.improve_last(inst,x,L,tracker,limit,stats)
        stats.setdefault('backend_stages',[]).append(dict(backend=f'e{limit}',seconds=time.perf_counter()-tick,
            input_value=inst.validate(x),output_value=tracker.value,output_selected=attr.packed(tracker.selected),
            completed_before_deadline=time.perf_counter()<tracker.deadline))
    stats.update(enum_seconds=time.perf_counter()-tick,enum_additions_this_stage=stats['enumeration_additions']-calls_before,
        prefix_value=tracker.value,prefix_selected=attr.packed(tracker.selected),prefix_elapsed_s=time.perf_counter()-tracker.started)
    return features

def solve(inst,method,seed,budget,loaded=None,train_seed=None,selected='omit_e'):
    if method in ('full','wide',*NEURAL):return old.solve(inst,method,seed,budget,loaded,train_seed)
    if method in ('lp_base','e10000','cc10'):return old.solve(inst,method,seed,budget)
    chosen=selected if method=='selected_short' else method
    assert chosen in LIMITS
    if chosen=='e1000_core':
        row=augment_original(old.solve(inst,'mitm_selected',seed,budget,selected='price'));row['method']=method;return row
    started=time.perf_counter();tracker=v59.Logged(inst,budget,started);stats=v60.stats_new()
    features=prefix(inst,LIMITS[chosen],seed,tracker,stats)
    if stats['construction_complete'] and time.perf_counter()<tracker.deadline and features is not None:
        old.repair(inst,tracker,features,stats,'price')
    return v60.finish(tracker.result(),tracker,started,budget,method,stats)
