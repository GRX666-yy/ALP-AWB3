"""Cumulative HiGHS time limits under a shared admission clock."""
import time
import highspy
from unittest.mock import patch
NEURAL=()
ARMS={"core_fixed":"e100_core"}
def solve(inst,method,seed,budget,loaded=None,train_seed=None):
    if method in ('full','wide',*NEURAL):return old.solve(inst,method,seed,budget,loaded,train_seed)
    started=time.perf_counter();fixed=not method.endswith('_legacy')
    arm=old.arm_for(len(inst.capacity),budget,POLICY) if method=='route_fixed' else ARMS[method]
    traces=[];original=v59.Logged
    class UnifiedClock(original):
        def __init__(self,instance,seconds,inner_started):super().__init__(instance,budget,started)
    def run_lp(lp,tracker,stats):
        cumulative=float(lp.getRunTime());remaining=tracker.deadline-time.perf_counter()
        if remaining<=0:return None
        limit=remaining+(cumulative if fixed else 0.)
        lp.setOptionValue('time_limit',limit);tick=time.perf_counter();lp.run();seconds=time.perf_counter()-tick
        stats['lp_s']+=seconds;stats['lp_calls']+=1;status=lp.getModelStatus();stats['lp_status'].append(str(status))
        left=tracker.deadline-time.perf_counter()
        traces.append(dict(cumulative_before_s=cumulative,outer_remaining_before_s=remaining,option_time_limit_s=limit,
            cumulative_after_s=float(lp.getRunTime()),call_wall_s=seconds,outer_remaining_after_s=left,status=str(status)))
        if status!=highspy.HighsModelStatus.kOptimal or left<=0:return None
        return lp.getSolution()
    with patch.object(v59,'Logged',UnifiedClock),patch.object(v59,'run_lp',run_lp):
        row=old.solve(inst,arm,seed,budget)
    row.update(method=method,elapsed_s=time.perf_counter()-started,budget_s=budget,lp_clock_traces=traces,
        clock_corrected=fixed,underlying_arm=arm,unified_admission_clock=True)
    if method=='route_fixed':row['routing']=dict(policy=POLICY['name'],threshold=30,backend='cc10',arm=arm,unified_admission_clock=True)
    row['overrun_s']=max(0.,row['elapsed_s']-budget)
    return row
