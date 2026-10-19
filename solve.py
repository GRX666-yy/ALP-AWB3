from pathlib import Path
import os,sys,importlib.util,time,json,argparse
from unittest.mock import patch
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent

def _load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

_common=_load(HERE/'runtime/common.py','_awrc88_common')
_saved={k:sys.modules.get(k) for k in ('common','bitmap')}
try:
 sys.modules['common']=_common
 _bitmap=_load(HERE/'runtime/bitmap.py','_awrc88_bitmap');sys.modules['bitmap']=_bitmap
 _core=_load(HERE/'runtime/core.py','_awrc88_core');_core.MODE='guided'
 _engine=_load(HERE/'runtime/engine.py','_awrc88_engine')
finally:
 for k,m in _saved.items():
  if m is None:sys.modules.pop(k,None)
  else:sys.modules[k]=m

def solve_instance(inst,seed=0,budget_s=.05):
 """Timed solver on an existing validated Instance; calls must be sequential."""
 with patch.object(_common.v62,'optimize',_core.optimize):
  r=_engine.solve(inst,'reuse',int(seed),float(budget_s),max_rounds=3)
 r.update(method='awrc_b3',model_id='mdkp-bitmap-three-core-v88',model_version=88,
  pair_search_mode='guided',configured_max_rounds=3)
 return r

def solve(values,weights,capacity,budget_s=.05,seed=0,name='mdkp_instance'):
 tick=time.perf_counter();p=np.asarray(values,float);w=np.asarray(weights,float);c=np.asarray(capacity,float)
 if p.ndim!=1 or c.ndim!=1 or w.shape!=(len(p),len(c)) or not len(p) or not len(c):raise ValueError('Expected values[N], weights[N,M], capacity[M], N/M>0')
 if not all(np.isfinite(a).all() for a in (p,w,c)) or np.any(p<=0) or np.any(w<0) or np.any(c<=0):raise ValueError('Requires positive finite values/capacities and nonnegative finite weights')
 if not np.isfinite(budget_s) or budget_s<0:raise ValueError('Invalid budget')
 i=_common.Instance(p,w,c,str(name),'user_input');r=solve_instance(i,seed,budget_s)
 x=_common.attr.unpacked(r['selected'],len(p));assert abs(i.validate(x)-r['value'])<1e-7
 r.update(selected_ids=np.flatnonzero(x).tolist(),wrapper_elapsed_s=time.perf_counter()-tick)
 return r

def main():
 a=argparse.ArgumentParser();a.add_argument('--input',required=True,type=Path);a.add_argument('--output',type=Path)
 a.add_argument('--budget-ms',type=float,default=50.);a.add_argument('--seed',type=int,default=0);z=a.parse_args()
 d=json.loads(z.input.read_text(encoding='utf-8-sig'));r=solve(d['values'],d['weights'],d['capacity'],z.budget_ms/1000,z.seed,d.get('name','mdkp_instance'))
 out=json.dumps(r,indent=2)+'\n'
 if z.output:z.output.write_text(out,encoding='utf-8')
 else:print(out)
if __name__=='__main__':main()
