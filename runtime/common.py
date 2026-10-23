"""Assemble the algorithm dependencies from local runtime modules."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def _load(filename,bindings=None):
 name=__name__+'_'+filename.removesuffix('.py')
 spec=importlib.util.spec_from_file_location(name,HERE/filename)
 module=importlib.util.module_from_spec(spec)
 if bindings:module.__dict__.update(bindings)
 sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:
  sys.modules.pop(name,None);raise
 return module
_problem=_load('problem.py')
Instance=_problem.Instance
attr=SimpleNamespace()
v59=_load('lp.py',dict(Incumbent=_problem.Incumbent,greedy=_problem.greedy,attr=attr))
attr.packed=v59.packed;attr.unpacked=v59.unpacked
v59.old=v59
_candidates=_load('candidates.py',dict(old=v59,attr=attr))
v62=_load('repair.py',dict(old=_candidates,attr=attr))
CORE=v62
_prefix=_load('prefix.py',dict(old=v62,v61=_candidates,v60=v59,v59=v59,attr=attr))
_clock=_load('clock.py',dict(old=_prefix,v59=v59))
_ordering=_load('ordering.py',dict(v61=_candidates))
_ordering.old=SimpleNamespace(old=_clock)
e=SimpleNamespace(old=_ordering)
FACTORS={'reuse':(1,0,0)}
CONTROLS=()
