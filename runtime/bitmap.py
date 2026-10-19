"""Exact-profit pairing with conservative floating bounds and exact final checks."""
import time
import numpy as np

def masks(pa,wa,upper,pb,wb,capacity,constraints,best,deadline,guided,stats):
 live=np.flatnonzero(upper>best+1e-9);out=np.zeros(len(pa),dtype=object)
 if not len(live) or not len(pb):return out
 allbits=(1<<len(pb))-1;out[live]=allbits
 # Python arbitrary-width integers perform intersections in native code.
 onebits=[1<<j for j in range(len(pb))]
 if guided:
  redundant=wa[live].max(0)+wb.max(0)<=capacity+1e-8
  constraints=np.array([j for j in constraints if not redundant[j]],int)
  stats['resource_dims_skipped']=int(sum(redundant))
 stats['indexed_dimensions']=len(constraints);stats['initial_query_rows']=len(live)
 for j in constraints:
  if time.perf_counter()>=deadline:return None
  tick=time.perf_counter();order=np.argsort(wb[:,j],kind='stable');sorted_w=wb[order,j]
  prefix=[0];acc=0
  for rank,index in enumerate(order):
   if rank%128==0 and time.perf_counter()>=deadline:
    stats['index_build_s']+=time.perf_counter()-tick;return None
   acc |= onebits[int(index)];prefix.append(acc)
  stats['index_build_s']+=time.perf_counter()-tick;stats['prefix_entries']+=len(prefix)
  tick=time.perf_counter()
  # A conservative enlargement prevents subtraction-rounding false negatives.
  # The winning pair is always rechecked with original (wa+wb <= cap+TOL).
  limit=capacity[j]+1e-8
  err=8*np.finfo(float).eps*(abs(limit)+np.abs(wa[live,j])+float(np.max(np.abs(wb[:,j])))+1.)
  ranks=np.searchsorted(sorted_w,(limit-wa[live,j])+err,side='right')
  allowed=np.array([prefix[int(rank)] for rank in ranks],dtype=object)
  out[live]=np.bitwise_and(out[live],allowed)
  stats['mask_intersections']+=len(live);stats['bitmap_query_s']+=time.perf_counter()-tick
  if guided:live=live[np.fromiter((bool(x) for x in out[live]),bool,count=len(live))]
  if not len(live):break
 return out

def best_in_block(pa,wa,pb,wb,bb,capacity,ids,query_masks,best,deadline,stats):
 winner=None;value=-np.inf;key=None
 for count,a0 in enumerate(ids):
  if count%16==0 and time.perf_counter()>=deadline:return winner,False
  a=int(a0);mask=int(query_masks[a]);stats['bitmap_rows_queried']+=1
  while mask:
   b=(mask&-mask).bit_length()-1;v=float(pa[a]+pb[b])
   if v<=best+1e-9 or v<value:break
   stats['candidate_validations']+=1
   if not np.all(wa[a]+wb[b]<=capacity+1e-8):
    mask &= ~(1<<b);stats['conservative_false_positives']+=1;continue
   # Addition can collapse distinct floating profits to the same combined value.
   # Keep the original right bit-mask tie order even in that case.
   same=np.flatnonzero(pa[a]+pb==v)
   if len(same)>1:
    for t in same[np.argsort(bb[same],kind='stable')]:
     t=int(t)
     if t==b:break
     if mask&(1<<t):
      stats['candidate_validations']+=1
      if np.all(wa[a]+wb[t]<=capacity+1e-8):b=t;break
   k=(a,int(bb[b]))
   if v>value or (v==value and (key is None or k<key)):winner=(a,b,v);value=v;key=k
   break
 return winner,True
