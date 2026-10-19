from common import *
from unittest.mock import patch
import time

def safe_bound(p,w,h,initial,prices):
 lam=np.maximum(np.asarray(prices),0.);best=float(p@initial)
 raw=min(float(p.sum()),float((h+1e-8)@lam+np.maximum(p-w@lam,0).sum()))
 safe=raw+1e-7*max(1.,abs(raw),abs(best));integer=bool(np.all(p==np.rint(p)) and abs(best)<2**52)
 skip=bool(safe<best+1.-1e-9 if integer else safe<=best+1e-9)
 return dict(window_bound_raw=raw,window_bound_safe=safe,window_integer=integer,window_incumbent=best,bound_skip=skip)

def half_states(p,w,h,deadline,ids,cache,stats):
 key=tuple(ids);cached=cache.get(key)
 if cached is not None:stats['half_cache_hits']+=1
 else:
  profits=np.zeros(1);weights=np.zeros((1,len(h)));bits=np.zeros(1,np.uint32);generated=1
  for j in range(len(p)):
   if time.perf_counter()>=deadline:return None
   profits=np.concatenate((profits,profits+p[j]));weights=np.concatenate((weights,weights+w[j]),axis=0)
   bits=np.concatenate((bits,bits|np.uint32(1<<j)));generated+=len(profits)
  cached=(profits,weights,bits,generated);cache[key]=cached;stats['half_cache_misses']+=1
 p0,w0,b0,g=cached;valid=np.all(w0<=h+1e-8,axis=1);stats['half_raw_lookups']+=len(p0)
 return p0[valid],w0[valid],b0[valid],g

def fit(h,w):
 if len(w)==0:return np.zeros(0)
 ratios=np.divide(np.maximum(h,0)[None,:],w,out=np.full_like(w,np.inf),where=w>0)
 return np.clip(ratios.min(axis=1),0,1)

def retain(inst,current,previous,new,features):
 p=inst.values;w=inst.weights;xx=np.asarray(features['x']);rc=np.asarray(features['rc']);prev=np.array(previous,int)
 order=np.lexsort((prev,-p[prev],np.abs(rc[prev]),~((xx[prev]>0)&(xx[prev]<1))))
 priority=prev[order].tolist();fractionals=int(sum((xx[prev]>0)&(xx[prev]<1)))
 protected=priority[:max(6,min(10,fractionals))];chosen=protected.copy()
 base=list(dict.fromkeys(new+chosen));selected=[j for j in base if current[j]]
 h=inst.capacity-inst.weights[current].sum(0)+w[selected].sum(0)
 universe=list(dict.fromkeys(previous+new));U=np.array([j for j in universe if not current[j]],int)
 for _ in range(min(2,10-len(chosen))):
  candidates=[j for j in priority if current[j] and j not in chosen]
  if not candidates:break
  before=fit(h,w[U]);scores=[]
  for j in candidates:
   gain=p[U]*(fit(h+w[j],w[U])-before);scores.append(float(np.sort(gain)[-4:].sum()/max(p[j],1e-12)))
  j=candidates[int(np.argmax(scores))];chosen.append(j);h=h+w[j]
 for _ in range(min(2,10-len(chosen))):
  candidates=[j for j in priority if not current[j] and j not in chosen]
  if not candidates:break
  score=p[candidates]*fit(h,w[candidates]);j=candidates[int(np.argmax(score))];chosen.append(j)
  if np.all(w[j]<=h+1e-8):h=h-w[j]
 chosen+=(j for j in priority if j not in chosen)
 chosen=chosen[:10];wanted=set(chosen);kept=[j for j in previous if j in wanted]
 assert len(kept)==10 and len(set(kept))==10
 return kept,dict(protected_ids=protected,resource_chosen_ids=[j for j in kept if j not in protected],previous_fractionals=fractionals)

def kernel(inst,t,f,variant,ids,index,cache,halfcache,kept,new,selection,factors,retain_meta):
 tick=time.perf_counter();before=t.selected.copy();ids=np.array(ids,int);fixed=before.copy();fixed[ids]=False
 key=(tuple(sorted(ids.tolist())),attr.packed(fixed));cached=cache.get(key);r,b,_=factors
 meta=dict(half_cache_hits=0,half_cache_misses=0,half_raw_lookups=0,bound_checked=False,bound_skip=False)
 fixed_value=float(inst.values@fixed);residual=inst.capacity-inst.weights[fixed].sum(0)
 if cached is not None:
  t.offer(attr.unpacked(cached[1],len(before)))
  d=dict(k=len(ids),half=len(ids)//2,variant=variant,complete=True,phase='cache',cache_hit=True,cached_from_round=cached[0],admitted_pool_value=t.value-fixed_value)
 else:
  if b:
   meta.update(safe_bound(inst.values[ids],inst.weights[ids],residual,before[ids],f['prices']));meta['bound_checked']=True
  if meta['bound_skip']:
   d=dict(k=len(ids),half=len(ids)//2,variant=variant,complete=True,phase='window_bound',cache_hit=False,admitted_pool_value=t.value-fixed_value,raw_best_pool_value=t.value-fixed_value)
  else:
   def publish(pool):
    x=fixed.copy();x[ids]=pool;t.offer(x);return t.value-fixed_value
   if r:
    parts=[ids[:len(ids)//2].tolist(),ids[len(ids)//2:].tolist()];calls=[0]
    def states(p,w,h,deadline):
     part=parts[calls[0]];calls[0]+=1;return half_states(p,w,h,deadline,part,halfcache,meta)
    with patch.object(CORE,'states',states):d=v62.optimize(inst.values[ids],inst.weights[ids],residual,before[ids],f['prices'],t.deadline,variant,publish)
   else:d=v62.optimize(inst.values[ids],inst.weights[ids],residual,before[ids],f['prices'],t.deadline,variant,publish)
   d['cache_hit']=False
 d.update(meta);d.update(round_index=index,free_ids=ids.tolist(),fixed_selected=attr.packed(fixed),before_selected=attr.packed(before),before_value=inst.validate(before),
  after_selected=attr.packed(t.selected),after_value=t.value,after_elapsed_s=time.perf_counter()-t.started,kept_ids=kept,new_ids=new,selection_seconds=selection,
  seconds=time.perf_counter()-tick+selection,retention=retain_meta,half_cache_entries=len(halfcache))
 if d['complete'] and cached is None:cache[key]=(index,d['after_selected'])
 return d

def solve(inst,arm,seed,budget,max_rounds=6):
 if arm in CONTROLS:return v76.solve(inst,arm,seed,budget)
 factors=FACTORS[arm];started=time.perf_counter();clock=v59.Logged;repair0=v62.repair
 class Outer(clock):
  def __init__(self,i,b,s):super().__init__(i,budget,started)
 def repair(i,t,f,s,variant):
  s['prefix_lp_calls']=s['lp_calls'];repair0(i,t,f,s,variant);first=s['mitm']
  first.update(round_index=0,kept_ids=[],new_ids=first['free_ids'].copy(),selection_seconds=0.,cache_hit=False,after_selected=attr.packed(t.selected),after_value=t.value,after_elapsed_s=time.perf_counter()-t.started)
  rounds=[first];seen=set(first['free_ids']);diff_seen=set();cache={(tuple(sorted(first['free_ids'])),first['fixed_selected']):(0,first['after_selected'])} if first['complete'] else {};halfcache={};stop='round_cap';guide=universe=None
  if first['complete'] and time.perf_counter()<t.deadline:
   tick=time.perf_counter();A=t.selected.copy();guide=(~A)&np.all(i.weights<=i.capacity+1e-8,axis=1);universe=A|guide
   s.update(difference_guide=attr.packed(guide),difference_universe=np.flatnonzero(universe).tolist(),difference_start_selected=attr.packed(A),guide_seconds=time.perf_counter()-tick,guide_only=True)
  for index in range(1,max_rounds):
   if not rounds[-1]['complete']:stop='incomplete_core';break
   if time.perf_counter()>=t.deadline:stop='deadline';break
   if guide is None:stop='no_guide';break
   tick=time.perf_counter();before=t.selected.copy();stream=e.old.difference_stream(i,before,guide,universe,f,s)
   new=[int(j) for j in stream if int(j) not in diff_seen][:20 if index==1 else 10]
   if not new or (index>1 and len(new)<10):stop='no_fresh_candidates';break
   retention={};kept=[] if index==1 else rounds[-1]['free_ids'][-10:]
   if index>1 and factors[2]:kept,retention=retain(i,before,rounds[-1]['free_ids'],new,f)
   ids=kept+new;selection=time.perf_counter()-tick
   if time.perf_counter()>=t.deadline:stop='selection_deadline';break
   d=kernel(i,t,f,variant,ids,index,cache,halfcache,kept,new,selection,factors,retention)
   rounds.append(d);seen.update(ids);diff_seen.update(ids)
  s.update(core_rounds=rounds,mitm=first,rolling=dict(mode=arm,max_rounds=max_rounds,rounds=len(rounds),complete_rounds=sum(d['complete'] for d in rounds),distinct_items=len(seen),
   cache_hits=sum(d.get('cache_hit',False) for d in rounds),kernel_calls=sum(not d.get('cache_hit',False) and not d.get('bound_skip',False) for d in rounds),stop_reason=stop),
   half_cache_bytes=sum(sum(a.nbytes for a in item[:3]) for item in halfcache.values()),second_total_seconds=0.,optimization_factors=list(factors))
 with patch.object(v59,'Logged',Outer),patch.object(v62,'repair',repair):row=e.old.old.old.solve(inst,'core_fixed',seed,budget)
 row.update(method=arm,elapsed_s=time.perf_counter()-started,budget_s=budget);row['overrun_s']=max(0.,row['elapsed_s']-budget)
 return row
