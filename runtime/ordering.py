"""Frozen selected/complement candidate alternation."""
from itertools import zip_longest
def difference_stream(inst,current,guide,universe,features,stats):
    ranked=v61.candidate_pool(inst,current,features,stats,len(current))
    remove=[int(j) for j in ranked if universe[j] and current[j] and not guide[j]]
    add=[int(j) for j in ranked if universe[j] and guide[j] and not current[j]]
    stream=[int(j) for pair in zip_longest(remove,add) for j in pair if j is not None];used=set(stream)
    stream.extend(int(j) for j in ranked if universe[j] and int(j) not in used)
    return stream
