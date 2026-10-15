"""Record actual API solve operand, preserving every numerical call and return."""
def compliance_with_displacement(solver,*args,**kwargs):
 import numpy as np
 original=solver.solve;had='solve' in solver.__dict__;previous=solver.__dict__.get('solve');record=[]
 def observed(*a,**k):
  answer=original(*a,**k)
  if isinstance(answer,tuple):raise ValueError('Profile solve unsupported by recorder')
  record.append(np.asarray(answer).copy())
  return answer
 solver.solve=observed
 try:answer=solver.compliance(*args,**kwargs)
 finally:
  if had:solver.solve=previous
  else:del solver.solve
 if len(record)!=1:raise ValueError('Expected exactly one API displacement operand')
 return answer,record[0]
