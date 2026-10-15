"""Retained-array decomposition only; no solve or OC update."""
import json,sys
from pathlib import Path
import numpy as np
from m3_4br2.memory import load_trajectory
from m3_4br2.preflight import OLD,FAILED
from m3_4br2.gradient_certificate import certify
from topopt_utils.fea.q4 import __get_KE__,__simp_stiffness_sensitivity__
def run(out):
 out=Path(out);tr=load_trajectory(OLD/'native/samples'/FAILED);p=tr.problem
 with np.load(out/'independent_replay/state_operands_00408.npz') as z:
  rho=z['replay_rho'];u=z['replay_u'];api_g=z['api_g_rho'];ue=u[p.element_dofs];ke=__get_KE__(hx=p.dx,hy=p.dy);slope=__simp_stiffness_sensitivity__(rho,p.penal)
  from_u=-slope*np.einsum('ij,jk,ik->i',ue,ke,ue)
  ld=np.longdouble;uel=ue.astype(ld);kel=ke.astype(ld)
  ref=-slope.astype(ld)*np.einsum('ei,ij,ej->e',uel,kel,uel,dtype=ld)
  mag=abs(slope.astype(ld))*np.einsum('ei,ij,ej->e',abs(uel),abs(kel),abs(uel),dtype=ld)
  from fractions import Fraction as F
  from m3_4br2.audit import upper
  u64=F(1,2**53);uld=F(1,2**int(np.finfo(ld).nmant+1));gamma=lambda k,u:k*u/(1-k*u)
  coefficient=upper((gamma(66,u64)+gamma(67,uld))/(1-gamma(67,uld)))
  bound=np.nextafter(coefficient*np.nextafter(mag.astype(float),np.inf),np.inf);error=abs(api_g.astype(ld)-ref);bad=np.flatnonzero(error>bound)
  data=dict(state=408,failing_components=bad.tolist(),component_details=[dict(index=int(i),API_gradient=float(api_g[i]),first_displacement_gradient=float(from_u[i]),extended_reference=float(ref[i]),certificate_error=float(error[i]),allowance=float(bound[i])) for i in bad],saved_and_first_displacement_identical=bool(np.array_equal(z['saved_u'],u)),first_gradient_matches_same_formula=bool(np.array_equal(z['replay_g_rho'],from_u)),API_gradient_max_difference_from_first_displacement=float(np.max(abs(api_g-from_u))),API_displacement_retained=False,mismatched_certificate=certify(p,rho,u,api_g,z['api_g_x']),interpretation='Certificate uses displacement from evaluate_state first solve; API compliance obtains another solve internally, whose actual displacement was not retained. This proves an operand association defect in replay instrumentation; it does not prove the second-solve gradient incorrect. Unchanged v2 rejects cross-pair evidence at one component.')
 (out/'transition408_diagnosis.json').write_text(json.dumps(data,indent=2))
if __name__=='__main__':run(sys.argv[1])
