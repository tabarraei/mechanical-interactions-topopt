"""Supplement historical global checks with conditioned local algebra checks.
No solve, optimization, or change to any sensitivity implementation.
"""
from fractions import Fraction as F
import numpy as np
from .audit import upper

def certify(problem,rho,u,g_rho,g_x):
 from topopt_utils.fea.q4 import __get_KE__,__simp_stiffness_sensitivity__
 from topopt_utils.fea.mechanical_problem import DensityMap,array_digest
 if np.finfo(np.longdouble).eps>=np.finfo(float).eps:raise ValueError('Extended reference unavailable')
 if any(np.asarray(a).dtype!=np.float64 or not np.isfinite(a).all() for a in (rho,u,g_rho,g_x)):raise ValueError('Certificate dtype/finite')
 ld=np.longdouble;ue=np.asarray(u[problem.element_dofs],dtype=ld);ke=np.asarray(__get_KE__(hx=problem.dx,hy=problem.dy),dtype=ld)
 # Treat SIMP slope's binary64 output as common input, not a changed derivative.
 slope=__simp_stiffness_sensitivity__(rho,problem.penal);s=np.asarray(slope,dtype=ld)
 ref=-s*np.einsum('ei,ij,ej->e',ue,ke,ue,dtype=ld)
 magnitude=abs(s)*np.einsum('ei,ij,ej->e',abs(ue),abs(ke),abs(ue),dtype=ld)
 # 64 quadratic terms, two products + at most63 additions + final slope:
 # gamma66(binary64). Extended reference/magnitude uses gamma67(extended).
 u64=F(1,2**53);uld=F(1,2**int(np.finfo(ld).nmant+1));gamma=lambda k,u:k*u/(1-k*u)
 coefficient=upper((gamma(66,u64)+gamma(67,uld))/(1-gamma(67,uld)))
 mag_upper=np.nextafter(np.asarray(magnitude,dtype=float),np.inf)
 allowance=np.nextafter(coefficient*mag_upper,np.inf)
 errors=np.nextafter(abs(np.asarray(g_rho,dtype=ld)-ref),ld(np.inf))
 passed=bool(np.all(errors<=allowance));mapped=DensityMap(problem).transpose(g_rho);chain=bool(np.array_equal(mapped,g_x))
 return dict(passed=passed and chain,local_quadratic_passed=passed,exact_recorded_filter_transpose=chain,maximum_error=float(np.max(errors)),maximum_allowance=float(np.max(allowance)),failing_components=int(np.sum(errors>allowance)),method='gamma66 binary64 local quadratic/slope; gamma67 extended reference and positive magnitude; exact F.T chain',input_hashes={k:array_digest(v) for k,v in dict(rho=rho,u=u,g_rho=g_rho,g_x=g_x).items()})
