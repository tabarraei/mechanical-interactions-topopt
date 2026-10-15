"""Versioned, canonical mechanical specification for M3.1.

Supported material/OC constants are deliberately restricted to the unchanged core.
No geometry/load definition is inferred from an element count or an old trajectory.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy import ndimage
from .solver import _build_connectivity
from ..topopt_2d import __create_filter_kernel_nd__

SCHEMA_VERSION = 'm3.1.problem.v1'
ORDERING = 'elements C(nely,nelx); nodes row-major; DOFs x,y; corners a,b,d,c'


def array_digest(array):
    a=np.ascontiguousarray(array)
    h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class MechanicalProblem:
    nelx: int
    nely: int
    width: float
    height: float
    fixed_dofs: tuple
    loads: tuple  # tuples (node_id, fx, fy), the single authoritative force definition
    volume_fraction: float = .5
    penal: float = 3.
    filter_radius: float = 2.
    maxiter: int = 100
    tolerance: float = .001
    seed: int = 0  # deterministic family; recorded, not used to alter mechanics

    def __post_init__(self):
        if not all(isinstance(v,(int,np.integer)) and v>0 for v in (self.nelx,self.nely,self.maxiter)):
            raise ValueError('Positive integer element counts and update limit required')
        if not all(np.isfinite(v) for v in (self.width,self.height,self.volume_fraction,self.penal,self.filter_radius,self.tolerance)):
            raise ValueError('Nonfinite problem settings')
        if min(self.width,self.height,self.penal,self.filter_radius,self.tolerance)<=0 or not .001<self.volume_fraction<1:
            raise ValueError('Invalid domain, material, filter, tolerance or volume')
        fixed=np.asarray(self.fixed_dofs)
        if fixed.ndim!=1 or not np.issubdtype(fixed.dtype,np.integer) or np.any(fixed<0) or np.any(fixed>=self.ndof) or len(set(fixed))!=len(fixed):
            raise ValueError('Invalid constrained DOFs')
        if tuple(sorted(self.fixed_dofs))!=self.fixed_dofs:raise ValueError('DOFs must be sorted')
        if any(not isinstance(node,(int,np.integer)) or not 0<=node<self.ndof//2 or not np.isfinite([fx,fy]).all() for node,fx,fy in self.loads):
            raise ValueError('Invalid nodal load')
        if np.any(self.forces[fixed]!=0) or np.linalg.norm(self.forces)==0:
            raise ValueError('Require nonzero free loads; forces on constrained DOFs are rejected')
        rigid=np.zeros((self.ndof,3));xy=self.nodes
        rigid[0::2,0]=1;rigid[1::2,1]=1;rigid[0::2,2]=-xy[:,1];rigid[1::2,2]=xy[:,0]
        if np.linalg.matrix_rank(rigid[fixed])!=3:raise ValueError('Supports do not remove rigid-body modes')

    @property
    def n(self):return self.nelx*self.nely
    @property
    def ndof(self):return 2*(self.nelx+1)*(self.nely+1)
    @property
    def dx(self):return self.width/self.nelx
    @property
    def dy(self):return self.height/self.nely
    @property
    def nodes(self):
        x,y=np.meshgrid(np.arange(self.nelx+1)*self.dx,np.arange(self.nely+1)*self.dy,indexing='xy')
        return np.column_stack((x.ravel(),y.ravel()))
    @property
    def element_dofs(self):return _build_connectivity(self.nelx,self.nely).element_dofs
    @property
    def connectivity(self):return self.element_dofs[:,0::2]//2
    @property
    def centers(self):return self.nodes[self.connectivity].mean(axis=1)
    @property
    def forces(self):
        f=np.zeros(self.ndof)
        for node,fx,fy in self.loads:f[2*node]+=fx;f[2*node+1]+=fy
        return f
    @property
    def solver_args(self):return (self.forces,np.asarray(self.fixed_dofs,dtype=np.int64),self.nelx,self.nely,self.dx,self.dy)

    def to_dict(self):
        loads=[dict(node_id=int(node),coordinates=self.nodes[node].tolist(),fx=float(fx),fy=float(fy)) for node,fx,fy in self.loads]
        return dict(schema_version=SCHEMA_VERSION,nelx=self.nelx,nely=self.nely,width=self.width,height=self.height,dx=self.dx,dy=self.dy,
            node_coordinates=self.nodes.tolist(),connectivity=self.connectivity.tolist(),element_dofs=self.element_dofs.tolist(),element_centers=self.centers.tolist(),ordering=ORDERING,
            coordinate_frame='increasing row-y Cartesian components; plots origin=lower',
            material=dict(E0=1.,nu=.3,thickness=1.,model='static plane stress Q4'),
            simp=dict(penal=self.penal,emin=.001,formula='0.001+0.999*rho**penal'),
            filter=dict(radius=self.filter_radius,radius_units='element-count',kind='boundary-renormalized linear density convolution'),projection=None,
            fixed_dofs=list(self.fixed_dofs),loads=loads,force_vector=self.forces.tolist(),volume_fraction=self.volume_fraction,
            optimizer=dict(name='OC',move=.1,x_min=.001,x_max=1.,lambda_lower=0.,lambda_upper=1e5,bisection_width=1e-4,maxiter=self.maxiter,tolerance=self.tolerance,volume_variable='x'),
            seed=self.seed,units=dict(length='consistent arbitrary domain units',force='consistent arbitrary force units',density='dimensionless',modulus='relative E0=1; implicit unit thickness'))

    @property
    def problem_id(self):
        encoded=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()
        return 'm3_1_'+hashlib.sha256(encoded).hexdigest()[:24]

    @classmethod
    def from_dict(cls,data):
        if data.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Unknown mechanical schema')
        opt=data['optimizer']
        obj=cls(data['nelx'],data['nely'],data['width'],data['height'],tuple(data['fixed_dofs']),
            tuple((l['node_id'],l['fx'],l['fy']) for l in data['loads']),data['volume_fraction'],data['simp']['penal'],data['filter']['radius'],opt['maxiter'],opt['tolerance'],data['seed'])
        expected=obj.to_dict()
        # Numerical checks of redundant saved arrays; never trust hashes alone.
        for name in ('node_coordinates','connectivity','element_dofs','element_centers','force_vector'):
            actual=np.asarray(data[name]);truth=np.asarray(expected[name])
            if actual.shape!=truth.shape or not np.array_equal(actual,truth):
                raise ValueError(f'Inconsistent saved {name}')
        if data!=expected:raise ValueError('Unsupported or inconsistent saved settings, ordering or material')
        return obj


def cantilever_problem(nelx,nely,**settings):
    fixed=tuple(2*(y*(nelx+1))+axis for y in range(nely+1) for axis in (0,1))
    node=(nely//2)*(nelx+1)+nelx
    return MechanicalProblem(nelx,nely,float(nelx),float(nely),fixed,((node,1.,0.),),**settings)


class DensityMap:
    """Exactly the existing OC boundary-renormalized filter, including transpose."""
    def __init__(self,problem):
        self.shape=(problem.nely,problem.nelx)
        self.kernel=__create_filter_kernel_nd__(problem.filter_radius,problem.filter_radius)
        self.weights=ndimage.convolve(np.ones(self.shape),self.kernel,mode='constant',cval=0.)
    def forward(self,x):
        return (ndimage.convolve(np.asarray(x).reshape(self.shape),self.kernel,mode='constant',cval=0.)/self.weights).ravel()
    def transpose(self,g):
        return ndimage.convolve(np.asarray(g).reshape(self.shape)/self.weights,self.kernel,mode='constant',cval=0.).ravel()
