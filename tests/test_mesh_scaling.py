import numpy as np
import pytest
from run_m3_3c import specification,restrict,filter_audit
from topopt_utils.fea.trajectory_dataset import features,load_problem

@pytest.mark.parametrize('nx',[48,96,192,384])
@pytest.mark.parametrize('experiment',['A','B'])
def test_physical_configuration(nx,experiment):
    p,x,s=specification(nx,experiment);q,y=load_problem(s)
    assert p.ndof==2*(nx+1)*(nx//2+1)
    assert (p.width,p.height)==((48,24) if experiment=='A' else (nx,nx//2))
    np.testing.assert_array_equal(p.nodes[p.loads[0][0]],[p.width,p.height/2])
    np.testing.assert_array_equal(p.forces.reshape(-1,2).sum(axis=0),[1,0])
    assert len(p.fixed_dofs)==2*(p.nely+1)
    assert p.filter_radius*p.dx==2 and p.filter_radius*p.dy==2
    assert np.array_equal(x,y) and q.to_dict()==p.to_dict()

@pytest.mark.parametrize('nx',[48,96])
def test_filter_and_native_features(nx):
    p,x,_=specification(nx,'A');audit=filter_audit(p);assert audit['adjoint_absolute_error']<1e-10
    f=features(p,x,0,'no_fem_v1');assert f['element'].shape==(9,p.nely,p.nelx)
    assert f['node'].shape==(8,p.nely+1,p.nelx+1)
    np.testing.assert_array_equal(f['node'][0].ravel(),p.forces[::2])
    np.testing.assert_array_equal(f['element'][2].ravel(),p.centers[:,0])
    assert f['node'][2].sum()==p.nely+1

def test_conservative_transfer():
    fine=np.arange(32.).reshape(4,8);coarse=restrict(fine,(2,4))
    assert coarse.sum()*4==fine.sum()
    np.testing.assert_array_equal(restrict(np.ones((4,8)),(2,4)),1)
    with pytest.raises(ValueError):restrict(fine,(3,4))

def test_execution_scope():
    from run_m3_3c import benchmark
    with pytest.raises(ValueError):benchmark(192,'A',None)
