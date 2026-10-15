"""Make accidental numerical-work imports fail before import side effects."""
import sys,importlib.abc
BLOCKED={'topopt_utils','pypardiso','m3_4b','m3_4br','m3_4br2','m3_4br3','m3_4br4','m3_4br4approved'}
class NoPhysics(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in BLOCKED:raise ImportError('M3.5A prohibits FEM/OC/production import: '+fullname)
def install():
 if any(name.split('.')[0] in BLOCKED for name in sys.modules):raise RuntimeError('Start M3.5A in a clean process without physics imports')
 if not any(isinstance(f,NoPhysics) for f in sys.meta_path):sys.meta_path.insert(0,NoPhysics())
