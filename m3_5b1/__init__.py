"""Planning only. Physics, production imports and neural training are forbidden."""
import sys,importlib.abc
class PlanningOnly(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'topopt_utils','pypardiso','torch','m3_4b','m3_4br','m3_4br2','m3_4br3','m3_4br4','m3_4br4approved'}:
   raise ImportError('M3.5B-1 planning forbids production/FEM/OC/training imports: '+fullname)
if not any(isinstance(x,PlanningOnly) for x in sys.meta_path):sys.meta_path.insert(0,PlanningOnly())
