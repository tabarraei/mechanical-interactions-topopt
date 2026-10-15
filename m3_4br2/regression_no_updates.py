"""Run the full discovered suite with explicit skips at numerical mutation entry
points. This is NOT evidence of an unqualified complete regression pass.
Skipped fixture tests remain a separately recorded readiness prerequisite.
"""
import json,sys
from pathlib import Path
import pytest
class Scope:
 def pytest_configure(self,config):
  import topopt_utils.fea.dataset_storage as storage
  import topopt_utils.fea.consistent_trajectory as consistent
  import topopt_utils.fea.convergence_reliability as convergence
  import topopt_utils.fea.robust_oc as oc
  import topopt_utils.topopt_2d as topopt
  import generate_m1_interactions as m1
  def prohibited(*a,**k):pytest.skip('R2 scope: numerical OC/generation fixture requires explicit exception')
  for module,name in [(storage,'generate_sample'),(consistent,'generate_trajectory'),(convergence,'run_trajectory'),(oc,'update'),(topopt,'optimize'),(m1,'generate_m1_interactions_dataset')]:
   setattr(module,name,prohibited)
  self.rows=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.skipped:self.rows.append(dict(test=report.nodeid,phase=report.when,outcome=report.outcome,reason=str(report.longrepr) if report.failed or report.skipped else None))
 def pytest_sessionfinish(self,session,exitstatus):
  Path(sys.argv[1]).write_text(json.dumps(dict(exit_code=int(exitstatus),scope='all collected tests; prohibited numerical entry points skipped',complete_unqualified_regression=False,rows=self.rows),indent=2))
if __name__=='__main__':raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests','m3_4br2/test_audit.py'],plugins=[Scope()]))
