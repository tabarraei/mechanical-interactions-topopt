import pytest
from .worker import is_fault
@pytest.mark.parametrize('point,path,count',[('frame_before_journal','frames/00003.npz',None),('transition_before_journal','updates/00002.json',None),('committed_update2','journal.json',3),('metadata_before_rename','metadata.json',None)])
def test_fault_points(point,path,count):
 assert is_fault(point,path,count)
 assert not is_fault(point,'different.json',0)
def test_commit_count_exact():
 assert not is_fault('committed_update2','journal.json',2)
 assert not is_fault('committed_update2','journal.json',4)
def test_no_fault():assert not is_fault(None,'00003.npz',3)
def test_original_spec_canonical_and_file_bindings():
 from .common import job
 m,j,spec=job(0)
 assert j['problem_id']==spec['problem_id']
def test_result_path_serialization():
 from .worker import serializable_answer
 from pathlib import Path
 import json
 result=serializable_answer(dict(folder=Path('/tmp/example'),status='completed_new'))
 assert json.loads(json.dumps(result))['folder']=='/tmp/example'
