"""In-memory parser fixtures are not issued receipts or proof of authority."""
import copy,sys
from datetime import datetime,timezone,timedelta
import pytest
from m3_5b2d.common import *
from m3_5b2d.authorization import approval_record,review_book
from m3_5b2d.policy import scope,decision_state

@pytest.fixture(scope='module')
def binding():return target_binding()
@pytest.fixture
def book(binding):
 now=datetime.now(timezone.utc)
 return {'schema_version':'m3.5b2d.reviewed_approvals.v1','records':[dict(kind='policy',sha256=digest(ROOT/'m3_5b2d/policy.py'),binding_sha256=canonical(binding),decision='D2',identity='SYNTHETIC_PARSER_FIXTURE_NOT_ISSUER',approved_at=(now-timedelta(seconds=1)).isoformat(),expires_at=(now+timedelta(minutes=5)).isoformat(),reference='SYNTHETIC_ONLY_NEVER_PUBLISHED')]}

def test_actual_policy_absent_and_no_authority_created(binding):
 assert not AUTHORITY.exists()
 with pytest.raises(ValueError,match='No protected human'):approval_record('policy',digest(ROOT/'m3_5b2d/policy.py'),binding)

@pytest.mark.parametrize('attack',['missing','schema','empty_issuer','wrong_hash','expired','wrong_decision','duplicate','unknown_field'])
def test_parser_rejects_invalid_receipt(book,binding,attack):
 r=book['records'][0]
 if attack=='missing':book['records']=[]
 elif attack=='schema':book['schema_version']='wrong'
 elif attack=='empty_issuer':r['identity']=''
 elif attack=='wrong_hash':r['sha256']='0'*64
 elif attack=='expired':r['expires_at']='2000-01-01T00:00:00Z'
 elif attack=='wrong_decision':r['decision']='D1'
 elif attack=='duplicate':book['records'].append(copy.deepcopy(r))
 else:r['forged_signature']='not-a-signature'
 with pytest.raises(ValueError):review_book(book,kind='policy',object_hash=digest(ROOT/'m3_5b2d/policy.py'),binding_hash=canonical(binding))

@pytest.mark.parametrize('key,value',[('historical_id','other'),('selected_id','other'),('manifest_sha256','0'*64),('equivalence_sha256','0'*64),('input_root','/tmp/other-production')])
def test_policy_scope_and_binding_mismatch(book,binding,key,value):
 other=copy.deepcopy(binding);other[key]=value
 with pytest.raises(ValueError):scope(other)
 with pytest.raises(ValueError):review_book(book,kind='policy',object_hash=digest(ROOT/'m3_5b2d/policy.py'),binding_hash=canonical(other))

def test_policy_row_cannot_authorize_other_approval_kinds(book,binding):
 for kind in ('protocolA_execution','alias','family_crosswalk','numerical_acceptance'):
  with pytest.raises(ValueError):review_book(book,kind=kind,object_hash=digest(ROOT/'m3_5b2d/policy.py'),binding_hash=canonical(binding))
 assert not decision_state('D2',policy_approved=True)['release_eligible']

def test_identity_authentication_limit_is_explicit(book,binding):
 # Important audit observation: a trusted-book identity is nonempty metadata,
 # not an authenticated identity-provider lookup. Do not claim "wrong named
 # issuer rejects" from a validator that has no issuer allowlist/signature.
 book['records'][0]['identity']='UNVERIFIED_NONEMPTY_IDENTITY_PARSER_FIXTURE'
 r=review_book(book,kind='policy',object_hash=digest(ROOT/'m3_5b2d/policy.py'),binding_hash=canonical(binding))
 assert r['identity']=='UNVERIFIED_NONEMPTY_IDENTITY_PARSER_FIXTURE'
 assert not AUTHORITY.exists()

def test_missing_execution_rejects_before_solver_import(tmp_path):
 from m3_5b2d.runner import dispatch
 with pytest.raises(ValueError):dispatch(None,tmp_path/'not_created')
 assert not (tmp_path/'not_created').exists()
 assert not any(n.startswith(('topopt_utils','numpy','torch','pypardiso','m3_4br3.replay')) for n in sys.modules)
