"""Four real retained fixtures plus comparator-only corruption/missing tests."""
import copy,hashlib,io,json
from pathlib import Path
import numpy as np
import pytest
from .evidence import load_cases,exact_checkpoint,independent_certificate,EvidenceError
from .policy import compare,array_digest,global_comparison

@pytest.fixture(scope='module')
def cases():return load_cases()
@pytest.mark.parametrize('index',range(4))
def test_real_retained_fixture(index,cases):
 point,a,b,proof=cases[index];result=compare(a,b,proof)
 assert result['status']=='Accepted',result
 assert result['historical_comparator']==(point=='transition_before_journal')
 assert proof['reconstructed_bytes_match_original']
 assert result['independent_correctness']=='Accepted'

@pytest.mark.parametrize('index',range(4))
def test_real_committed_checkpoint_byte(index,cases):
 point,a,b,proof=cases[index];t=proof['committed_states']-1
 buf=io.BytesIO();np.savez_compressed(buf,**{k:b['arrays'][k][t] for k in b['metadata']['arrays'] if k!='update_change'})
 payload=bytearray(buf.getbuffer());expected=proof['hash_checks'][t]['original_sha256']
 assert hashlib.sha256(payload).hexdigest()==expected
 payload[len(payload)//2]^=1;assert hashlib.sha256(payload).hexdigest()!=expected
 failed={**proof,'status':'hash_mismatch'};assert compare(a,b,failed)['status']=='Rejected'

def test_real_committed_one_ulp_design_rejected(cases):
 point,a,b,proof=cases[0];changed={**b,'arrays':{**b['arrays'],'x':np.array(b['arrays']['x'])}}
 changed['arrays']['x'][0,0]=np.nextafter(changed['arrays']['x'][0,0],np.inf)
 result=compare(a,changed,proof);assert result['status']=='Rejected';assert 'Committed native' in result['reason']
 with pytest.raises(EvidenceError):exact_checkpoint(changed,point)

def test_real_localized_displacement_cannot_hide_in_global_norm(cases):
 point,a,b,proof=cases[0];changed={**b,'arrays':{**b['arrays'],'u':np.array(b['arrays']['u'])},'certificate':copy.deepcopy(b['certificate'])}
 changed['arrays']['u'][66,1879]+=1e-9
 # Alter only synthetic in-memory binding to isolate the numerical local guard;
 # immutable real certificate files/manifest are never modified or forged.
 changed['certificate']['native_digests']['u']=array_digest(changed['arrays']['u'])
 assert global_comparison(changed['arrays']['u'][66],a['arrays']['u'][66],1e-9)['passed']
 result=compare(a,changed,proof);assert result['status']=='Rejected';assert 'Localized/global displacement' in result['reason']

def test_missing_real_independent_operand_is_indeterminate(tmp_path,cases):
 _,a,b,_=cases[0];folder=Path(a['certificate']['evidence_folder'])
 for name in ('summary.json','states.jsonl','transitions.jsonl'):(tmp_path/name).write_bytes((folder/name).read_bytes())
 with pytest.raises(EvidenceError) as error:independent_certificate(a,tmp_path)
 assert error.value.status=='Indeterminate';assert 'displacement operands' in str(error.value)

def test_real_missing_checkpoint_binding_is_indeterminate(cases):
 _,a,b,proof=cases[0];broken={**proof};del broken['committed_native_digests']
 assert compare(a,b,broken)['status']=='Indeterminate'
