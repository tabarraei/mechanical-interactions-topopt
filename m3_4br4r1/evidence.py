"""Read-only retained artifacts and reconstruction hashes. No physics imports."""
import hashlib,io,json,os
from pathlib import Path
import numpy as np
from .policy import EvidenceError,array_digest,canonical
ROOT=Path(__file__).resolve().parents[1]
R4=ROOT/'m3_4br4/results/20261014T174028Z'
R3=ROOT/'m3_4br3/results'
ORIGINAL=ROOT/'m3_4b/generations/20261014T013017Z'
PID='m3_3b_93ad168499a05bed721257ad'
POINTS=('frame_before_journal','transition_before_journal','committed_update2','metadata_before_rename')
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check_record(row):
 if row.get('passed') is not True:raise EvidenceError('Rejected','Retained independent primary comparison failed')
 if 'absolute_error' in row:
  if not np.isfinite([row['absolute_error'],row['absolute_tolerance']]).all() or row['absolute_error']>row['absolute_tolerance']:raise EvidenceError('Rejected','Inconsistent independent comparison record')

def load_native(folder):
 folder=Path(folder);spec=read(folder/'problem.json');meta=read(folder/'metadata.json');a={}
 for key,e in meta['arrays'].items():
  p=folder/e['file']
  if not p.exists():raise EvidenceError('Indeterminate','Missing native array '+str(p))
  if digest(p)!=e['sha256']:raise EvidenceError('Rejected','Native output hash '+str(p))
  z=np.load(p,mmap_mode='r',allow_pickle=False)
  if list(z.shape)!=e['shape'] or str(z.dtype)!=e['dtype']:raise EvidenceError('Rejected','Native metadata shape/dtype')
  a[key]=z
 for file,key in [('problem.json','problem_sha256'),('transitions.json','transitions_sha256')]:
  if digest(folder/file)!=meta[key]:raise EvidenceError('Rejected','Native '+file+' hash')
 return dict(folder=str(folder),spec=spec,metadata=meta,arrays=a,transitions=read(folder/'transitions.json'))

def independent_certificate(side,folder):
 folder=Path(folder);summary=read(folder/'summary.json');a=side['arrays'];T=side['metadata']['state_count'];identity=summary['identity']
 if not summary.get('passed') or summary['states']!=T or summary['transitions']!=T-1:raise EvidenceError('Rejected','Incomplete independent summary')
 if identity['native_source']!=side['metadata']['source_id'] or identity['problem_id']!=side['spec']['problem_id']:raise EvidenceError('Rejected','Replay source/problem identity')
 if identity['array_hashes']!={k:e['sha256'] for k,e in side['metadata']['arrays'].items()}:raise EvidenceError('Rejected','Independent native hash identity')
 for name,h in identity['validation_sources'].items():
  if digest(ROOT/'m3_4br3'/name)!=h:raise EvidenceError('Rejected','Independent replay source fingerprint '+name)
 states=[json.loads(line) for line in (folder/'states.jsonl').read_text().splitlines()];trans=[json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
 if [r['state'] for r in states]!=list(range(T)) or [r['transition'] for r in trans]!=list(range(T-1)):raise EvidenceError('Rejected','Independent record state/transition ordering')
 expected_primary={'rho','u','compliance','g_rho','g_x','k_diagonal','k_probe_actions','api_u','api_C','api_gradient','transpose','COO'}
 for t,row in enumerate(states):
  if not row.get('passed') or not expected_primary.issubset(row['primary']):raise EvidenceError('Rejected','Incomplete independent mechanics')
  for r in row['primary'].values():check_record(r)
  op=folder/f'state_operands_{t:05d}.npz'
  if not op.exists():raise EvidenceError('Indeterminate','Missing independent displacement operands')
  with np.load(op,allow_pickle=False) as z:
   for key in ('x','rho','u','compliance','g_rho','g_x'):
    if not np.array_equal(z['saved_'+key],a[key][t]):raise EvidenceError('Rejected','Independent saved operand differs '+key)
   for key in ('api_u','api_g_rho','api_g_x'):
    if key not in z.files or not np.isfinite(z[key]).all():raise EvidenceError('Indeterminate','Missing actual API operand '+key)
 for t,row in enumerate(trans):
  if not row.get('passed') or row['policy']!='m3.4b.transition_change_audit.v3' or not row['change_audit']['passed']:raise EvidenceError('Rejected','Independent OC/v3 audit')
  if not all(row['strict_feasibility'].values()):raise EvidenceError('Rejected','Independent strict feasibility')
  for key,r in row['primary'].items():
   if key=='incoming' and r.get('policy')=='m3.4b.transition_change_audit.v3':
    if not r['passed'] or not row['change_audit']['passed']:raise EvidenceError('Rejected','Adapted incoming audit failed')
   else:check_record(r)
  if not row['checks']['selected_multiplier']['passed']:raise EvidenceError('Rejected','Independent multiplier')
  for d in row['diagnostics'].values():
   if not d['new_passed']:raise EvidenceError('Rejected','Independent OC diagnostic')
  op=folder/row['operand_file']
  if digest(op)!=row['operand_sha256']:raise EvidenceError('Rejected','Independent action operand hash')
  with np.load(op,allow_pickle=False) as z:
   if not np.array_equal(z['saved_action'],a['x'][t+1]) or not np.array_equal(z['saved_x'],a['x'][t]):raise EvidenceError('Rejected','Independent action/source bookkeeping')
 return dict(status='verified_complete',evidence_folder=str(folder),states=T,transitions=T-1,problem_id=side['spec']['problem_id'],source_id=side['metadata']['source_id'],configuration_digest=canonical(side['spec']),policy=identity['policy'],source_fingerprints_verified=True,retained_records_verified=True,native_digests={k:array_digest(v) for k,v in a.items()},environment=identity['environment'],summary_sha256=digest(folder/'summary.json'),new_FEM_calls=0,new_OC_calls=0)

def exact_checkpoint(side,point):
 before=read(R4/'work_counts'/('fault_'+point)/'fault_evidence.json');raw=R4/'work_counts'/('fault_'+point)/'interrupted_journal.json';after=read(Path(side['folder'])/'journal.json');original=before['journal'];s=side['spec'];source=side['metadata']['source_id']
 if digest(raw)!=before['journal_sha256'] or read(raw)!=original:raise EvidenceError('Rejected','Interrupted journal bytes')
 expected=dict(problem_id=s['problem_id'],spec_sha=canonical(s),source_id=source)
 if original['binding']!=expected or after['binding']!=expected or before['source_id']!=source:raise EvidenceError('Rejected','Checkpoint exact identity')
 if after['states'][:len(original['states'])]!=original['states'] or after['updates'][:len(original['updates'])]!=original['updates']:raise EvidenceError('Rejected','Committed journal prefix/order changed')
 a=side['arrays'];fields=[k for k in side['metadata']['arrays'] if k!='update_change'];checks=[]
 for t,e in enumerate(original['states']):
  if e['file']!=f'frames/{t:05d}.npz' or before['committed_hashes'].get(e['file'])!=e['sha256']:raise EvidenceError('Rejected','Committed frame entry/order')
  # Deterministic serialization only: reconstruct compressed bytes from retained
  # final fields, then require exact equality with the pre-interruption SHA256.
  buf=io.BytesIO();np.savez_compressed(buf,**{k:a[k][t] for k in fields});h=hashlib.sha256(buf.getbuffer()).hexdigest()
  if h!=e['sha256']:raise EvidenceError('Rejected','Committed frame reconstruction hash '+str(t))
  checks.append(dict(file=e['file'],original_sha256=e['sha256'],reconstructed_sha256=h,passed=True))
 for t,e in enumerate(original['updates']):
  if e['file']!=f'updates/{t:05d}.json' or before['committed_hashes'].get(e['file'])!=e['sha256']:raise EvidenceError('Rejected','Committed transition entry/order')
  encoded=(json.dumps(side['transitions'][t],indent=2,allow_nan=False)+'\n').encode();h=hashlib.sha256(encoded).hexdigest()
  if h!=e['sha256']:raise EvidenceError('Rejected','Committed transition bytes '+str(t))
  checks.append(dict(file=e['file'],original_sha256=e['sha256'],reconstructed_sha256=h,passed=True))
 # Validate full native frame/update order, including the post-prefix continuation.
 if len(after['states'])!=len(a['x']) or len(after['updates'])!=len(side['transitions']):raise EvidenceError('Rejected','Final journal count')
 for t,e in enumerate(after['states']):
  if e['file']!=f'frames/{t:05d}.npz':raise EvidenceError('Rejected','Final state order')
 for t,e in enumerate(after['updates']):
  if e['file']!=f'updates/{t:05d}.json':raise EvidenceError('Rejected','Final transition order')
 return dict(status='verified_exact',source_identity=True,configuration_identity=True,source_id=source,configuration_digest=canonical(s),journal_sha256=before['journal_sha256'],committed_states=len(original['states']),committed_transitions=len(original['updates']),committed_native_digests={k:array_digest(v[:len(original['updates']) if k=='update_change' else len(original['states'])]) for k,v in a.items()},hash_checks=checks,atomic_fault_point=point,original_frame_files_retained=False,reconstructed_bytes_match_original=True,original_frames_deleted_by_documented_finalization=True)

def verify_baseline(out):
 out=Path(out);records={};mismatches=[]
 for label,base in [('R3',R3),('R4',R4)]:
  manifest=read(base/'artifact_manifest.json')
  for name,h in manifest.items():
   path=base/name
   if not path.exists() or digest(path)!=h:mismatches.append(str(path))
  records[label]=dict(artifact_files=len(manifest),artifact_manifest_sha256=digest(base/'artifact_manifest.json'))
 if mismatches:raise EvidenceError('Rejected','Historical artifact integrity mismatch '+str(mismatches[:10]))
 for group in ('m3_4br3','m3_4br2'):
  for path,e in read(R4/'executing_source_manifest.json').items():
   if path.startswith(group+'/') and digest(ROOT/path)!=e['sha256']:raise EvidenceError('Rejected','Frozen audit dependency '+path)
 original=read(ORIGINAL/'source_manifest.json')
 for name,e in original.items():
  if digest(ROOT/name)!=e['sha256'] or digest(ORIGINAL/e['archive'])!=e['sha256']:raise EvidenceError('Rejected','Original mechanical source/archive '+name)
 if read(R4/'final_readiness.json')['saved_states']!=3368:raise EvidenceError('Rejected','R4 baseline count')
 if read(R3/'complete_validation.json')['FEM_calls']!=1003:raise EvidenceError('Rejected','R3 baseline accounting')
 report=dict(passed=True,old_source_archive_count=len(original),artifacts=records,original_ledger_sha256=digest(ROOT/'m3_4b/authorization.json'),historical_gate_unchanged=True,new_FEM=0,new_OC=0)
 (out/'baseline_integrity.json').write_text(json.dumps(report,indent=2));return report

def load_cases():
 control=load_native(R4/'recovery/control/samples'/PID);control['certificate']=independent_certificate(control,R4/'validation/recovery_control_replay');cases=[]
 for point in POINTS:
  side=load_native(R4/'recovery'/point/'samples'/PID);side['certificate']=independent_certificate(side,R4/'validation'/('replay_'+point));proof=exact_checkpoint(side,point)
  if side['certificate']['environment']!=control['certificate']['environment']:raise EvidenceError('Rejected','Replay environment identities differ')
  cases.append((point,control,side,proof))
 return cases
