from __future__ import annotations
import json,os,pathlib,shutil,subprocess,tempfile
from omega_common import *
TRACK_FILES={
'T1':['T1_MOUSE_ELECTRICAL__dandi_inventory.json','T1_MOUSE_ELECTRICAL','000458'],
'T2':['T2_HUMAN_IEEG__openneuro_inventory.json','T2_HUMAN_IEEG','ds004696','on004696'],
'T3':['T3_HUMAN_TMSEEG__openneuro_inventory.json','T3_HUMAN_TMSEEG','ds005779','on005779'],
'T4':['T4_MOUSE_OPTO_PFC__dandi_inventory.json','T4_MOUSE_OPTO_PFC','001260'],
'T5':['T5_MOUSE_OPTO_ILLUSION__dandi_inventory.json','T5_MOUSE_OPTO_ILLUSION','000248'],
'T6A':['T6A_RAT_TFUS_ANESTHETIZED__dandi_inventory.json','T6A_RAT_TFUS_ANESTHETIZED','000637'],
'T6B':['T6B_RAT_TFUS_AWAKE__dandi_inventory.json','T6B_RAT_TFUS_AWAKE','000945'],
'T7':['T7_MACAQUE_BUMP__dandi_inventory.json','T7_MACAQUE_BUMP','000127'],
'T8':['T8_WORM_CAUSAL_ATLAS__dandi_inventory.json','T8_WORM_CAUSAL_ATLAS','001075']}
SEARCH_ROOTS=[pathlib.Path('/home/deploy/rari_benchmark_autopilot'),pathlib.Path('/home/deploy/rari_codex')]
EXACT={'REALBIO_DATASET_CONTRACT_V5.json','V5_3X20H_ARCHITECTURE_DRAFT.json','PARALLEL9_DATA_CONTRACT_RESOLVED.json','PARALLEL9_20H_ARCHITECTURE_DRAFT.json','STEP2D_SUMMARY.json','STEP2D_PARALLEL9_SHA256_MANIFEST.txt','PREFLIGHT_SHA256_MANIFEST.txt','AUTHORITATIVE_RESULTS.json','P1_P5_TRACK_TABLE.csv','STATISTICAL_CONTRACT_FROZEN.json','FROZEN_PROTOCOL.json','WX_DEFINITIONS_FROZEN.json','scorer.py','COVERAGE_CONSTRUCT_AUDIT.json','META_ANALYSIS_TABLE.csv','postrun_frozen_scoring_supervisor.sh','SUBJECT_SPLITS_V2.json','TRACK_CLAIM_MATRIX_V2.json'}

def _discover():
 hits=[];needles={x for v in TRACK_FILES.values() for x in v}
 for root in SEARCH_ROOTS:
  if not root.exists():continue
  try:p=subprocess.run(['find',str(root),'-xdev','-type','f','-size','-128M','-print'],capture_output=True,text=True,timeout=120)
  except subprocess.TimeoutExpired:continue
  for ss in p.stdout.splitlines():
   q=pathlib.Path(ss);name=q.name
   if name in EXACT or any(n in name or n in ss for n in needles) or 'candidate_rows.json'==name:
    try:hits.append({'path':ss,'name':name,'size':q.stat().st_size,'sha256':sha(q)})
    except OSError:pass
 return hits

def _find_authoritative(hits):
 out=[]
 for h in hits:
  if h['name']=='AUTHORITATIVE_RESULTS.json':
   try:
    d=read(h['path']);out.append({'path':h['path'],'sha256':h['sha256'],'top_keys':sorted(d)[:80] if isinstance(d,dict) else [],'json':d})
   except Exception as e:out.append({'path':h['path'],'sha256':h['sha256'],'read_error':str(e)})
 return out

def _small_snapshot(src,dst,cap):
 allowed={'.py','.json','.csv','.tsv','.txt','.md','.npy','.npz'};used=0;copied=[]
 for p in sorted(src.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.suffix.lower() not in allowed:continue
  try:n=p.stat().st_size
  except OSError:continue
  if n>16*1024*1024 or used+n>cap:continue
  rel=p.relative_to(src);q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);used+=n;copied.append({'path':str(rel),'size':n,'sha256':sha(q)})
 return used,copied

def _replay_track(t,hits,total_budget_state):
 track=t['id'];name=t['name'];tokens=TRACK_FILES[track];related=[h for h in hits if any(x in h['name'] or x in h['path'] for x in tokens)]
 scorer=[h for h in related if h['name']=='scorer.py' and ('/staging/'+name+'/' in h['path'] or '/'+name+'/' in h['path'])]
 out={'status':'NOT_AVAILABLE','source_related_files':len(related),'scorer_candidates':[x['path'] for x in scorer]}
 if not scorer:return out
 src=pathlib.Path(sorted(scorer,key=lambda x:len(x['path']))[0]['path']).parent
 per=int(protocol()['execution']['realbio_replay_snapshot_cap_per_track_bytes']);remain=max(0,int(protocol()['execution']['realbio_replay_snapshot_cap_total_bytes'])-total_budget_state['used']);cap=min(per,remain)
 if cap<=0:return {'status':'RESOURCE_CAP_NOT_EVALUABLE','reason':'realbio snapshot total cap exhausted','source_dir':str(src)}
 dst=ROOT/'private'/'realbio_replay'/track
 if dst.exists():shutil.rmtree(dst)
 dst.mkdir(parents=True,exist_ok=True)
 used,copied=_small_snapshot(src,dst,cap);total_budget_state['used']+=used
 before=file_manifest(dst)
 scorer_path=dst/pathlib.Path(scorer[0]['path']).name
 if not scorer_path.is_file():return {'status':'ATTEMPT_FAILED','reason':'scorer not present after bounded snapshot','source_dir':str(src),'snapshot_bytes':used,'copied':copied}
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 try:r=cmd([str(PYTHON),'-B',str(scorer_path)],timeout=600,cwd=dst,env=env);timeout=False
 except subprocess.TimeoutExpired as e:
  r=None;timeout=True
 after=file_manifest(dst)
 created={k:v for k,v in after.items() if k not in before or before[k]!=v}
 rec={'source_dir':str(src),'source_scorer_sha256':scorer[0]['sha256'],'snapshot_bytes':used,'snapshot_file_count':len(copied),'snapshot_manifest_sha256':digest(before),'changed_or_created_files':created,'timeout':timeout}
 if timeout:rec['status']='ATTEMPT_FAILED_TIMEOUT';return rec
 rec.update({'returncode':r.returncode,'stdout_tail':r.stdout[-6000:],'stderr_tail':r.stderr[-6000:],'status':'FROZEN_SCORER_REPLAY_PASS' if r.returncode==0 else 'REPLAY_ATTEMPT_FAILED'})
 return rec

def _candidate_evidence(track,hits):
 tokens=TRACK_FILES[track];cand=[h for h in hits if h['name']=='candidate_rows.json']
 matched=[];row_count=0;subject_tokens=set()
 for h in cand:
  try:
   txt=pathlib.Path(h['path']).read_text(errors='ignore')
  except Exception:continue
  if not any(tok in txt or tok in h['path'] for tok in tokens):continue
  try:d=json.loads(txt)
  except Exception:continue
  matched.append({'path':h['path'],'sha256':h['sha256'],'size':h['size']})
  stack=[d]
  while stack:
   x=stack.pop()
   if isinstance(x,list):stack.extend(x);continue
   if isinstance(x,dict):
    if any(k in x for k in ('subject','subject_id','animal','animal_id','unit','unit_id','worm','participant_id')):
     row_count+=1
     for k in ('subject','subject_id','animal','animal_id','unit','unit_id','worm','participant_id'):
      if k in x:subject_tokens.add(str(x[k]))
    stack.extend(x.values())
 return {'candidate_files':matched[:500],'matched_candidate_files':len(matched),'unit_like_rows':row_count,'distinct_unit_tokens':len(subject_tokens)}

def _track_translation(t,old,replay,candidate,related):
 n=t['id'];x=old['tracks'][n];out={'track':n,'name':t['name'],'dataset':t['dataset'],'version':t['version'],'expected_units':t['expected_units'],'legacy_extracted':x.get('extracted'),'legacy_decision':x.get('legacy_decision'),'evidence_mode':'RETAINED_V5_AUTHORITATIVE_AND_UNIT/STAGING_EVIDENCE_REPLAY_WHERE_AVAILABLE','new_prospective_biological_confirmation':False,'replay_attempt':replay,'candidate_evidence':candidate,'server_cache_provenance':related[:1000],'endpoints':{}}
 out['endpoints']['RB0_FROZEN_V5_REPLAY']={'status':replay['status']}
 if 'P1' in x:
  v=x['P1'];out['endpoints']['RB1_PERTURBATION_INFORMATION_GAIN']={'legacy_effect':v,'status':'HISTORICAL_DIRECTION_POSITIVE_NOT_NEW_CONFIRMATION' if v>0 else 'HISTORICAL_DIRECTION_NEGATIVE_NOT_NEW_CONFIRMATION' if v<0 else 'HISTORICAL_NULL_NOT_NEW_CONFIRMATION'}
 else:out['endpoints']['RB1_PERTURBATION_INFORMATION_GAIN']={'status':'NOT_EVALUABLE_FROM_AVAILABLE_REALBIO_EVIDENCE'}
 out['endpoints']['RB2_EXCITATION_ERROR_RELATION']={'status':'NOT_EVALUABLE_WITHOUT_EXPLICIT_WITHIN_UNIT_EXCITATION_VECTORS','retained_unit_like_rows':candidate['unit_like_rows']}
 if all(k in x for k in ('P4_raw','P4_wrong','P4_shuffle')):
  vals={k:x[k] for k in ('P4_raw','P4_wrong','P4_shuffle')};out['endpoints']['RB3_COMMON_OBSERVABLE_AGREEMENT']={'legacy_controls':vals,'status':'HISTORICAL_SUPPORT_PATTERN_NOT_NEW_CONFIRMATION' if min(vals.values())>0 else 'HISTORICAL_CHALLENGE_PATTERN'};out['endpoints']['RB4_NULL_SENSITIVITY']={'legacy_wrong_pair':x['P4_wrong'],'legacy_shuffle':x['P4_shuffle'],'status':'HISTORICAL_SUPPORT_PATTERN_NOT_NEW_CONFIRMATION' if x['P4_wrong']>0 and x['P4_shuffle']>0 else 'HISTORICAL_CHALLENGE_PATTERN'}
 else:
  out['endpoints']['RB3_COMMON_OBSERVABLE_AGREEMENT']={'status':'NOT_EVALUABLE_FROM_AVAILABLE_REALBIO_EVIDENCE'};out['endpoints']['RB4_NULL_SENSITIVITY']={'status':'NOT_EVALUABLE_FROM_AVAILABLE_REALBIO_EVIDENCE'}
 if 'P5' in x:
  cov=x['P5'];out['endpoints']['RB5_COVERAGE_TRANSFER']={'legacy_coverage':cov,'synthetic_reference_minimum':.85,'historical_realbio_target':.95,'status':'MEETS_SYNTHETIC_REFERENCE_ONLY_NOT_BIO_CONFIRMATION' if cov>=.85 else 'BELOW_SYNTHETIC_REFERENCE'}
 else:out['endpoints']['RB5_COVERAGE_TRANSFER']={'status':'NOT_EVALUABLE_FROM_AVAILABLE_REALBIO_EVIDENCE'}
 out['endpoints']['RB6_OUT_OF_UNIT_GENERALIZATION']={'status':'SUPPORTIVE_ONLY_SINGLE_UNIT' if t['expected_units']==1 else ('RETAINED_UNIT_EVIDENCE_PRESENT_BUT_NEW_LOO_METRIC_NOT_PREDECLARED' if candidate['unit_like_rows'] else 'NOT_EVALUABLE_FROM_AVAILABLE_UNIT_EVIDENCE')}
 statuses=[v['status'] for v in out['endpoints'].values()];out['terminal_status']='COMPLETED_REALBIO_TRACK_REPLAY_AND_TRANSLATION_ATTEMPT';out['has_new_raw_biological_confirmation']=False;out['any_replay_pass']=replay['status']=='FROZEN_SCORER_REPLAY_PASS';out['not_evaluable_count']=sum('NOT_EVALUABLE' in s for s in statuses);return out

def run():
 p=protocol();legacy=read(PKG/'inputs/LEGACY_REALBIO_V5_SUMMARY.json');out=ROOT/'results/realbio';out.mkdir(parents=True,exist_ok=True);hits=_discover();write(out/'SERVER_REALBIO_DISCOVERY.json',{'status':'COMPLETE','hits':hits,'roots':[str(x) for x in SEARCH_ROOTS]});auth=_find_authoritative(hits);write(out/'AUTHORITATIVE_DISCOVERY.json',{'files':[{k:v for k,v in a.items() if k!='json'} for a in auth]})
 tracks=[];budget={'used':0}
 for t in p['realbio']['tracks']:
  tokens=TRACK_FILES[t['id']];related=[h for h in hits if any(x in h['name'] or x in h['path'] for x in tokens)];replay=_replay_track(t,hits,budget);candidate=_candidate_evidence(t['id'],hits);r=_track_translation(t,legacy,replay,candidate,related);tracks.append(r);write(out/f"{t['id']}_{t['name']}.json",r)
 endpoint_counts={};replay_counts={}
 for r in tracks:
  replay_counts[r['replay_attempt']['status']]=replay_counts.get(r['replay_attempt']['status'],0)+1
  for k,v in r['endpoints'].items():endpoint_counts.setdefault(k,{}).setdefault(v['status'],0);endpoint_counts[k][v['status']]+=1
 result={'schema':'rari.omega.realbio.v2','status':'COMPLETED_ALL_NINE_REALBIO_TRACK_REPLAY_AND_TRANSLATION_ATTEMPTS','tracks':tracks,'endpoint_status_counts':endpoint_counts,'replay_status_counts':replay_counts,'authoritative_files_found':len(auth),'snapshot_bytes_used':budget['used'],'overall_biological_validation':'NOT_ESTABLISHED_BY_THIS_RETROSPECTIVE_REPLAY','historical_v5_preserved':'NO_TRACK_PASSED_HISTORICAL_FINAL_P1_P5_BATTERY','interpretation':'All nine archived biological lines were attempted. OMEGA R2 re-verifies retained authoritative/staging/unit-like evidence and attempts the frozen scorer in an isolated bounded snapshot where available. It does not call summary reuse a new prospective biological confirmation, does not pool heterogeneous tracks, and does not download entire public datasets. Missing exact within-unit arrays remain NOT_EVALUABLE rather than imputed.','stronger_claim_requirement':'A separately frozen raw-unit biological protocol with exact per-track schemas and prospective held-out units is required for new biological validation.'};write(out/'REALBIO_OMEGA_RESULTS.json',result);return {'results':'results/realbio/REALBIO_OMEGA_RESULTS.json','sha256':sha(out/'REALBIO_OMEGA_RESULTS.json'),'tracks':len(tracks),'terminal':True,'replay_status_counts':replay_counts}
if __name__=='__main__':print(json.dumps(run(),indent=2))
