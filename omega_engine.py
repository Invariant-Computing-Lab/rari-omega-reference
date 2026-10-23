from __future__ import annotations
import json,os,pathlib,sys,traceback
from omega_common import *
import omega_synth,omega_realbio,omega_gop
from omega_deliver import build_archive,upload

def main():
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'results').mkdir(exist_ok=True);(ROOT/'control').mkdir(exist_ok=True);(ROOT/'delivery').mkdir(exist_ok=True)
 try:
  pf=preflight();
  if not (ROOT/'control/PREFLIGHT.json').is_file(): write(ROOT/'control/PREFLIGHT.json',pf,replace=False)
  freeze={'schema':'rari.omega.freeze.v1','at':utc(),'protocol_sha256':sha(PROTOCOL),'source_v4_results_lock_sha256':EXPECTED_V4_LOCK,'source_v7_audit_sha256':sha(V4/'results/INDEPENDENT_AUDIT_V7.json'),'gop_core_sha256':EXPECTED_GOP_CORE,'positive_outcome_required':False,'old_scientific_locks_mutable':False,'gop_core_mutable':False,'provider_calls_authorized':0,'E2B_calls_authorized':0,'Kimi_calls_authorized':0};write(ROOT/'OMEGA_FREEZE.json',freeze,replace=False)
  phase('synthetic_confirmation',omega_synth.run)
  phase('realbio_nine_track_translation',omega_realbio.run)
  phase('gop_finite_sidecar',omega_gop.run)
  def formulate():
   s=read(ROOT/'results/synthetic/SYNTHETIC_OMEGA_RESULTS.json');b=read(ROOT/'results/realbio/REALBIO_OMEGA_RESULTS.json');g=read(ROOT/'results/gop/GOP_OMEGA_RESULTS.json')
   summary={'schema':'rari.omega.formulation.v1','status':'ALL_DECLARED_OUTCOMES_FORMULATED','synthetic_cards':s['hypothesis_cards'],'biological_overall':b['overall_biological_validation'],'gop_finite':{x['id']:x['status'] for x in g['finite_tests']},'gop_unresolved':{x['id']:x['status'] for x in g['unresolved']},'interpretive_rule':'SUPPORT, CHALLENGE, INCONCLUSIVE, NOT_EVALUABLE and UNRESOLVED are all terminal scientific outcomes; none is rewritten to force success.','cross_program_conclusion':'Synthetic RARI can be prospectively retested; archived heterogeneous biological V5 evidence does not establish biological validation; GOP finite sidecars constrain bounded scope while universal U remains unestablished.'};write(ROOT/'results/OMEGA_FORMULATION.json',summary);return {'results':'results/OMEGA_FORMULATION.json','sha256':sha(ROOT/'results/OMEGA_FORMULATION.json')}
  phase('formulation',formulate)
  def audit_phase():
   a=cmd([str(PYTHON),'-B',str(PKG/'omega_audit.py')],timeout=900,cwd=PKG,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});print(a.stdout);print(a.stderr,file=sys.stderr);require(a.returncode==0,'OMEGA_INDEPENDENT_AUDIT_FAILED');return {'audit':'results/INDEPENDENT_AUDIT_OMEGA.json','sha256':sha(ROOT/'results/INDEPENDENT_AUDIT_OMEGA.json')}
  phase('independent_audit',audit_phase)
  audit=read(ROOT/'results/INDEPENDENT_AUDIT_OMEGA.json');status_name='COMPLETED_ALL_DECLARED_ATTEMPTS_FORMULATED_AND_AUDITED' if audit.get('approved') else 'COMPLETED_ATTEMPTS_AUDIT_QUALIFIED'
  manifest={'protocol_sha256':sha(PROTOCOL),'freeze_sha256':sha(ROOT/'OMEGA_FREEZE.json'),'results':{str(p.relative_to(ROOT)):sha(p) for p in all_files(ROOT/'results')},'phases':{str(p.relative_to(ROOT)):sha(p) for p in all_files(ROOT/'phases')},'old_v4_results_lock_sha256':sha(V4/'RESULTS_LOCK.json'),'gop_core_sha256':EXPECTED_GOP_CORE};write(ROOT/'control/FINAL_MANIFEST.json',manifest)
  lock={'schema':'rari.omega.results_lock.v1','at':utc(),'status':status_name,'independent_approval':audit.get('approved') is True,'old_v4_scientific_outputs_unchanged':sha(V4/'RESULTS_LOCK.json')==EXPECTED_V4_LOCK,'gop_frozen_core_unchanged':True,'protocol_sha256':sha(PROTOCOL),'final_manifest_sha256':sha(ROOT/'control/FINAL_MANIFEST.json')};write(ROOT/'RESULTS_LOCK.json',lock)
  status={'schema':'rari.omega.final.v2','at':utc(),'status':status_name,'synthetic_attempts_complete':True,'realbio_tracks_attempted':9,'gop_finite_tests_attempted':7,'gop_universal_claim':'CONJECTURE_UNESTABLISHED','independent_approval':lock['independent_approval'],'positive_result_required':False,'provider_calls':0,'E2B_calls':0,'Kimi_calls':0,'drive_status':'PENDING'};write(ROOT/'FINAL_STATUS.json',status)
  build_archive()
  try:dr=upload()
  except Exception as e:dr={'status':'UPLOAD_OR_VERIFICATION_PENDING','error':str(e)};write(ROOT/'control/DRIVE_RECEIPT.json',dr)
  status=read(ROOT/'FINAL_STATUS.json');status['drive_status']=dr['status'];write(ROOT/'FINAL_STATUS.json',status);build_archive();
  if dr.get('status')=='VERIFIED_IMMUTABLE_ARCHIVE':
   try: dr=upload()
   except Exception as e: dr={'status':'FINAL_REUPLOAD_OR_VERIFICATION_PENDING','error':str(e)};write(ROOT/'control/DRIVE_RECEIPT.json',dr)
   status=read(ROOT/'FINAL_STATUS.json');status['drive_status']=dr['status'];write(ROOT/'FINAL_STATUS.json',status)
  print(json.dumps({'status':status,'audit':audit,'drive':dr},indent=2));return 0
 except BaseException as e:
  failure={'at':utc(),'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'provider_calls':0,'E2B_calls':0,'Kimi_calls':0,'old_v4_lock_sha256':sha(V4/'RESULTS_LOCK.json') if (V4/'RESULTS_LOCK.json').is_file() else None};ROOT.mkdir(parents=True,exist_ok=True);write(ROOT/'FAILURE.json',failure);print(json.dumps(failure,indent=2),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
