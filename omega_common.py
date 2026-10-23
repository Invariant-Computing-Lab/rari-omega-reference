from __future__ import annotations
import datetime,hashlib,json,os,pathlib,shutil,subprocess,tempfile,zipfile
from typing import Any
PARENT=pathlib.Path('/home/deploy/rari_codex/rebuilds/astra_d0fd3951c099ce94')
V4=PARENT/'neurobrain_final_v4'
V1=PARENT/'neurobrain_v1'
PKG=PARENT/'rari_omega_package_v2'
ROOT=PARENT/'rari_omega_final_v2'
PYTHON=pathlib.Path('/home/deploy/rari_cortex_kimi_k25_hf/venv/bin/python')
UNIT='rari-omega-final-v2.service'
PROTOCOL=PKG/'OMEGA_PROTOCOL.json'
EXPECTED_V4_LOCK='1fe80f4d8714ec70368a7307c10ea10f3f165706de42837ef3dafb2e789c70d5'
EXPECTED_GOP_CORE='cc99f94bd75a67c0035c8342ef3fdc4ba1d937a586bef2ede4354f1d829ce8b7'
class GateError(RuntimeError):pass
def require(c,m):
 if not c:raise GateError(m)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(v:Any)->bytes:return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(v:Any)->str:return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
 h=hashlib.sha256();p=pathlib.Path(p)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def optional(p):
 p=pathlib.Path(p);return read(p) if p.is_file() else None
def write(p,v,replace=True):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=canonical(v)
 if p.exists() and not replace:
  require(p.read_bytes()==b,'IMMUTABLE_VALUE_CHANGED:'+str(p));return
 fd,tmp=tempfile.mkstemp(prefix='.omega-',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def cmd(args,timeout=60,cwd=None,env=None):
 return subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout,cwd=cwd,env=env,close_fds=True)
def unit_props(name):
 fs=['LoadState','ActiveState','SubState','MainPID','Result','ExecMainStatus','InvocationID']
 p=cmd(['systemctl','--user','show',name,*['-p'+x for x in fs]],timeout=15)
 d=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x);d['query_exit_code']=p.returncode;return d
def is_live(d):return d.get('ActiveState') in {'active','activating','deactivating','reloading'} or int(d.get('MainPID','0') or 0)>0
def protocol():return read(PROTOCOL)
def all_files(root):return [p for p in sorted(pathlib.Path(root).rglob('*')) if p.is_file() and not p.is_symlink()]
def file_manifest(root,exclude=()):
 root=pathlib.Path(root);ex=set(exclude);return {str(p.relative_to(root)):sha(p) for p in all_files(root) if str(p.relative_to(root)) not in ex}
def verify_gop_archive():
 z=PKG/'inputs/GOP_TOE_FOUNDATION_v1_0.zip';require(z.is_file(),'GOP_ARCHIVE_MISSING')
 with zipfile.ZipFile(z) as q:
  report=json.loads(q.read('GOP_TOE_FOUNDATION_v1_0/RELEASE_REPORT.json'))
  require(report['core_sha256']==EXPECTED_GOP_CORE,'GOP_CORE_HASH_DECLARATION_CHANGED')
 return {'zip_sha256':sha(z),'core_sha256':report['core_sha256'],'release_status':report['release_status'],'universal_claim_status':report['universal_claim_status']}
def preflight():
 require(PROTOCOL.is_file(),'OMEGA_PROTOCOL_MISSING');p=protocol()
 require(V4.is_dir() and V1.is_dir(),'NEUROBRAIN_BASE_MISSING')
 require(sha(V4/'RESULTS_LOCK.json')==EXPECTED_V4_LOCK,'V4_LOCK_CHANGED')
 v7=read(V4/'FINAL_STATUS_V7.json');a7=read(V4/'results/INDEPENDENT_AUDIT_V7.json')
 require(v7.get('status')=='COMPLETED_FORMULATED_INDEPENDENTLY_AUDITED' and v7.get('independent_approval') is True,'V7_NOT_FINAL')
 require(a7.get('approved') is True and not a7.get('issues'),'V7_AUDIT_NOT_APPROVED')
 clean=read(V4/'control/CLEANUP.json');c=clean.get('counts',{})
 require(clean.get('all_paused') and c.get('paused')==100 and c.get('running')==0 and c.get('unknown')==0 and c.get('conflict')==0,'OLD_COHORT_NOT_QUIESCENT')
 for u in ('rari-neurobrain-final-v4.service','rari-neurobrain-final-guard-v4.service','rari-neurobrain-audit-v5.service','rari-neurobrain-audit-v6.service','rari-neurobrain-audit-v7.service','rari-omega-final-v1.service'):
  require(not is_live(unit_props(u)),u+':STILL_ACTIVE')
 led=read(V1/'BUDGET_STATE.json');require(led.get('active_since_epoch') is None and not led.get('holds'),'OLD_SHARED_BUDGET_NOT_QUIESCENT')
 free=shutil.disk_usage(PARENT).free
 ex=p['execution'];pkg_bytes=sum(x.stat().st_size for x in all_files(PKG)) if PKG.exists() else 0
 projected=pkg_bytes+int(ex['realbio_replay_snapshot_cap_total_bytes'])+int(ex['synthetic_workspace_budget_bytes'])+int(ex['archive_headroom_bytes'])
 required=max(int(ex['minimum_free_bytes']),projected)
 require(free>=required,'DISK_HEADROOM_TOO_LOW:free=%d:required=%d'%(free,required))
 g=verify_gop_archive()
 return {'status':'PASS','at':utc(),'v4_results_lock_sha256':EXPECTED_V4_LOCK,'v7_audit_sha256':sha(V4/'results/INDEPENDENT_AUDIT_V7.json'),
         'old_workers':c,'old_budget':{'active_since_epoch':led.get('active_since_epoch'),'holds':led.get('holds'),'available_usd':led.get('available_usd')},
         'free_bytes':free,'required_free_bytes':required,'projected_working_set_bytes':projected,'gop':g,'protocol_sha256':sha(PROTOCOL),'provider_calls_authorized':0,'E2B_calls_authorized':0,'Kimi_calls_authorized':0}
def phase(name,fn):
 d=ROOT/'phases'/name;commit=d/'COMMIT.json'
 if commit.is_file():return read(commit)
 d.mkdir(parents=True,exist_ok=True);started=utc()
 try:
  out=fn();out={} if out is None else out
  c={'phase':name,'status':'COMMITTED','started_at':started,'completed_at':utc(),'outputs':out}
  write(commit,c,replace=False);return c
 except Exception as e:
  write(d/'FAILURE.json',{'phase':name,'at':utc(),'error':type(e).__name__+': '+str(e)},replace=True);raise
