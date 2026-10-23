from omega_common import *
import fcntl,os,pathlib
def service():return f'''[Unit]\nDescription=RARI OMEGA R2 final synthetic+realbio-replay+GOP terminal battery\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=exec\nWorkingDirectory={PKG}\nEnvironment=HOME=/home/deploy\nEnvironment=PATH=/home/deploy/.local/bin:/usr/local/bin:/usr/bin:/bin\nEnvironment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=PYTHONNOUSERSITE=1\nEnvironment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nEnvironment=MKL_NUM_THREADS=1\nEnvironment=PYTHONPATH={PKG}:{PKG}/reference\nExecStart={PYTHON} -B {PKG}/omega_engine.py\nRestart=no\nKillMode=control-group\nTimeoutStartSec=infinity\nTimeoutStopSec=120\nUMask=0077\n'''
def main():
 pf=preflight();ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'control').mkdir(exist_ok=True)
 if is_live(unit_props(UNIT)):print('RARI_OMEGA_ALREADY_RUNNING');return 0
 if (ROOT/'control/LAUNCH_INTENT.json').is_file():print('RARI_OMEGA_ALREADY_SUBMITTED_NO_AUTOMATIC_REPLAY');return 20
 with (PARENT/'rari_omega_final_v1.launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tests=cmd([str(PYTHON),'-B','-m','unittest','discover','-s',str(PKG/'tests'),'-v'],timeout=1200,cwd=PKG,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':str(PKG)+':'+str(PKG/'reference'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});print(tests.stdout);print(tests.stderr);require(tests.returncode==0,'OMEGA_PACKAGE_TESTS_FAILED')
  units=pathlib.Path('/home/deploy/.config/systemd/user');units.mkdir(parents=True,exist_ok=True);target=units/UNIT;data=service().encode()
  if target.exists():require(target.read_bytes()==data,'OMEGA_EXISTING_UNIT_DIFFERS')
  else:target.write_bytes(data)
  require(cmd(['systemctl','--user','daemon-reload']).returncode==0,'SYSTEMD_RELOAD_FAILED');write(ROOT/'control/PREFLIGHT.json',pf,replace=False);write(ROOT/'control/LAUNCH_INTENT.json',{'at':utc(),'unit':UNIT,'protocol_sha256':sha(PROTOCOL),'old_v4_results_lock_sha256':EXPECTED_V4_LOCK,'gop_core_sha256':EXPECTED_GOP_CORE,'provider_calls_authorized':0,'E2B_calls_authorized':0,'Kimi_calls_authorized':0,'positive_results_required':False},replace=False);require(cmd(['systemctl','--user','start',UNIT],timeout=30).returncode==0,'OMEGA_SERVICE_START_FAILED');print('RARI_OMEGA_FINAL_SUBMITTED');print('UNIT='+UNIT);print('Old V4/V7 locks and GOP core are read-only; no E2B/Kimi/provider calls are authorized.')
 return 0
if __name__=='__main__':raise SystemExit(main())
