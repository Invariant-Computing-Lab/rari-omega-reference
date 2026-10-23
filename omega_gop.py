from __future__ import annotations
import json,math,os,pathlib,shutil,subprocess,sys,zipfile
import numpy as np
from omega_common import *

def qfeatures(X):
 X=np.asarray(X,float);cols=[np.ones(len(X))];cols += [X[:,i] for i in range(X.shape[1])]
 for i in range(X.shape[1]):
  for j in range(i,X.shape[1]):cols.append(X[:,i]*X[:,j])
 return np.column_stack(cols)
def fit_quad(X,Y):
 F=qfeatures(X);coef=np.linalg.lstsq(F,Y,rcond=1e-12)[0];return coef
def pred_quad(coef,X):return qfeatures(X)@coef
def rk4_vec(fn,x,h,n=1):
 x=np.asarray(x,float);hh=h/n
 for _ in range(n):
  k1=fn(x);k2=fn(x+hh*k1/2);k3=fn(x+hh*k2/2);k4=fn(x+hh*k3);x=x+hh*(k1+2*k2+2*k3+k4)/6
 return x
def g1_gauge():
 rng=np.random.default_rng(6101);x=rng.normal(size=(256,2));H=np.array([[1.2,-.3],[.4,.9]]);g=1.7;y=g*(x@H.T);c=2.3;xp=c*x;gp=g/c;yp=gp*(xp@H.T);err=float(np.max(abs(y-yp)))
 return {'id':'G1_UNKNOWN_SENSOR_GAIN_GAUGE','status':'SUPPORT_NEGATIVE_CONTROL_NONIDENTIFIABLE','max_observation_difference':err,'equivalence':'x*=c x, gain*=gain/c','claim':'unknown global sensor gain cannot be separated from latent scale without an anchor'}
def g2_inverse_square():
 def f(z):
  x=z[:2];v=z[2:];r=np.linalg.norm(x);return np.r_[v,-x/(r**3)]
 rows=[]
 for k,(r0,vt) in enumerate([(1.,1.),(1.2,.85),(.8,1.1),(1.5,.7)]):
  z=np.array([r0,0.,0.,vt]);
  for t in range(800):
   rows.append(np.r_[z,f(z)]);z=rk4_vec(f,z,.005,2)
 a=np.asarray(rows);X=a[:,:4];Y=a[:,4:];cut=int(.6*len(X));coef=fit_quad(X[:cut],Y[:cut]);pr=pred_quad(coef,X[cut:]);err=pr-Y[cut:]
 return {'id':'G2_CENTRAL_INVERSE_SQUARE','status':'CHALLENGE_DIRECT_QUADRATIC_EXACTNESS' if np.max(abs(err))>1e-6 else 'SUPPORT_DIRECT_QUADRATIC','test_mse':float(np.mean(err**2)),'test_max_abs_derivative_error':float(np.max(abs(err))),'scope':'direct coordinate quadratic derivative fit on bounded central-force trajectories; auxiliary lifts/GR not tested'}
def g3_quintic_stochastic():
 # x SDE: dx=(-a x-b x^5)dt+sigma dW. Exact Ito lift [x,v=x^2,w=x^4] has quadratic drift/diffusion.
 a,b,s=.9,.6,.2;rng=np.random.default_rng(6103);base=rng.normal(size=(2048,512));rows=[]
 for n in (64,128,256,512):
  group=512//n;noise=base.reshape(2048,n,group).sum(2)/math.sqrt(group);h=.5/n;x=np.full(2048,.5);v=x*x;w=v*v
  for t in range(n):
   dW=math.sqrt(h)*noise[:,t]
   # simultaneous EM from old state, exact Ito coefficients
   x0,v0,w0=x.copy(),v.copy(),w.copy()
   x=x0+(-a*x0-b*x0*w0)*h+s*dW
   v=v0+(-2*a*v0-2*b*v0*w0+s*s)*h+(2*s*x0)*dW
   w=w0+(-4*a*w0-4*b*w0*w0+6*s*s*v0)*h+(4*s*x0*v0)*dW
  rms2=float(np.sqrt(np.mean((v-x*x)**2)));rms4=float(np.sqrt(np.mean((w-x**4)**2)));rows.append({'steps':n,'x2_manifold_rms':rms2,'x4_manifold_rms':rms4})
 decreasing=rows[-1]['x4_manifold_rms']<rows[0]['x4_manifold_rms'];met=rows[-1]['x4_manifold_rms']<=1e-5
 return {'id':'G3_QUINTIC_STOCHASTIC_LIFT','status':'NUMERICAL_TOLERANCE_MET' if met else 'EXACT_ITO_LIFT_DERIVED_NUMERICAL_TOLERANCE_NOT_MET','exact_lift':{'dx':'-a*x-b*x*w; diffusion sigma','dv':'-2a*v-2b*v*w+sigma^2; diffusion 2sigma*x','dw':'-4a*w-4b*w^2+6sigma^2*v; diffusion 4sigma*x*v'},'rows':rows,'convergence_direction':decreasing,'tolerance':1e-5}
def g4_s05():
 rng=np.random.default_rng(6104);data=[]
 for world in range(24):
  mode=world%3;met=.16;cost=.08;gain=rng.uniform(.7,1.1);leak=.12;nonlinear=world%4==3;u=.45+rng.uniform(-.2,.2,160);x=np.array([rng.uniform(.65,1.4),0.,.5]);
  for t in range(160):
   r,m,act=x;inp=u[t]*act/(.5+act) if nonlinear else u[t]*act;raw=r+.1*(inp-met-cost*act**2);nr=np.clip(raw,0,2)
   if mode==0:nm,na=0.,.5
   elif mode==1:nm,na=0.,np.clip(.5+gain*(1-r),0,1)
   else:nm=np.clip((1-.1*leak)*m+.1*(1-r),-1,1);na=np.clip(.5+gain*(1-r)+m,0,1)
   nxt=np.array([nr,nm,na]);data.append((world,t,np.r_[x,u[t]],nxt));x=nxt
 X=np.array([r[2] for r in data]);Y=np.array([r[3] for r in data]);train=np.array([r[0]<16 for r in data]);coef=fit_quad(X[train],Y[train]);e=pred_quad(coef,X[~train])-Y[~train]
 return {'id':'G4_S05_RESOURCE_SATURATION','status':'UNMAPPED_EXACT_FROZEN_QUADRATIC_CORE' if np.max(abs(e))>1e-6 else 'APPROXIMATED_WITHIN_TOLERANCE','heldout_mse':float(np.mean(e**2)),'heldout_max_abs':float(np.max(abs(e))),'scope':'legacy clipped resource/saturating-harvest trajectories; direct quadratic one-step map; clipping/rational mechanism retained'}
def g5_s07():
 rng=np.random.default_rng(6105);data=[]
 for world in range(24):
  decay=rng.uniform(.65,.8);common=world%3==0;coupling=0 if common else rng.uniform(.06,.16);c12,c21=(1.8*coupling,0.) if world%4==3 and not common else (coupling,coupling);g=rng.uniform(.7,1.3,2);x=rng.normal(0,.4,2);u=rng.uniform(-.3,.3,(160,2));
  if common:u[:,1]=u[:,0]
  for t in range(160):
   nxt=decay*x+np.array([c12*np.tanh(x[1]-x[0]),c21*np.tanh(x[0]-x[1])])+g*u[t];data.append((world,np.r_[x,u[t]],nxt));x=nxt
 X=np.array([r[1] for r in data]);Y=np.array([r[2] for r in data]);train=np.array([r[0]<16 for r in data]);coef=fit_quad(X[train],Y[train]);e=pred_quad(coef,X[~train])-Y[~train]
 return {'id':'G5_S07_TANH_COUPLING','status':'UNMAPPED_EXACT_FROZEN_QUADRATIC_CORE' if np.max(abs(e))>1e-6 else 'APPROXIMATED_WITHIN_TOLERANCE','heldout_mse':float(np.mean(e**2)),'heldout_max_abs':float(np.max(abs(e))),'scope':'legacy tanh-coupling family; direct finite quadratic one-step approximation only'}
def _bloch_density(r):
 x,y,z=r;return .5*np.array([[1+z,x-1j*y],[x+1j*y,1-z]],complex)
def g6_quantum_learned():
 rng=np.random.default_rng(6106);gamma=.27
 def channel(r):return np.array([math.sqrt(1-gamma)*r[0],math.sqrt(1-gamma)*r[1],(1-gamma)*r[2]+gamma])
 def sample(n):
  q=[]
  while len(q)<n:
   r=rng.uniform(-1,1,3)
   if np.linalg.norm(r)<=1:q.append(r)
  return np.array(q)
 tr=sample(64);te=sample(256);F=np.c_[np.ones(len(tr)),tr];coef=np.linalg.lstsq(F,np.array([channel(r) for r in tr]),rcond=None)[0];pred=np.c_[np.ones(len(te)),te]@coef;tar=np.array([channel(r) for r in te]);mse=float(np.mean((pred-tar)**2));mineig=min(float(np.linalg.eigvalsh(_bloch_density(r)).min()) for r in pred)
 # unital no-intercept comparator intentionally cannot learn amplitude damping translation.
 c0=np.linalg.lstsq(tr,np.array([channel(r) for r in tr]),rcond=None)[0];p0=te@c0;mse0=float(np.mean((p0-tar)**2))
 return {'id':'G6_LEARNED_QUANTUM_CHANNEL_COMPARATOR','status':'FINITE_LEARNED_CHANNEL_COMPARATOR_PASS' if mse<1e-10 and mineig>-1e-10 else 'FINITE_COMPARATOR_INCONCLUSIVE','heldout_bloch_mse':mse,'minimum_predicted_density_eigenvalue':mineig,'unital_no_intercept_mse':mse0,'scope':'synthetic single-qubit amplitude-damping channel identification; does not derive GKSL/Born rules or validate physical quantum brain claims'}
def _partial_trace_memory(rho):
 # rho indices system,memory. Return system trace over memory.
 r=rho.reshape(2,2,2,2);return np.einsum('imjm->ij',r)
def g7_memory():
 I=np.eye(2)/2;z0=np.array([[1,0],[0,0]],complex);z1=np.array([[0,0],[0,1]],complex);rho0=np.kron(I,z0);rho1=np.kron(I,z1)
 # SWAP maps memory state onto system: same reduced system before, maximally different after.
 U=np.zeros((4,4),complex)
 for s in range(2):
  for m in range(2):U[m*2+s,s*2+m]=1
 a0=_partial_trace_memory(rho0);a1=_partial_trace_memory(rho1);b0=_partial_trace_memory(U@rho0@U.conj().T);b1=_partial_trace_memory(U@rho1@U.conj().T);before=float(np.linalg.norm(a0-a1,ord='nuc')/2);after=float(np.linalg.norm(b0-b1,ord='nuc')/2)
 return {'id':'G7_REDUCED_QUANTUM_MEMORY','status':'REDUCED_HISTORY_FREE_CLOSURE_CHALLENGED','before_reduced_trace_distance':before,'after_same_dynamics_trace_distance':after,'scope':'finite two-qubit memory witness; augmented joint state remains adequate'}
def _extract_gop():
 dst=ROOT/'gop_work/GOP_TOE_FOUNDATION_v1_0'
 if dst.exists():return dst
 dst.parent.mkdir(parents=True,exist_ok=True);z=PKG/'inputs/GOP_TOE_FOUNDATION_v1_0.zip'
 with zipfile.ZipFile(z) as q:
  for i in q.infolist():
   p=pathlib.PurePosixPath(i.filename);require(not p.is_absolute() and '..' not in p.parts,'GOP_ZIP_PATH_ESCAPE')
  q.extractall(dst.parent)
 return dst
def _integrate(root,result_path):
 (ROOT/'delivery').mkdir(parents=True,exist_ok=True)
 study=ROOT/'gop_study';study.mkdir(parents=True,exist_ok=True);src=study/'omega_gop_results.json';shutil.copy2(result_path,src)
 rec={'schema_version':'1.0','template_only':False,'study_id':'RARI_OMEGA_FINAL_V1_GOP_SIDECAR','evidence_id':'RARI_OMEGA_GOP_FINITE_OBLIGATIONS_V1','foundation_id':'GOP_TOE_FOUNDATION_v1_0','core_sha256':EXPECTED_GOP_CORE,'primary_program':'RARI',
 'source':{'path':'omega_gop_results.json','sha256':sha(src),'locator':'complete JSON result','access_level':'report_only'},'evidence_type':'synthetic_experiment','source_claim':'Finite sidecar executes previously unexecuted bounded GOP obligations and preserves unresolved universal sectors.','integration_interpretation':'Constrains finite representability/identification scope; does not validate universal U.','target_claim_ids':['U','B-LIFT','P-ID','P-Q'],'proposed_relation':'constrains','assumption_mapping':[{'source_assumption':'Finite synthetic numerical sidecar; no external physical measurement','core_assumption_id':'A-PHYS','compatibility':'matched','reason':'The sidecar explicitly imports physical axioms and does not claim to derive them.'},{'source_assumption':'Nondimensional finite simulations only','core_assumption_id':'A-UNIT','compatibility':'matched','reason':'All sidecar quantities are nondimensional unless explicitly noted.'}],'scope':{'system_or_population':'synthetic finite mathematical/physical control systems','independent_unit':'declared finite trajectory/world/channel case','information_access':'oracle or synthetic observations as labelled per test','units_and_scales':'nondimensional unless noted','intervention_status':'mixed'},'lineage_group':'RARI_OMEGA_FINAL_V1','reproduction_status':'fresh_data_internal_replication','review_status':'unreviewed','requested_effect':'evidence_only','core_change_requested':False,'limitations':['No GR/QFT/Lorentz/global apparatus/human-clinical validation','Finite simulations do not prove universal claim']}
 (study/'study_record.json').write_text(json.dumps(rec,indent=2)+'\n')
 env={**os.environ,'PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 steps=[]
 for args in [
  [str(PYTHON),'-B','-m','gop_intake','ingest','--core',str(root),'--record',str(study/'study_record.json')],
  [str(PYTHON),'-B','-m','gop_intake','assess','--core',str(root),'--study-id',rec['study_id']],
  [str(PYTHON),'-B','-m','gop_intake','verify','--core',str(root)],
  [str(PYTHON),'-B','-m','gop_intake','export','--core',str(root),'--study-id',rec['study_id'],'--out',str(ROOT/'delivery/GOP_OMEGA_ADDON.zip')]]:
   pr=cmd(args,timeout=300,cwd=root,env=env);steps.append({'args':args[3:] if len(args)>3 else args,'returncode':pr.returncode,'stdout_tail':pr.stdout[-2000:],'stderr_tail':pr.stderr[-2000:]});require(pr.returncode==0,'GOP_INTAKE_STEP_FAILED:'+str(args))
 return steps
def run():
 root=_extract_gop();rep=read(root/'RELEASE_REPORT.json');require(rep['core_sha256']==EXPECTED_GOP_CORE,'GOP_CORE_DECLARATION_CHANGED')
 env={**os.environ,'PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 tests=cmd([str(PYTHON),'-B','-m','unittest','discover','-s','tests','-v'],timeout=900,cwd=root,env=env);require(tests.returncode==0,'GOP_FOUNDATION_TESTS_FAILED')
 verify=cmd([str(PYTHON),'-B','-m','gop_intake','verify','--core',str(root)],timeout=300,cwd=root,env=env);require(verify.returncode==0,'GOP_FOUNDATION_VERIFY_FAILED')
 finite=[g1_gauge(),g2_inverse_square(),g3_quintic_stochastic(),g4_s05(),g5_s07(),g6_quantum_learned(),g7_memory()]
 unresolved=[{'id':x,'status':'UNRESOLVED_NOT_TESTED_BY_THIS_BATTERY'} for x in protocol()['gop_sidecar']['always_unresolved_unless_external_new_evidence']]
 out=ROOT/'results/gop';out.mkdir(parents=True,exist_ok=True);result={'schema':'rari.omega.gop_sidecar.v1','foundation_id':rep['foundation_id'],'candidate_id':rep['candidate_id'],'source_release_status':rep['release_status'],'source_universal_claim':rep['universal_claim_status'],'core_sha256':rep['core_sha256'],'core_mutated':False,'foundation_tests':'PASS','foundation_verify':'PASS','finite_tests':finite,'unresolved':unresolved,'overall_status':'COMPLETED_FINITE_SIDECAR_UNIVERSAL_CLAIM_REMAINS_UNESTABLISHED'};write(out/'GOP_OMEGA_RESULTS.json',result)
 intake=_integrate(root,out/'GOP_OMEGA_RESULTS.json');write(out/'GOP_INTAKE_EXECUTION.json',{'steps':intake,'core_sha256_expected':EXPECTED_GOP_CORE,'note':'Working-copy evidence intake only; core files remain manifest-protected.'})
 return {'results':'results/gop/GOP_OMEGA_RESULTS.json','sha256':sha(out/'GOP_OMEGA_RESULTS.json'),'addon':'delivery/GOP_OMEGA_ADDON.zip','terminal':True}
if __name__=='__main__':print(run())
