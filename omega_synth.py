from __future__ import annotations
import hashlib,math,pathlib,sys
import numpy as np
from omega_common import *
sys.path.insert(0,str(PKG/'reference'))
from reference.rari_neurobrain.experiment import collect,dataset,cross_observer_analysis,matrix,electrode_boundary
from reference.rari_neurobrain.world import Bank,HiddenFilter,seed_for
from reference.rari_neurobrain.subject import Brain,Perturbation
from reference.rari_neurobrain.science import TASKS,train_models,predict_models,infer_filter,interventions,adaptive_next
from reference.rari_neurobrain.evaluation import conformal_radius
from omega_localfleet import LocalFleet

def boot(v,q=(.025,.975),seed=20261006,draws=1999):
 a=np.asarray(v,float);rng=np.random.default_rng(seed);b=np.array([rng.choice(a,len(a),replace=True).mean() for _ in range(draws)])
 return {'mean':float(a.mean()),'ci':np.quantile(b,q).tolist(),'draws':draws,'seed':seed,'n':len(a)}
def signflip(v,seed=20261007,draws=1999):
 a=np.asarray(v,float);obs=float(a.mean());rng=np.random.default_rng(seed);n=sum(float((a*rng.choice([-1,1],len(a))).mean())>=obs for _ in range(draws));return (n+1)/(draws+1)
def family_losses(pred,targets,scales):
 out={}
 for name in ('pca','classical_phase','quantum_phase'):
  for k in range(2,31):
   vals=[]
   for task in TASKS:
    y=np.asarray(targets[task]);p=np.asarray(pred[f'{name}/{k}/{task}']);vals.append(np.mean((p-y)**2,axis=1)/scales[task])
   out[f'{name}/{k}']=np.mean(np.vstack(vals),axis=0)
 return out
def _scores(models,records,scales):
 x,y=dataset(records);pred=predict_models(models,x);return pred,y,family_losses(pred,y,scales)
def _direct_contrast(records,metric,ref,cand='balanced'):
 fs=sorted({r['family'] for r in records});v=[];absd=[]
 for f in fs:
  a=np.mean([r[metric] for r in records if r['family']==f and r['policy']==cand]);b=np.mean([r[metric] for r in records if r['family']==f and r['policy']==ref]);v.append((b-a)/max(b,1e-8));absd.append(b-a)
 return np.asarray(v),np.asarray(absd)
def _common_summary(records):
 r=cross_observer_analysis(records);valid=[x for x in r['families'] if x.get('dimension',0)>0]
 return {'eligible':len(valid),'empty':len(r['families'])-len(valid),'same_mean':float(np.mean([x['disagreement'] for x in valid])) if valid else None,
 'wrong_mean':float(np.mean([x['wrong_pair_mse'] for x in valid])) if valid else None,'target_mean':float(np.mean([x['target_mse'] for x in valid])) if valid else None,
 'raw_mean':float(np.mean([x['raw_disagreement'] for x in valid])) if valid else None,'rows':r['families']}
def _coverage(cal,records,models,selected,scales):
 x,y=dataset(cal);cp=predict_models(models,x);radii={}
 for task in TASKS:
  p=np.asarray(cp[f'{selected}/{task}']);yy=np.asarray(y[task]);res=np.sqrt(np.mean((p-yy)**2,axis=1));radii[task]=float(conformal_radius(res,.1))
 cfs=sorted({r['family'] for r in cal});direct=np.array([np.sqrt(np.mean([r['reconstruction_mse'] for r in cal if r['family']==f and r['policy']=='balanced'])) for f in cfs]);dr=float(conformal_radius(direct,.1))
 out={'radii':radii,'direct_radius':dr,'nominal':.9}
 for label,rs in records.items():
  xx,yy=dataset(rs);pp=predict_models(models,xx);task_cov={}
  for task in TASKS:
   res=np.sqrt(np.mean((np.asarray(pp[f'{selected}/{task}'])-np.asarray(yy[task]))**2,axis=1));task_cov[task]=float(np.mean(res<=radii[task]))
  fs=sorted({r['family'] for r in rs});dd=np.array([np.sqrt(np.mean([r['reconstruction_mse'] for r in rs if r['family']==f and r['policy']=='balanced'])) for f in fs]);out[label]={'direct':float(np.mean(dd<=dr)),'tasks':task_cov}
 return out
def _excitation(bankkey):
 bank=Bank(bankkey);fleet=LocalFleet();rows=[];counts=protocol()['synthetic']['hypotheses']['S7_EXCITATION_SATURATION']['probe_counts'];grid=np.array([[a,b] for a in np.linspace(-.5,.5,7) for b in np.linspace(-.5,.5,7)])
 for family in range(3000,3032):
  _,world,f=bank.episode('null',family,0);initial=np.asarray(world.state);truthA=matrix(f)
  for count in counts:
   u=interventions('balanced',count,0);fleet.restore(Brain().checkpoint());obs=fleet.block([f.observe((initial+z).tolist()) for z in u],[Perturbation()]*len(u),operation_id=f'excite/{family}/{count}',reset_each=True);y=np.asarray([electrode_boundary(x) for x in obs]);est=infer_filter(u,y)
   raw=grid@np.asarray(est['A']).T+np.asarray(est['offset']);pred=np.tanh(raw) if est['mode']=='tanh' else raw;target=np.asarray([f.observe((initial+z).tolist()) for z in grid]);hold=float(np.mean((pred-target)**2));rec=np.asarray(est['world_representative']);P=np.asarray(est['observable_projection']);trueP=np.linalg.pinv(truthA)@truthA
   rows.append({'family':family,'probes':count,'heldout_mse':hold,'reconstruction_mse':float(np.mean((P@rec-trueP@initial)**2)),'J_rank':est['parameter_identifiability']['J_P_rank'],'J_condition':est['parameter_identifiability']['condition'],'A_rank':est['rank']})
 agg={str(c):{'heldout_mean':float(np.mean([r['heldout_mse'] for r in rows if r['probes']==c])),'reconstruction_mean':float(np.mean([r['reconstruction_mse'] for r in rows if r['probes']==c])),'full_J_fraction':float(np.mean([r['J_rank']==6 for r in rows if r['probes']==c]))} for c in counts}
 return {'rows':rows,'aggregate':agg,'interpretation':'Prospective excitation/saturation diagnostic; no universal threshold inferred automatically.'}
def _out_of_family(bankkey):
 bank=Bank(bankkey);grid=np.array([[a,b] for a in np.linspace(-.5,.5,7) for b in np.linspace(-.5,.5,7)]);train=interventions('balanced',12,0);rows=[]
 for family in range(4000,4032):
  _,world,f=bank.episode('null',family,0);initial=np.asarray(world.state);A=matrix(f);off=np.zeros(2)
  def lin(u):return A@(initial+u)
  transforms={
   'clipped':lambda u,prev=None:np.clip(lin(u),-.35,.35),
   'deadzone':lambda u,prev=None:np.where(np.abs(lin(u))<.15,0,lin(u)),
   'cubic_mixing':lambda u,prev=None:(lambda x:x+.25*x**3)(lin(u))}
  for name,fn in transforms.items():
   y=np.array([fn(u) for u in train]);est=infer_filter(train,y);raw=grid@np.asarray(est['A']).T+np.asarray(est['offset']);pred=np.tanh(raw) if est['mode']=='tanh' else raw;tar=np.array([fn(u) for u in grid]);rows.append({'family':family,'transform':name,'heldout_mse':float(np.mean((pred-tar)**2)),'mode':est['mode'],'rank':est['rank']})
  prev=np.zeros(2);ys=[]
  for u in train:
   cur=.7*lin(u)+.3*prev;ys.append(cur);prev=cur
  est=infer_filter(train,np.array(ys));prev=np.zeros(2);tar=[]
  for u in grid:
   cur=.7*lin(u)+.3*prev;tar.append(cur);prev=cur
  raw=grid@np.asarray(est['A']).T+np.asarray(est['offset']);pred=np.tanh(raw) if est['mode']=='tanh' else raw;rows.append({'family':family,'transform':'hysteretic_proxy','heldout_mse':float(np.mean((pred-np.asarray(tar))**2)),'mode':est['mode'],'rank':est['rank']})
 agg={n:boot([r['heldout_mse'] for r in rows if r['transform']==n],seed=20261008) for n in sorted({r['transform'] for r in rows})}
 return {'rows':rows,'aggregate':agg,'status':'SCOPE_STRESS_TEST','note':'These observers are deliberately outside the frozen affine/tanh admissible family; degradation constrains scope rather than falsifying the in-family estimator.'}
def _perturb_transfer(a,b):
 def summary(rs):
  d={}
  for r in rs:
   if r['policy']!='balanced' or r['observer']!=0:continue
   x=np.asarray(r.get('causal_paired_difference',[]),float);kind=r['perturbations'][8]['kind'] if r.get('perturbations') and len(r['perturbations'])>8 else 'unknown';d.setdefault(kind,[]).append(float(np.sqrt(np.mean(x*x))) if x.size else 0.)
  return {k:float(np.mean(v)) for k,v in d.items()}
 sa,sb=summary(a),summary(b);keys=sorted(set(sa)&set(sb));corr=float(np.corrcoef([sa[k] for k in keys],[sb[k] for k in keys])[0,1]) if len(keys)>=3 and np.std([sa[k] for k in keys])>0 and np.std([sb[k] for k in keys])>0 else None
 return {'confirm_a':sa,'confirm_b':sb,'common_kinds':keys,'effect_pattern_correlation':corr,'status':'DESCRIPTIVE_TRANSFER'}

def _direct_probe(fleet,bank,split,number,policy,observer_index=0):
    opaque,world,f=bank.episode(split,number,0)
    if observer_index:f=HiddenFilter(f.gain*.8,f.angle+.8,f.lossy,f.nonlinear)
    initial=np.asarray(world.state);u=interventions(policy,12,seed_for(bank.key,split,number,0)%2**32)
    if policy=='adaptive_d_opt':u=interventions('balanced',6,0)
    x=[f.observe((initial+v).tolist()) for v in u];fleet.restore(Brain().checkpoint());rows=fleet.block(x,[Perturbation()]*len(x),operation_id=f'omega-direct/{split}/{number}/{policy}/{observer_index}/probes',reset_each=True);y=[electrode_boundary(r).tolist() for r in rows]
    if policy=='adaptive_d_opt':
        pilot=infer_filter(u,y);working=list(y);batch=[]
        for n in range(6,12):
            nxt=adaptive_next(u,np.asarray(working));u=np.vstack([u,nxt]);batch.append(nxt);pred=np.asarray(pilot['A'])@nxt+pilot['offset'];pred=np.tanh(pred) if pilot['mode']=='tanh' else pred;working.append(pred.tolist())
        fleet.restore(Brain().checkpoint());rr=fleet.block([f.observe((initial+v).tolist()) for v in batch],[Perturbation()]*6,operation_id=f'omega-direct/{split}/{number}/{policy}/{observer_index}/adaptive',reset_each=True);y.extend([electrode_boundary(z).tolist() for z in rr])
    est=infer_filter(u,y);truthA=matrix(f);trueP=np.linalg.pinv(truthA)@truthA;rec=np.asarray(est['world_representative'])
    return {'family':number,'policy':policy,'observer':observer_index,'Fhat':est,'recovered':rec.tolist(),'world_truth':initial.tolist(),'true_projection':trueP.tolist(),'responses':y,'filter_mse':float(np.mean((np.asarray(est['A'])-truthA)**2)),'reconstruction_mse':float(np.mean((trueP@rec-trueP@initial)**2))}

def _collect_omega_split(fleet,root,source_split,key,families):
    bank=Bank(key);dynamic=[];direct=[]
    from reference.rari_neurobrain.experiment import family as full_family
    for number in families:
        d=full_family(fleet,bank,source_split,number,root,'balanced',0,0);dynamic.append(d)
        direct.append({k:d[k] for k in ('family','policy','observer','Fhat','recovered','world_truth','true_projection','responses','filter_mse','reconstruction_mse')})
        for observer,policy in [(1,'balanced'),(0,'passive'),(1,'passive'),(0,'adaptive_d_opt'),(1,'adaptive_d_opt')]:direct.append(_direct_probe(fleet,bank,source_split,number,policy,observer))
    return dynamic,direct

def _direct_contrast(records,metric,ref,cand='balanced'):
    fs=sorted({r['family'] for r in records});v=[];absd=[]
    for f in fs:
        a=np.mean([r[metric] for r in records if r['family']==f and r['policy']==cand]);b=np.mean([r[metric] for r in records if r['family']==f and r['policy']==ref]);v.append((b-a)/max(b,1e-8));absd.append(b-a)
    return np.asarray(v),np.asarray(absd)

def _common_summary(records):
    families=sorted({r['family'] for r in records});rows=[]
    for i,f in enumerate(families):
        a=next(r for r in records if r['family']==f and r['policy']=='balanced' and r['observer']==0);b=next(r for r in records if r['family']==f and r['policy']=='balanced' and r['observer']==1)
        Pa,Pb=np.asarray(a['true_projection']),np.asarray(b['true_projection']);_,sv,vh=np.linalg.svd(np.vstack([np.eye(2)-Pa,np.eye(2)-Pb]));basis=vh[sv<1e-6].T
        if basis.shape[1]==0:rows.append({'family':f,'dimension':0,'status':'NO_COMMON_IDENTIFIABLE_COMPONENT'});continue
        qa=basis.T@np.asarray(a['recovered']);qb=basis.T@np.asarray(b['recovered']);target=basis.T@np.asarray(a['world_truth']);nf=families[(i+1)%len(families)];wrong=next(r for r in records if r['family']==nf and r['policy']=='balanced' and r['observer']==1)
        rows.append({'family':f,'dimension':basis.shape[1],'disagreement':float(np.mean((qa-qb)**2)),'target_mse':float(np.mean(((qa+qb)/2-target)**2)),'wrong_pair_mse':float(np.mean((qa-basis.T@np.asarray(wrong['recovered']))**2)),'raw_disagreement':float(np.mean((np.asarray(a['responses'])[0]-np.asarray(b['responses'])[0])**2))})
    valid=[x for x in rows if x.get('dimension',0)>0]
    return {'eligible':len(valid),'empty':len(rows)-len(valid),'same_mean':float(np.mean([x['disagreement'] for x in valid])) if valid else None,'wrong_mean':float(np.mean([x['wrong_pair_mse'] for x in valid])) if valid else None,'target_mean':float(np.mean([x['target_mse'] for x in valid])) if valid else None,'raw_mean':float(np.mean([x['raw_disagreement'] for x in valid])) if valid else None,'rows':rows}

def _coverage(cal_dynamic,cal_direct,records_dynamic,records_direct,models,selected,scales):
    x,y=dataset(cal_dynamic);cp=predict_models(models,x);radii={}
    for task in TASKS:
        pr=np.asarray(cp[f'{selected}/{task}']);yy=np.asarray(y[task]);res=np.sqrt(np.mean((pr-yy)**2,axis=1));radii[task]=float(conformal_radius(res,.1))
    cfs=sorted({r['family'] for r in cal_direct});direct=np.array([np.sqrt(np.mean([r['reconstruction_mse'] for r in cal_direct if r['family']==f and r['policy']=='balanced'])) for f in cfs]);dr=float(conformal_radius(direct,.1));out={'radii':radii,'direct_radius':dr,'nominal':.9}
    for label,rs in records_dynamic.items():
        xx,yy=dataset(rs);pp=predict_models(models,xx);task_cov={}
        for task in TASKS:
            res=np.sqrt(np.mean((np.asarray(pp[f'{selected}/{task}'])-np.asarray(yy[task]))**2,axis=1));task_cov[task]=float(np.mean(res<=radii[task]))
        ds=records_direct[label];fs=sorted({r['family'] for r in ds});dd=np.array([np.sqrt(np.mean([r['reconstruction_mse'] for r in ds if r['family']==f and r['policy']=='balanced'])) for f in fs]);out[label]={'direct':float(np.mean(dd<=dr)),'tasks':task_cov}
    return out

def run():
 p=protocol();out=ROOT/'results/synthetic';out.mkdir(parents=True,exist_ok=True);key=hashlib.sha256(p['synthetic']['seed_label'].encode()).digest();write(ROOT/'control/SYNTH_SEED_COMMITMENT.json',{'label_sha256':hashlib.sha256(p['synthetic']['seed_label'].encode()).hexdigest(),'key_not_secret':True,'protocol_sha256':sha(PROTOCOL)},replace=False)
 fleet=LocalFleet();dynamic={};direct={}
 for logical in ('development','validation','calibration','confirm_a','confirm_b'):
  sp=p['synthetic']['logical_splits'][logical]
  dynamic[logical],direct[logical]=_collect_omega_split(fleet,ROOT/'synthetic_workspace',sp['source_split'],key,sp['ids'])
  write(out/f'{logical}_collection.json',{'families':len(sp['ids']),'dynamic_records':len(dynamic[logical]),'direct_records':len(direct[logical]),'source_split':sp['source_split'],'family_ids':sp['ids']})
  write(out/f'{logical}_direct_records.json',direct[logical])
 xdev,ydev=dataset(dynamic['development']);models=train_models(xdev,ydev);scales={t:max(float(np.var(ydev[t])),1e-8) for t in TASKS};write(ROOT/'private/SYNTH_MODELS.json',models);write(ROOT/'private/SYNTH_TARGET_SCALES.json',scales)
 pval,yval,lv=_scores(models,dynamic['validation'],scales);selected=min(lv,key=lambda k:(float(np.mean(lv[k])),int(k.split('/')[1]),k.split('/')[0]));write(out/'validation.json',{'selected':selected,'normalized_scores':{k:float(np.mean(v)) for k,v in lv.items()}})
 confirm={};losses={}
 for label in ('confirm_a','confirm_b'):
  pred,y,ll=_scores(models,dynamic[label],scales);confirm[label]={'pred':pred,'targets':y};losses[label]=ll
 primary={}
 for metric,h1,h2 in [('reconstruction_mse','S1_PASSIVE_RECONSTRUCTION','S3_ACTIVE_POLICY_EQUIVALENCE'),('filter_mse','S2_PASSIVE_FILTER','S3_ACTIVE_POLICY_EQUIVALENCE')]:
  primary[metric]={}
  for label in ('confirm_a','confirm_b'):
   v,ad=_direct_contrast(direct[label],metric,'passive');va,ada=_direct_contrast(direct[label],metric,'adaptive_d_opt');primary[metric][label]={'vs_passive_relative':boot(v,seed=20261006),'vs_passive_p_one_sided':signflip(v),'vs_passive_absolute':boot(ad,seed=20261009),'balanced_minus_adaptive_absolute':boot(-ada,q=(.05,.95),seed=20261010),'floor_count_vs_adaptive':int(sum(np.mean([r[metric] for r in direct[label] if r['family']==f and r['policy']=='adaptive_d_opt'])<1e-8 for f in sorted({r['family'] for r in direct[label]})))}
 primary['interpretation']='New prospective Omega banks; parent FINAL-V7 results are not reused as confirmation data.'
 # plateau and phase estimand
 reps={}
 for label in ('confirm_a','confirm_b'):
  l=losses[label];krel=(l['pca/11']-l['pca/12'])/np.maximum(l['pca/12'],1e-8);qrel=(l['classical_phase/12']-l['quantum_phase/12'])/np.maximum(l['classical_phase/12'],1e-8)
  reps[label]={'K11_vs_K12':boot(krel,q=(.05,.95),seed=20261011),'K11_abs':boot(l['pca/11']-l['pca/12'],seed=20261012),'phase_relative_classical_minus_quantum':boot(qrel,seed=20261013),'phase_absolute_classical_minus_quantum':boot(l['classical_phase/12']-l['quantum_phase/12'],seed=20261014),'all_K_mean':{k:float(np.mean(v)) for k,v in l.items()}}
 observer={k:_common_summary(direct[k]) for k in ('confirm_a','confirm_b')};coverage=_coverage(dynamic['calibration'],direct['calibration'],{'confirm_a':dynamic['confirm_a'],'confirm_b':dynamic['confirm_b']},{'confirm_a':direct['confirm_a'],'confirm_b':direct['confirm_b']},models,selected,scales);exc=_excitation(key);oof=_out_of_family(key);pt=_perturb_transfer(dynamic['confirm_a'],dynamic['confirm_b'])
 # null constant response
 u=interventions('balanced',12,0);null_est=infer_filter(u,np.zeros((12,2)));null={'constant_response_rank':null_est['rank'],'constant_J_rank':null_est['parameter_identifiability']['J_P_rank'],'expected_A_rank':0,'status':'PASS' if null_est['rank']==0 else 'FAIL'}
 # status cards
 cards={}
 for metric,hid in [('reconstruction_mse','S1_PASSIVE_RECONSTRUCTION'),('filter_mse','S2_PASSIVE_FILTER')]:
  ok=[]
  for label in ('confirm_a','confirm_b'):
   z=primary[metric][label]['vs_passive_relative'];ok.append(z['mean']>=.10 and z['ci'][0]>0 and primary[metric][label]['vs_passive_p_one_sided']<=.05)
  cards[hid]={'status':'SUPPORT' if all(ok) else 'CHALLENGE_OR_INCONCLUSIVE','both_banks_meet_rule':all(ok)}
 eq=[]
 for metric in ('reconstruction_mse','filter_mse'):
  margin=1e-6;eq.append(all(abs(x)<=margin for lab in ('confirm_a','confirm_b') for x in primary[metric][lab]['balanced_minus_adaptive_absolute']['ci']))
 cards['S3_ACTIVE_POLICY_EQUIVALENCE']={'status':'SUPPORT' if all(eq) else 'INCONCLUSIVE_OR_CHALLENGE','margin_absolute':1e-6}
 eqk=all(all(abs(x)<=.01 for x in reps[lab]['K11_vs_K12']['ci']) for lab in ('confirm_a','confirm_b'));cards['S4_REPRESENTATION_PLATEAU']={'status':'SUPPORT' if eqk else 'INCONCLUSIVE_OR_CHALLENGE','band':.01}
 obsok=all(observer[lab]['eligible']>0 and observer[lab]['same_mean']<observer[lab]['wrong_mean'] and observer[lab]['target_mean']<observer[lab]['wrong_mean'] for lab in observer);cards['S5_COMMON_OBSERVABLE_AGREEMENT']={'status':'SUPPORT' if obsok else 'INCONCLUSIVE_OR_CHALLENGE'}
 covok=all(coverage[lab]['direct']>=.85 for lab in ('confirm_a','confirm_b'));cards['S6_COVERAGE_TRANSFER']={'status':'SUPPORT_MINIMUM_GATE' if covok else 'CHALLENGE_MINIMUM_GATE','nominal':.9,'minimum':.85}
 cards['S7_EXCITATION_SATURATION']={'status':'DESCRIPTIVE_PROSPECTIVE','aggregate':exc['aggregate']};cards['S8_OUT_OF_FAMILY_FILTER_SCOPE']={'status':'SCOPE_CONSTRAINT','aggregate':oof['aggregate']};cards['S9_PAIRED_PERTURBATION_TRANSFER']={'status':'DESCRIPTIVE_PROSPECTIVE','effect_pattern_correlation':pt['effect_pattern_correlation']};cards['S10_NULL_SENSITIVITY']={'status':'SUPPORT' if null['status']=='PASS' else 'CHALLENGE'}
 result={'schema':'rari.omega.synthetic.v1','status':'COMPLETED_ALL_DECLARED_SYNTHETIC_ATTEMPTS','selected_representation':selected,'primary':primary,'representations':reps,'observer':observer,'coverage':coverage,'excitation':exc,'out_of_family':oof,'paired_perturbation':pt,'null':null,'hypothesis_cards':cards,'scientific_scope':'fresh synthetic confirmation only; no biological promotion'}
 write(out/'SYNTHETIC_OMEGA_RESULTS.json',result);return {'results':'results/synthetic/SYNTHETIC_OMEGA_RESULTS.json','sha256':sha(out/'SYNTHETIC_OMEGA_RESULTS.json'),'terminal':True}
if __name__=='__main__':print(run())
