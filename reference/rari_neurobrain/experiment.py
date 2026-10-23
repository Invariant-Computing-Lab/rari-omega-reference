"""Controller-side experiment generator. Raw W/F never enter a parcel or role.
Scientific units are independently seeded world families, never workers/ticks.
"""
import dataclasses,json,math,time
from pathlib import Path
import numpy as np
from .records import write,read,digest
from .world import Bank,HiddenFilter,seed_for
from .subject import Brain,Perturbation,PERTURBATIONS
from .science import infer_filter,interventions,adaptive_next,features,TASKS,POLICIES


def electrode_boundary(rows):
    # Reset-assay dynamics: integration=tanh(.2*filtered_input). Invert known sensor response.
    return (np.arctanh(np.clip(np.asarray([r['electrodes'] for r in rows]),-.999999,.999999))/.2).mean(0)


def observation_array(rows):
    return [[*r['electrodes'],r['spike_count'],r['action']] for r in rows]


def matrix(f):
    c,s=math.cos(f.angle),math.sin(f.angle)
    a=f.gain*np.array([[c,-s],[s,c]])
    if f.lossy:a[1]=0
    return a


def family(fleet,bank,split,number,root,policy,observer_index=0,kimi_action=0):
    root=Path(root);out=root/'private/banks'/split/f'{number:03d}_{policy}_{observer_index}'
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    opaque,world,f=bank.episode(split,number,0)
    if observer_index:f=HiddenFilter(f.gain*.8,f.angle+.8,f.lossy,f.nonlinear)
    initial=np.asarray(world.state);u=interventions(policy,12,seed_for(bank.key,split,number,0)%2**32)
    # Every policy receives the same number of reset-assay observations and known interventions.
    if policy=='adaptive_d_opt':u=interventions('balanced',6,0)
    x=[f.observe((initial+v).tolist()) for v in u]
    fleet.restore(Brain().checkpoint())
    rows=fleet.block(x,[Perturbation()]*len(x),operation_id=f'{split}/{number}/{policy}/{observer_index}/probes',reset_each=True)
    y=[electrode_boundary(r).tolist() for r in rows]
    if policy=='adaptive_d_opt':
        # Two-stage response-adaptive D-optimal design: six measured pilots,
        # then a six-probe batch optimized under the fitted response/Jacobian.
        pilot=infer_filter(u,y);working_y=list(y);batch=[]
        for n in range(6,12):
            next_u=adaptive_next(u,np.asarray(working_y));u=np.vstack([u,next_u]);batch.append(next_u)
            predicted=np.asarray(pilot['A'])@next_u+pilot['offset']
            if pilot['mode']=='tanh':predicted=np.tanh(predicted)
            working_y.append(predicted.tolist())
        fleet.restore(Brain().checkpoint())
        r=fleet.block([f.observe((initial+v).tolist()) for v in batch],[Perturbation()]*6,
            operation_id=f'{split}/{number}/{policy}/{observer_index}/adaptive_batch',reset_each=True)
        y.extend([electrode_boundary(item).tolist() for item in r])
    estimate=infer_filter(u,y)
    recovered=np.asarray(estimate['world_representative']);P=np.asarray(estimate['observable_projection'])
    # Truth chooses scoring target only after predictions are durable. Store privately now.
    truthA=matrix(f);trueP=np.linalg.pinv(truthA)@truthA
    row={'opaque_id':opaque+f'-o{observer_index}-{policy}','family':number,'policy':policy,'observer':observer_index,
         'Fhat':estimate,'recovered':recovered.tolist(),'u':u.tolist(),'responses':y}
    write(out/'PREDICTION.json',row)
    # The dynamic assay resets the brain/world and uses a common frozen cognitive
    # action. It is exactly policy-independent, so each observer's measured
    # balanced trajectory is shared by all policy comparisons without re-execution.
    baseline=root/'private/banks'/split/f'{number:03d}_balanced_{observer_index}'/'COMPLETE.json'
    if policy!='balanced' and baseline.exists():
        from .records import file_hash
        shared=read(baseline)
        record={**shared,**row,'trajectory_shared_reference_sha256':file_hash(baseline),
                'filter_mse':float(np.mean((np.asarray(estimate['A'])-truthA)**2)),
                'reconstruction_mse':float(np.mean((trueP@recovered-trueP@initial)**2))}
        write(out/'COMPLETE.json',record);return record
    # Free-running, coupled dynamic trajectory with perturbations and eight-tick tract delay.
    fleet.restore(Brain().checkpoint());trajectory=[];action=float(kimi_action);perturbations=[];truths=[];causal=[];events=[]
    for block in range(4):
        channels=[];pts=[]
        for t in range(8):
            w=world.step(action);truths.append(w);channels.append(f.observe(w))
            kind=PERTURBATIONS[1+(number%12)] if block==1 else 'sham'
            pt=Perturbation(kind,1+number%101,.5,8,8) if kind!='sham' else Perturbation()
            pts.append(pt);perturbations.append(dataclasses.asdict(pt))
        if block==1:
            before=fleet.checkpoint()
            sham=fleet.block(channels,[Perturbation()]*8,operation_id=f'{split}/{number}/{policy}/{observer_index}/paired_sham')
            fleet.restore(before)
        observed=fleet.block(channels,pts,operation_id=f'{split}/{number}/{policy}/{observer_index}/trajectory/{block}')
        events.extend(getattr(fleet,'last_events',[]))
        if block==1:
            causal=(np.asarray([observation_array(r) for r in observed])-np.asarray([observation_array(r) for r in sham])).tolist()
        trajectory.extend([observation_array(r) for r in observed]);action=float(np.clip(np.mean([r['action'] for r in observed[-1]])+.1*kimi_action,-1,1))
    tr=np.asarray(trajectory);checkpoint=fleet.checkpoint()
    targets={'reconstruction':(trueP@initial).tolist(),
             'hidden_filter_identification':truthA.ravel().tolist(),
             'perturbation_prediction':tr[8:16,:,0].mean(0)[:16].tolist(),
             'action_prediction':[float(tr[-1,:,3].mean())],
             'cross_world_generalization':(trueP@np.asarray(truths[-1])).tolist(),
             'recovery_attractor':[float(np.linalg.norm(tr[-1,:,:2]-tr[7,:,:2])),float(tr[-8:,:,2].mean())]}
    # Task features are pre-perturbation history only: no target time leakage.
    record={**row,'features':features(tr[:8]),'targets':targets,'trajectory':trajectory,
            'world_truth':initial.tolist(),'filter_truth':{'A':truthA.tolist(),'nonlinear':f.nonlinear},
            'spike_events':events,'causal_paired_difference':causal,'true_projection':trueP.tolist(),'perturbations':perturbations,'checkpoint':checkpoint,
            'filter_mse':float(np.mean((np.asarray(estimate['A'])-truthA)**2)),
            'reconstruction_mse':float(np.mean((trueP@recovered-trueP@initial)**2)),
            'claim_scope':'synthetic two-dimensional filter; known reset-vMEA sensor; no biological validation'}
    write(out/'COMPLETE.json',record);return record


def collect(fleet,root,split,key,families,kimi_action=0):
    records=[];bank=Bank(key)
    for number in families:
        for observer in [0,1]:
            for policy in ['balanced','passive','random','adaptive_d_opt']:
                records.append(family(fleet,bank,split,number,root,policy,observer,kimi_action))
    return records


def dataset(records,policy='balanced',observer=0):
    rs=[r for r in records if r['policy']==policy and r['observer']==observer]
    return np.asarray([r['features'] for r in rs]),{t:np.asarray([r['targets'][t] for r in rs]) for t in TASKS}


def development_summary(records):
    return {'status':'MEASURED_SYNTHETIC_DEVELOPMENT','families':len({r['family'] for r in records}),
            'policies':{p:{'filter_mse':float(np.mean([r['filter_mse'] for r in records if r['policy']==p])),
                           'reconstruction_mse':float(np.mean([r['reconstruction_mse'] for r in records if r['policy']==p]))} for p in POLICIES},
            'units':'world families; parcels and timepoints are not independent replicates',
            'Kimi_is_subject':True,'truth_exported':False}


def cross_observer_analysis(records):
    scores=[];families=sorted({r['family'] for r in records})
    for fi,f in enumerate(families):
        rs=[r for r in records if r['family']==f and r['policy']=='balanced']
        a,b=sorted(rs,key=lambda r:r['observer'])
        # Shared target is the intersection of observable subspaces; report empty intersections.
        Pa,Pb=np.asarray(a['true_projection']),np.asarray(b['true_projection'])
        _,sv,vh=np.linalg.svd(np.vstack([np.eye(2)-Pa,np.eye(2)-Pb]));basis=vh[sv<1e-6].T
        if basis.shape[1]==0:scores.append({'family':f,'status':'NO_COMMON_IDENTIFIABLE_COMPONENT','dimension':0});continue
        qa=basis.T@np.asarray(a['recovered']);qb=basis.T@np.asarray(b['recovered']);target=basis.T@np.asarray(a['world_truth'])
        scores.append({'family':f,'dimension':basis.shape[1],'disagreement':float(np.mean((qa-qb)**2)),
                       'target_mse':float(np.mean(((qa+qb)/2-target)**2)),
                       'wrong_pair_mse':float(np.mean((qa-basis.T@np.asarray(next(r['recovered'] for r in records if r['family']==families[(fi+1)%len(families)] and r['observer']==1 and r['policy']=='balanced')))**2)),
                       'raw_disagreement':float(np.mean((np.asarray(a['responses'])[0]-np.asarray(b['responses'])[0])**2)),
                       'truth_used_for_filter_inference':False})
    return {'status':'LOCKED_SYNTHETIC_ANALYSIS','families':scores,'independent_filter_inference':True,
            'scope':'common identifiable quotient only; no claims about unobservable directions'}
