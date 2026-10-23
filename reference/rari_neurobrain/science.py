"""Preregisterable perturbational estimators. No provider or private-bank access.
A filter is inferred only from intervention coordinates and measured responses.
"""
import numpy as np
from .evaluation import ridge_fit,ridge_predict,mse,holm,conformal_radius
from .records import digest

TASKS=('reconstruction','hidden_filter_identification','perturbation_prediction',
       'action_prediction','cross_world_generalization','recovery_attractor')
POLICIES=('passive','random','balanced','adaptive_d_opt')


def infer_filter(u,y,*,alpha=1e-8):
    u=np.asarray(u,float);y=np.asarray(y,float)
    if u.ndim!=2 or u.shape[1]!=2 or y.shape!=u.shape or len(u)<6:raise ValueError('FILTER_PROBE_SHAPE')
    designs=np.column_stack([np.ones(len(u)),u]);train=np.arange(len(u))%3!=2;val=~train
    candidates=[]
    for mode in ('linear','tanh'):
        if mode=='tanh' and np.any(np.abs(y)>=1):continue
        z=y if mode=='linear' else np.arctanh(np.clip(y,-1+1e-10,1-1e-10))
        coef=np.linalg.solve(designs[train].T@designs[train]+np.eye(3)*alpha,designs[train].T@z[train])
        pred=designs[val]@coef
        if mode=='tanh':pred=np.tanh(pred)
        candidates.append((mse(pred,y[val]),mode,z))
    _,mode,z=min(candidates,key=lambda v:(v[0],v[1]))
    coef=np.linalg.solve(designs.T@designs+np.eye(3)*alpha,designs.T@z)
    A=coef[1:].T;b=coef[0];sv=np.linalg.svd(A,compute_uv=False)
    rank=int(np.sum(sv>max(1e-6,sv[0]*1e-5)))
    pinv=np.linalg.pinv(A,rcond=1e-5);projection=pinv@A;world=pinv@b
    residual=z-designs@coef;sigma=residual.T@residual/max(1,len(u)-3)
    covariance=np.kron(np.linalg.pinv(designs.T@designs),sigma)
    J=np.kron(designs,np.eye(2))
    if mode=='tanh':J*=((1-np.tanh(designs@coef)**2).ravel()[:,None])
    jsv=np.linalg.svd(J,compute_uv=False);jrank=int(np.sum(jsv>max(J.shape)*1e-7*jsv[0]))
    return {'parameter_identifiability':{'J_P_rank':jrank,'theta_dimension':6,'singular_values':jsv.tolist(),'condition':float(jsv[0]/jsv[-1]) if jrank==6 else None,'assumptions':'local affine or tanh response, parameters are A and offset; distinct from world invertibility','unbounded_parameter_directions':6-jrank},'A':A.tolist(),'offset':b.tolist(),'mode':mode,'rank':rank,
            'singular_values':sv.tolist(),'condition':float(sv[0]/sv[-1]) if rank==2 else None,
            'world_representative':world.tolist(),'observable_projection':projection.tolist(),
            'nullspace_projector':(np.eye(2)-projection).tolist(),'parameter_covariance':covariance.tolist(),'theta_order':['b0','b1','A00','A10','A01','A11'],'covariance_assumption':'local homoskedastic residual approximation; singular directions explicitly unbounded',
            'uncertainty_unbounded_in_nullspace':rank<2,'equivalence_class':'world_representative + null(A)',
            'fit_inputs_sha256':digest({'u':u.tolist(),'observations':y.tolist()}),
            'truth_used_for_fit':False,'probe_validation_mse':float(min(x[0] for x in candidates))}


def interventions(policy,count,seed,*,responses=None):
    rng=np.random.default_rng(seed)
    if policy=='passive':return np.zeros((count,2))
    if policy=='random':return rng.uniform(-.5,.5,(count,2))
    candidates=np.array([[0,0],[.5,0],[-.5,0],[0,.5],[0,-.5],[.35,.35],[-.35,.35]])
    if policy=='balanced':return candidates[np.arange(count)%len(candidates)]
    if policy!='adaptive_d_opt':raise ValueError('POLICY')
    # Sequential D-optimal excitation with response-weighted information; same probe count.
    information=np.eye(3)*1e-3;chosen=[]
    for t in range(count):
        weights=np.ones(len(candidates))
        if responses is not None and len(responses)>=6:
            model=infer_filter(np.asarray(chosen[:len(responses)]),responses)
            if model['mode']=='tanh':
                predicted=candidates@np.asarray(model['A']).T+model['offset']
                weights=np.maximum(.05,np.mean((1-np.tanh(predicted)**2)**2,axis=1))
        d=np.column_stack([np.ones(len(candidates)),candidates]);inv=np.linalg.inv(information)
        gain=np.log1p(np.einsum('ij,jk,ik->i',d,inv,d)*weights)
        index=int(np.argmax(gain));chosen.append(candidates[index]);information+=weights[index]*np.outer(d[index],d[index])
    return np.asarray(chosen)


def adaptive_next(u,y):
    candidates=np.array([[0,0],[.5,0],[-.5,0],[0,.5],[0,-.5],[.35,.35],[-.35,.35]])
    d=np.column_stack([np.ones(len(u)),u]);info=d.T@d+np.eye(3)*1e-3
    model=infer_filter(u,y);v=np.column_stack([np.ones(len(candidates)),candidates])
    weights=np.ones(len(candidates))
    if model['mode']=='tanh':
        weights=np.maximum(.05,np.mean((1-np.tanh(candidates@np.asarray(model['A']).T+model['offset'])**2)**2,axis=1))
    score=np.einsum('ij,jk,ik->i',v,np.linalg.inv(info),v)*weights
    return candidates[int(np.argmax(score))].tolist()


def features(trajectory):
    x=np.asarray(trajectory,float)
    # N4-derived vMEA history: mean/std/last observation, no world/filter truth.
    return np.concatenate([x.mean(0).ravel(),x.std(0).ravel(),x[-1].ravel()]).tolist()


def train_models(x,targets,*,alpha=.1):
    x=np.asarray(x,float);mu=x.mean(0);_,_,vt=np.linalg.svd(x-mu,full_matrices=False)
    if min(x.shape)<30:raise ValueError('K_REQUIRES_30_TRAIN_ROWS')
    models={};rng=np.random.default_rng(7102);projection=rng.normal(size=(x.shape[1],30))/np.sqrt(x.shape[1])
    # Equal feature width, same projection, nonlinear-op budget, fits, targets and ridge penalty.
    z=x@projection
    classical=np.cos(z+.7)**2;quantum=np.cos(z+0.)**2
    for k in range(2,31):
        for name,Z in [('pca',(x-mu)@vt[:30].T),('classical_phase',classical),('quantum_phase',quantum)]:
            for task in TASKS:
                model=ridge_fit(Z[:,:k],targets[task],alpha)
                models[f'{name}/{k}/{task}']={key:(v.tolist() if isinstance(v,np.ndarray) else v) for key,v in model.items()}
    return {'mean':mu.tolist(),'basis':vt[:30].T.tolist(),'projection':projection.tolist(),'models':models,
            'feature_formula_classical':'cos(XP+0.7)^2','feature_formula_quantum':'cos(XP)^2',
            'compute_per_map':{'matmul_multiply_adds':int(x.size*30),'cos_calls':len(x)*30,'squares':len(x)*30,'adds_including_padding':len(x)*30},
            'fit_count_per_family':29*6,'regularization':alpha,'k11_privileged':False}


def predict_models(bank,x):
    x=np.asarray(x,float);z=x@np.asarray(bank['projection'])
    maps={'pca':(x-bank['mean'])@np.asarray(bank['basis']),
          'classical_phase':np.cos(z+.7)**2,'quantum_phase':np.cos(z+0.)**2}
    output={}
    for key,raw in bank['models'].items():
        name,k,task=key.split('/');model={a:np.asarray(v) if isinstance(v,list) else v for a,v in raw.items()}
        output[key]=ridge_predict(model,maps[name][:,:int(k)]).tolist()
    return output


def score_predictions(predictions,targets):
    return {k:mse(v,targets[k.split('/')[-1]]) for k,v in predictions.items()}


def uncertainty(contrasts,seed=9921):
    a=np.asarray(contrasts,float);rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(a,size=len(a),replace=True).mean() for _ in range(1999)])
    return {'mean':float(a.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'independent_units':len(a),'resampling_unit':'world_family','seed':seed}


def multiplicity(contrasts,seed=123,repeats=1999):
    a=np.asarray(contrasts,float);rng=np.random.default_rng(seed);observed=a.mean(1);count=np.zeros(len(a))
    for _ in range(repeats):count+=(a*rng.choice([-1,1],size=a.shape[1])).mean(1)>=observed
    p=((count+1)/(repeats+1)).tolist()
    return {'raw_p':p,'holm_p':holm(p),'family_count':a.shape[1],'repeats':repeats,
            'assumption':'independent symmetric paired world-family contrasts'}
