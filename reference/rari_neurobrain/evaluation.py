"""Numerical building blocks. No routine here promotes a scientific hypothesis."""
import numpy as np


def array(x):
    a = np.asarray(x,dtype=float)
    if not np.isfinite(a).all(): raise ValueError("NONFINITE_DATA")
    return a


def jacobian_diagnostics(response, theta, *, step=1e-5, rtol=1e-7):
    theta = array(theta)
    if theta.ndim != 1 or step <= 0 or rtol <= 0: raise ValueError("JACOBIAN_CONFIGURATION")
    columns=[]
    for j in range(len(theta)):
        plus,minus = theta.copy(),theta.copy()
        plus[j]+=step; minus[j]-=step
        columns.append((array(response(plus)).ravel()-array(response(minus)).ravel())/(2*step))
    jac=np.stack(columns,axis=1)
    sv=np.linalg.svd(jac,compute_uv=False)
    threshold=max(jac.shape)*rtol*(sv[0] if len(sv) else 0)
    rank=int(np.sum(sv>threshold))
    condition=float(sv[0]/sv[-1]) if rank==len(theta) and sv[-1]>0 else None
    return {"rank":rank,"parameter_dimension":len(theta),"singular_values":sv.tolist(),
            "condition_number":condition,"rank_tolerance":float(threshold),"difference_step":step,
            "local_full_column_rank":rank==len(theta),"universal_theorem":False,
            "assumptions":"local differentiability; specified response and sampled perturbations only"}


def ridge_fit(x,y,alpha=0.1):
    x,y=array(x),array(y)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or alpha <= 0: raise ValueError("RIDGE_SHAPE")
    if y.ndim==1: y=y[:,None]
    mean=x.mean(axis=0); scale=np.maximum(x.std(axis=0),1e-8)
    z=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.eye(z.shape[1])*alpha;penalty[0,0]=0
    coef=np.linalg.solve(z.T@z+penalty,z.T@y)
    return {"mean":mean,"scale":scale,"coef":coef,"alpha":alpha}


def ridge_predict(model,x):
    x=array(x)
    return np.column_stack([np.ones(len(x)),(x-model["mean"])/model["scale"]])@model["coef"]


def mse(prediction,target):
    a,b=array(prediction),array(target)
    if a.shape!=b.shape: raise ValueError("METRIC_SHAPE_MISMATCH")
    return float(np.mean((a-b)**2))


def k_challenge(train, validation, train_targets, validation_targets, *, alpha=0.1):
    x,v=array(train),array(validation)
    if min(x.shape) < 30 or x.shape[1]!=v.shape[1]: raise ValueError("K_CHALLENGE_NEEDS_30_DIMENSIONS")
    if set(train_targets)!=set(validation_targets): raise ValueError("TARGET_MISMATCH")
    mean=x.mean(0)
    _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
    scores=[]
    # Fixed order and tie rule; no privileged 11. PCA sees development only.
    for k in range(2,31):
        basis=vt[:k].T;z=(x-mean)@basis;zv=(v-mean)@basis
        tasks={}
        for name in sorted(train_targets):
            model=ridge_fit(z,train_targets[name],alpha)
            tasks[name]=mse(ridge_predict(model,zv),validation_targets[name])
        scores.append({"k":k,"tasks":tasks,"representation_reconstruction_mse":mse(zv@basis.T+mean,v)})
    return {"status":"ENGINEERING_FIXTURE_ONLY","candidates":scores,"k11_privileged":False,
            "fit_split":"development","score_split":"validation","scientific_selection":None}


def representation_challenge(x,v,y,yv,k=11,alpha=0.1):
    x,v=array(x),array(v)
    rng=np.random.default_rng(7102)
    projection=rng.normal(size=(x.shape[1],k))/np.sqrt(x.shape[1])
    tx,tv=x@projection,v@projection
    maps={"classical_linear":(tx,tv),"classical_nonlinear":(np.tanh(tx),np.tanh(tv)),
          "classical_random_fourier":(np.cos(tx+0.7),np.cos(tv+0.7)),
          "quantum_inspired_phase":(np.cos(tx)**2,np.cos(tv)**2)}
    rows=[]
    for name,(a,b) in maps.items():
        fit=ridge_fit(a,y,alpha)
        rows.append({"model":name,"mse":mse(ridge_predict(fit,b),yv),
                     "trainable_parameters":int(fit["coef"].size),"fits":1,"features":k,
                     "training_rows":len(x),"regularization":alpha})
    return {"status":"ENGINEERING_FIXTURE_ONLY","rows":rows,"equal_data_and_parameters":True,
            "compute_budget_gate":"NOT_MEASURED: operation/timing budget must be admitted before science",
            "physical_quantum_claim":"WAITING_MEASUREMENTS"}


def conformal_radius(calibration_residuals, alpha=0.1):
    r=array(calibration_residuals)
    if r.ndim!=1 or len(r)==0 or np.any(r<0) or not 0<alpha<1: raise ValueError("CALIBRATION_INPUT")
    rank=int(np.ceil((len(r)+1)*(1-alpha)))
    if rank > len(r): return None  # Unbounded set; no fake finite guarantee.
    return float(np.sort(r)[rank-1])


def holm(pvalues):
    p=array(pvalues)
    if p.ndim!=1 or np.any((p<0)|(p>1)): raise ValueError("PVALUE_RANGE")
    order=np.argsort(p,kind="stable");result=np.empty(len(p));last=0
    for i,j in enumerate(order):
        last=max(last,min(1.0,(len(p)-i)*p[j]));result[j]=last
    return result.tolist()


def family_sign_flip(contrasts, seed=123, repeats=999):
    x=array(contrasts)
    if x.ndim!=2 or x.shape[1]<2 or repeats<1: raise ValueError("FAMILY_CONTRAST_SHAPE")
    rng=np.random.default_rng(seed);observed=x.mean(axis=1);count=np.zeros(len(x))
    for _ in range(repeats):
        shuffled=(x*rng.choice([-1,1],size=x.shape[1])).mean(axis=1)
        count += shuffled >= observed
    raw=((count+1)/(repeats+1)).tolist()
    return {"raw_p":raw,"holm_p":holm(raw),"independent_units":x.shape[1],"repeats":repeats,
            "assumption":"independent symmetric FAMILY contrasts; time samples are not replicates",
            "adaptive_pipeline_null_calibration":"NOT_IMPLEMENTED"}


def cross_observer(observations, target, train_indices, test_indices):
    target=array(target);train=list(train_indices);test=list(test_indices)
    if not train or not test or set(train)&set(test): raise ValueError("OBSERVER_SPLIT_OVERLAP")
    predictions=[];rows=[]
    for observer in observations:
        x=array(observer)
        fit=ridge_fit(x[train],target[train])
        pred=ridge_predict(fit,x[test]);predictions.append(pred)
        wrong=pred[::-1]
        rows.append({"target_mse":mse(pred,target[test]),"wrong_pair_mse":mse(wrong,target[test]),
                     "prediction_variance":float(pred.var(axis=0).mean())})
    disagreement=float(np.var(np.stack(predictions),axis=0).mean())
    return {"status":"ENGINEERING_SUPERVISED_DECODER_FIXTURE_ONLY","observers":rows,
            "recovered_disagreement":disagreement,"independent_filter_inference":"NOT_IMPLEMENTED",
            "limitation":"supervised development Q(W) decoding is not perturbational filter discovery"}
