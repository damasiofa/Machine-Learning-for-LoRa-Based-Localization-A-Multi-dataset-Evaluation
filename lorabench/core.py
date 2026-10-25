"""Main V5 equal-footing benchmark: models, hyperparameter grids, partitions, evaluation,
checkpoints, ablation, per-scenario aggregation, and the 1:1:1:1 deployment synthesis.

The numerical code is unchanged from run_equal_footing_benchmark.py (V5). The only changes are the
package imports, an explicit protocol_version argument (used by the extensions instead of
overwriting a module global), the thread pools recorded in system_metadata.json, and run(args) in
place of main(). The thread limit itself is applied by the command line (lorabench/threads.py).
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .threads import pools as thread_pools
from .data import (
    ALL_DATASETS, PRIMARY_SCENARIOS, DEPLOYMENTS, SCENARIOS_BY_DEPLOYMENT,
    FeatureStore, build_dataset, build_many, expand_dataset_tokens, deployment_for_scenario,
)

PROTOCOL_VERSION = "MULTIDATASET_EQUAL_FOOTING_20SEEDS_V5"
METHODS = ["kNN", "WkNN", "RF", "XGBoost", "MLP", "ELM"]
MAIN_VARIANTS = ["RSSI_ONLY", "COMMON_FULL"]
SEEDS_20 = [7,13,42,73,101,256,314,512,1001,1729,2026,2718,4096,8191,12345,16381,32749,65537,99991,104729]
EARTH_RADIUS_M = 6371000.0


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def parse_dataset_list(value: str) -> List[str]:
    return expand_dataset_tokens(value)

def grid(profile: str, method: str) -> List[Dict[str,Any]]:
    if profile == "quick":
        if method in ("kNN","WkNN"):
            return [{"metric":m,"k_values":[5,10,15,20]} for m in ["manhattan","euclidean"]]
        if method=="RF":
            return [{"n_estimators":150,"max_depth":15,"min_samples_leaf":1},{"n_estimators":250,"max_depth":None,"min_samples_leaf":2}]
        if method=="XGBoost":
            return [{"n_estimators":200,"max_depth":4,"learning_rate":0.10},{"n_estimators":350,"max_depth":6,"learning_rate":0.05},{"n_estimators":500,"max_depth":8,"learning_rate":0.03}]
        if method=="MLP":
            return [{"hidden_layer_sizes":(128,64),"activation":"relu","alpha":1e-4,"learning_rate_init":1e-3},{"hidden_layer_sizes":(128,128,64),"activation":"relu","alpha":1e-3,"learning_rate_init":5e-4}]
        if method=="ELM":
            return [{"n_hidden":500,"activation":"tanh","alpha":1e-3},{"n_hidden":1000,"activation":"tanh","alpha":1e-3},{"n_hidden":1000,"activation":"relu","alpha":1e-3},{"n_hidden":1000,"activation":"sigmoid","alpha":1e-3}]
    if method in ("kNN","WkNN"):
        ks=list(range(1,21))+[25,30,35,40,50]
        return [{"metric":m,"k_values":ks} for m in ["manhattan","euclidean","cosine"]]
    if method=="RF":
        return [
            {"n_estimators":200,"max_depth":15,"min_samples_leaf":1},
            {"n_estimators":300,"max_depth":20,"min_samples_leaf":1},
            {"n_estimators":400,"max_depth":25,"min_samples_leaf":1},
            {"n_estimators":400,"max_depth":None,"min_samples_leaf":2},
            {"n_estimators":500,"max_depth":None,"min_samples_leaf":4},
        ]
    if method=="XGBoost":
        return [
            {"n_estimators":200,"max_depth":4,"learning_rate":0.10},
            {"n_estimators":300,"max_depth":4,"learning_rate":0.05},
            {"n_estimators":500,"max_depth":4,"learning_rate":0.03},
            {"n_estimators":200,"max_depth":6,"learning_rate":0.10},
            {"n_estimators":300,"max_depth":6,"learning_rate":0.05},
            {"n_estimators":500,"max_depth":6,"learning_rate":0.03},
            {"n_estimators":700,"max_depth":6,"learning_rate":0.02},
            {"n_estimators":300,"max_depth":8,"learning_rate":0.05},
            {"n_estimators":500,"max_depth":8,"learning_rate":0.03},
            {"n_estimators":700,"max_depth":8,"learning_rate":0.02},
            {"n_estimators":300,"max_depth":10,"learning_rate":0.05},
            {"n_estimators":500,"max_depth":10,"learning_rate":0.03},
            {"n_estimators":700,"max_depth":10,"learning_rate":0.02},
        ]
    if method=="MLP":
        return [
            {"hidden_layer_sizes":(128,64),"activation":"relu","alpha":1e-4,"learning_rate_init":1e-3},
            {"hidden_layer_sizes":(256,128),"activation":"relu","alpha":1e-4,"learning_rate_init":5e-4},
            {"hidden_layer_sizes":(128,128,64),"activation":"relu","alpha":1e-3,"learning_rate_init":5e-4},
            {"hidden_layer_sizes":(256,256,128),"activation":"tanh","alpha":1e-3,"learning_rate_init":3e-4},
        ]
    if method=="ELM":
        return [
            *[{"n_hidden":h,"activation":"tanh","alpha":1e-3} for h in [200,300,500,750,1000,1500,2000,2500,3000]],
            *[{"n_hidden":h,"activation":"relu","alpha":1e-3} for h in [500,1000,1500,2000]],
            *[{"n_hidden":h,"activation":"sigmoid","alpha":1e-3} for h in [500,1000,1500,2000]],
        ]
    raise ValueError(method)


class ELMRegressor:
    def __init__(self,n_hidden=1000,activation="tanh",alpha=1e-3,random_state=42):
        self.n_hidden=int(n_hidden); self.activation=activation; self.alpha=float(alpha); self.random_state=int(random_state)
    def _act(self,z):
        if self.activation=="tanh": return np.tanh(z)
        if self.activation=="relu": return np.maximum(z,0)
        if self.activation=="sigmoid":
            z=np.clip(z,-30,30); return 1.0/(1.0+np.exp(-z))
        raise ValueError(self.activation)
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float64)
        rng=np.random.default_rng(self.random_state)
        self.W_=rng.normal(0,1,size=(X.shape[1],self.n_hidden)).astype(np.float32)/max(1.0,np.sqrt(X.shape[1]))
        self.b_=rng.normal(0,1,size=self.n_hidden).astype(np.float32)
        H=self._act(X@self.W_+self.b_).astype(np.float32,copy=False)
        self.out_=Ridge(alpha=self.alpha,fit_intercept=True,solver="lsqr",tol=1e-4)
        self.out_.fit(H,y)
        del H; gc.collect(); return self
    def predict(self,X):
        X=np.asarray(X,dtype=np.float32)
        H=self._act(X@self.W_+self.b_).astype(np.float32,copy=False)
        p=self.out_.predict(H)
        del H
        return p


def build_model(method: str, params: Dict[str,Any], threads: int):
    if method=="RF":
        return RandomForestRegressor(**params,n_jobs=threads,random_state=42,max_features="sqrt")
    if method=="XGBoost":
        base=XGBRegressor(**params,subsample=0.9,colsample_bytree=0.9,objective="reg:squarederror",tree_method="hist",n_jobs=threads,random_state=42,verbosity=0)
        return MultiOutputRegressor(base,n_jobs=1)
    if method=="MLP":
        return MLPRegressor(**params,max_iter=300,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,random_state=42,batch_size=256)
    if method=="ELM":
        return ELMRegressor(**params,random_state=42)
    raise ValueError(method)


def error_vector(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> np.ndarray:
    y_true=np.asarray(y_true,dtype=np.float64); y_pred=np.asarray(y_pred,dtype=np.float64)
    if y_pred.ndim==1: y_pred=y_pred[:,None]
    if metric=="haversine_m":
        lat1=np.radians(y_true[:,0]); lon1=np.radians(y_true[:,1]); lat2=np.radians(y_pred[:,0]); lon2=np.radians(y_pred[:,1])
        dlat=lat2-lat1; dlon=lon2-lon1
        a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return (2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(a,0,1)))).astype(np.float64)
    if metric=="euclidean_m": return np.linalg.norm(y_true-y_pred,axis=1).astype(np.float64)
    if metric=="absolute_error_m": return np.abs(y_true[:,0]-y_pred[:,0]).astype(np.float64)
    raise ValueError(metric)


def summarize_errors(e: np.ndarray) -> Dict[str,float]:
    e=np.asarray(e,dtype=np.float64)
    out={
        "mean_error_m":float(np.mean(e)),"median_error_m":float(np.median(e)),"std_error_m":float(np.std(e)),
        "p75_error_m":float(np.percentile(e,75)),"p90_error_m":float(np.percentile(e,90)),"p95_error_m":float(np.percentile(e,95)),"p99_error_m":float(np.percentile(e,99)),"max_error_m":float(np.max(e)),
    }
    for t in [5,10,25,50,100,250,500,1000]: out[f"coverage_lt_{t}m_pct"]=float(100*np.mean(e<t))
    out["robust_score"]=0.50*out["mean_error_m"]+0.25*out["median_error_m"]+0.25*out["p90_error_m"]
    return out


def split_random(indices: np.ndarray, seed: int) -> Dict[str,np.ndarray]:
    tr,tmp=train_test_split(indices,test_size=0.20,random_state=seed,shuffle=True)
    va,te=train_test_split(tmp,test_size=0.50,random_state=seed,shuffle=True)
    return {"train":np.sort(tr),"val":np.sort(va),"test":np.sort(te)}


def split_spatial(indices: np.ndarray, groups_all: np.ndarray, seed: int) -> Dict[str,np.ndarray]:
    """Whole-block 80/10/10 holdout with the same block-count rule in every dataset."""
    g=np.asarray(groups_all[indices]); ug=np.unique(g)
    if len(ug)<3: raise ValueError("Not enough spatial groups for holdout")
    rng=np.random.default_rng(seed); shuffled=np.asarray(ug).copy(); rng.shuffle(shuffled)
    n=len(shuffled)
    n_test=max(1,int(round(0.10*n))); n_val=max(1,int(round(0.10*n)))
    if n_test+n_val>=n: n_test=n_val=1
    test_groups=shuffled[:n_test]; val_groups=shuffled[n_test:n_test+n_val]; train_groups=shuffled[n_test+n_val:]
    out={
        "train":np.sort(indices[np.isin(g,train_groups)]),
        "val":np.sort(indices[np.isin(g,val_groups)]),
        "test":np.sort(indices[np.isin(g,test_groups)]),
        "group_train":np.sort(train_groups),"group_val":np.sort(val_groups),"group_test":np.sort(test_groups),
    }
    return out

def deterministic_subset(store: FeatureStore, dataset_id: str, max_samples: Optional[int]) -> np.ndarray:
    n=len(store.y)
    if not max_samples or max_samples>=n: return np.arange(n,dtype=np.int64)
    rng=np.random.default_rng(20260830)
    return np.sort(rng.choice(n,size=max_samples,replace=False)).astype(np.int64)

def predict_neighbors(nn, X, yref, k:int, weighted:bool, batch:int=512):
    preds=[]
    for start in range(0,len(X),batch):
        dist,ind=nn.kneighbors(X[start:start+batch],n_neighbors=k,return_distance=True)
        yy=yref[ind]
        if weighted:
            w=1.0/np.maximum(dist,1e-9); p=(yy*w[:,:,None]).sum(axis=1)/w.sum(axis=1)[:,None]
        else: p=yy.mean(axis=1)
        preds.append(p)
    return np.vstack(preds)


def search_neighbors(method,Xtr,ytr,Xv,yv,configs,threads,x_scaler,preprocess_fit_s,metric_name):
    weighted=method=="WkNN"; best=None; best_nn=None; rows=[]; selected_fit=np.nan
    t0=time.perf_counter()
    for cfg in configs:
        metric=cfg["metric"]; ks=[k for k in cfg["k_values"] if k<=len(Xtr)]
        if not ks: continue
        maxk=max(ks); tf=time.perf_counter(); nn=NearestNeighbors(n_neighbors=maxk,metric=metric,algorithm="brute",n_jobs=threads); nn.fit(Xtr); fit_s=time.perf_counter()-tf
        sums={k:0.0 for k in ks}; count=0
        for start in range(0,len(Xv),512):
            xb=Xv[start:start+512]; yb=yv[start:start+512]; dist,ind=nn.kneighbors(xb,n_neighbors=maxk,return_distance=True); yy=ytr[ind]
            for k in ks:
                if weighted:
                    w=1/np.maximum(dist[:,:k],1e-9); pred=(yy[:,:k,:]*w[:,:,None]).sum(1)/w.sum(1)[:,None]
                else: pred=yy[:,:k,:].mean(1)
                sums[k]+=float(error_vector(yb,pred,metric_name).sum())
            count+=len(xb)
        for k in ks:
            score=sums[k]/count; rows.append({"metric":metric,"k":k,"mean_error_m":score,"fit_time_s":fit_s})
            if best is None or score<best[0]:
                if best_nn is not None: del best_nn
                best=(score,{"metric":metric,"k":int(k)}); best_nn=nn; selected_fit=fit_s
        if best_nn is not nn: del nn
        gc.collect()
    if best is None: raise RuntimeError("No valid neighbor candidate")
    bundle={"model":best_nn,"params":best[1],"x_scaler":x_scaler,"y_scaler":None,"y_ref":ytr,"selected_model_fit_time_s":float(selected_fit),"preprocess_fit_time_s":float(preprocess_fit_s),"reference_bytes":int(best_nn._fit_X.nbytes+ytr.nbytes)}
    return best[1],rows,time.perf_counter()-t0,bundle


def candidate_search(method,Xtr,ytr,Xv,yv,configs,threads,metric_name):
    preprocess_fit_s=0.0
    if method in ("kNN","WkNN"):
        t=time.perf_counter(); xs=StandardScaler(copy=False); Xtr2=np.asarray(xs.fit_transform(Xtr),dtype=np.float32); Xv2=np.asarray(xs.transform(Xv),dtype=np.float32); preprocess_fit_s=time.perf_counter()-t
        return search_neighbors(method,Xtr2,ytr,Xv2,yv,configs,threads,xs,preprocess_fit_s,metric_name)
    needs_scale=method in ("MLP","ELM"); xs=ys=None
    if needs_scale:
        t=time.perf_counter(); xs=StandardScaler(copy=False); Xtr2=np.asarray(xs.fit_transform(Xtr),dtype=np.float32); Xv2=np.asarray(xs.transform(Xv),dtype=np.float32); ys=StandardScaler(); ytr2=ys.fit_transform(ytr); preprocess_fit_s=time.perf_counter()-t
    else: Xtr2,Xv2,ytr2=Xtr,Xv,ytr
    best=None; best_model=None; rows=[]; selected_fit=np.nan; t0=time.perf_counter()
    for i,params in enumerate(configs):
        gc.collect(); tf=time.perf_counter(); model=build_model(method,params,threads)
        fit_target=ytr2.ravel() if (np.ndim(ytr2)==2 and ytr2.shape[1]==1 and method in ("RF","MLP")) else ytr2
        model.fit(Xtr2,fit_target); fit_s=time.perf_counter()-tf
        tp=time.perf_counter(); pred=model.predict(Xv2); pred_s=time.perf_counter()-tp
        pred=np.asarray(pred); pred=pred[:,None] if pred.ndim==1 else pred
        if ys is not None: pred=ys.inverse_transform(pred)
        met=summarize_errors(error_vector(yv,pred,metric_name)); rows.append({"config_index":i,"params":json.dumps(params,sort_keys=True),"fit_time_s":fit_s,"validation_predict_time_s":pred_s,**met})
        if best is None or met["mean_error_m"]<best[0]:
            if best_model is not None: del best_model
            best=(met["mean_error_m"],dict(params)); best_model=model; selected_fit=fit_s
        else: del model
        del pred; gc.collect()
    bundle={"model":best_model,"params":best[1],"x_scaler":xs,"y_scaler":ys,"y_ref":None,"selected_model_fit_time_s":float(selected_fit),"preprocess_fit_time_s":float(preprocess_fit_s),"reference_bytes":None}
    return best[1],rows,time.perf_counter()-t0,bundle


def model_size_bytes(method: str, selected: Dict[str,Any]) -> int:
    if method in ("kNN","WkNN"):
        size=int(selected.get("reference_bytes") or 0)
        xs=selected.get("x_scaler")
        if xs is not None:
            for a in [getattr(xs,"mean_",None),getattr(xs,"scale_",None),getattr(xs,"var_",None)]:
                if a is not None: size+=np.asarray(a).nbytes
        return size
    try: return len(pickle.dumps((selected["model"],selected.get("x_scaler"),selected.get("y_scaler")),protocol=pickle.HIGHEST_PROTOCOL))
    except Exception: return -1


def evaluate_selected(method,selected,Xtest_raw,ytest,metric_name,inference_reps):
    xs=selected.get("x_scaler"); ys=selected.get("y_scaler"); model=selected["model"]; params=selected["params"]
    tp=time.perf_counter(); Xtest=xs.transform(Xtest_raw) if xs is not None else Xtest_raw; Xtest=np.asarray(Xtest,dtype=np.float32) if method in ("kNN","WkNN","MLP","ELM") else Xtest; preprocess_s=time.perf_counter()-tp
    if method in ("kNN","WkNN"):
        def pred_fn(): return predict_neighbors(model,Xtest,selected["y_ref"],int(params["k"]),method=="WkNN")
    else:
        def pred_fn():
            p=np.asarray(model.predict(Xtest)); p=p[:,None] if p.ndim==1 else p
            return ys.inverse_transform(p) if ys is not None else p
    _=pred_fn(); times=[]; pred=None
    for _i in range(max(1,inference_reps)):
        t=time.perf_counter(); pred=pred_fn(); times.append(time.perf_counter()-t)
    pred=np.asarray(pred,dtype=np.float64); model_s=float(np.median(times)); e2e_s=model_s+preprocess_s
    errs=error_vector(ytest,pred,metric_name); met=summarize_errors(errs); size=model_size_bytes(method,selected)
    comp={
        "final_fit_time_s":float(selected["selected_model_fit_time_s"]+selected["preprocess_fit_time_s"]),"selected_model_fit_time_s":float(selected["selected_model_fit_time_s"]),"preprocess_fit_time_s":float(selected["preprocess_fit_time_s"]),
        "test_preprocess_time_s":float(preprocess_s),"inference_total_time_s_median":model_s,"inference_ms_per_sample":1000*model_s/len(ytest),"throughput_samples_s":len(ytest)/model_s,
        "end_to_end_inference_total_time_s":e2e_s,"end_to_end_inference_ms_per_sample":1000*e2e_s/len(ytest),"end_to_end_throughput_samples_s":len(ytest)/e2e_s,
        "model_size_bytes":int(size),"model_size_mb":float(size/(1024**2)) if size>=0 else np.nan,"inference_repetitions":int(max(1,inference_reps)),"inference_times_s":times,
        "selected_validation_model_reused":True,"refit_after_selection":False,"final_model_training_scope":"train_only",
    }
    return pred,errs,met,comp


def fit_fixed_ablation(method,params,Xtr,ytr,Xte,yte,threads,metric_name,inference_reps):
    train_preprocess_s=0.0; test_preprocess_s=0.0; xs=ys=None
    if method in ("kNN","WkNN","MLP","ELM"):
        xs=StandardScaler(copy=False)
        t=time.perf_counter(); Xtr2=np.asarray(xs.fit_transform(Xtr),dtype=np.float32); train_preprocess_s+=time.perf_counter()-t
        t=time.perf_counter(); Xte2=np.asarray(xs.transform(Xte),dtype=np.float32); test_preprocess_s+=time.perf_counter()-t
    else: Xtr2,Xte2=Xtr,Xte
    if method in ("MLP","ELM"):
        t=time.perf_counter(); ys=StandardScaler(); ytr2=ys.fit_transform(ytr); train_preprocess_s+=time.perf_counter()-t
    else: ytr2=ytr
    if method in ("kNN","WkNN"):
        tf=time.perf_counter(); model=NearestNeighbors(n_neighbors=int(params["k"]),metric=params["metric"],algorithm="brute",n_jobs=threads); model.fit(Xtr2); fit_s=time.perf_counter()-tf
        selected={"model":model,"params":params,"x_scaler":None,"y_scaler":None,"y_ref":ytr,"selected_model_fit_time_s":fit_s,"preprocess_fit_time_s":0.0,"reference_bytes":int(Xtr2.nbytes+ytr.nbytes)}
    else:
        model=build_model(method,params,threads); tf=time.perf_counter()
        fit_target=ytr2.ravel() if (np.ndim(ytr2)==2 and ytr2.shape[1]==1 and method in ("RF","MLP")) else ytr2
        model.fit(Xtr2,fit_target); fit_s=time.perf_counter()-tf
        selected={"model":model,"params":params,"x_scaler":None,"y_scaler":ys,"y_ref":None,"selected_model_fit_time_s":fit_s,"preprocess_fit_time_s":0.0,"reference_bytes":None}
    pred,errs,met,comp=evaluate_selected(method,selected,Xte2,yte,metric_name,inference_reps)
    model_s=float(comp["inference_total_time_s_median"]); e2e_s=model_s+test_preprocess_s
    comp["final_fit_time_s"]=float(fit_s+train_preprocess_s)
    comp["preprocess_fit_time_s"]=float(train_preprocess_s)
    comp["test_preprocess_time_s"]=float(test_preprocess_s)
    comp["end_to_end_inference_total_time_s"]=e2e_s
    comp["end_to_end_inference_ms_per_sample"]=1000*e2e_s/len(yte)
    comp["end_to_end_throughput_samples_s"]=len(yte)/e2e_s
    comp["selected_validation_model_reused"]=False
    comp["final_model_training_scope"]="train_only_fixed_hyperparameters"
    return pred,errs,met,comp

def checkpoint_path(output,dataset,split_type,seed,variant,method): return output/dataset/"checkpoints"/split_type/f"seed_{seed}"/variant/f"{method}.json"

def load_checkpoint(p:Path):
    if not p.exists(): return None
    try:
        d=json.loads(p.read_text(encoding="utf-8")); return d if d.get("status")=="complete" else None
    except Exception: return None


def save_case(output:Path,dataset:str,result:Dict,errors:np.ndarray,preds:Optional[np.ndarray],validation_rows:Optional[List[Dict]]):
    cp=checkpoint_path(output,dataset,result["split_type"],result["seed"],result["variant"],result["method"]); cp.parent.mkdir(parents=True,exist_ok=True)
    stem=cp.with_suffix("")
    ep=stem.parent/f"{stem.name}_errors.npy"; np.save(ep,np.asarray(errors,dtype=np.float64),allow_pickle=False); result["errors_file"]=str(ep.relative_to(output/dataset))
    if preds is not None:
        pp=stem.parent/f"{stem.name}_predictions.npy"; np.save(pp,np.asarray(preds,dtype=np.float64),allow_pickle=False); result["predictions_file"]=str(pp.relative_to(output/dataset))
    if validation_rows is not None:
        vp=stem.parent/f"{stem.name}_validation.csv"; pd.DataFrame(validation_rows).to_csv(vp,index=False); result["validation_file"]=str(vp.relative_to(output/dataset))
    result["status"]="complete"; result["finished_at"]=time.strftime("%Y-%m-%d %H:%M:%S")
    cp.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")


def save_split(output,dataset,split_type,seed,split):
    p=output/dataset/"splits"/split_type/f"seed_{seed}.npz"
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=np.load(p)
        keys=[k for k in ["train","val","test"] if k in split]
        if all(k in old.files and np.array_equal(old[k],split[k]) for k in keys):
            return
        raise RuntimeError(f"Existing split differs from requested protocol: {p}. Use a new output directory or explicitly remove incompatible checkpoints/splits.")
    np.savez(p,**split)


def run_tuned_case(store,output,dataset,split_type,seed,variant,method,split,profile,threads,inference_reps,force=False,protocol_version=None):
    cp=checkpoint_path(output,dataset,split_type,seed,variant,method)
    if not force:
        old=load_checkpoint(cp)
        if old is not None: log(f"SKIP {dataset} {split_type} seed={seed} {variant} {method}"); return old
    tr,va,te=split["train"],split["val"],split["test"]
    log(f"START {dataset} {split_type} seed={seed} {variant} {method} n={len(tr)}/{len(va)}/{len(te)}")
    tcase=time.perf_counter(); Xtr=store.get_x(variant,tr); Xv=store.get_x(variant,va); Xte=store.get_x(variant,te); ytr=store.get_y(tr); yv=store.get_y(va); yte=store.get_y(te)
    best,rows,search_s,selected=candidate_search(method,Xtr,ytr,Xv,yv,grid(profile,method),threads,store.meta["target_metric"])
    pred,errs,met,comp=evaluate_selected(method,selected,Xte,yte,store.meta["target_metric"],inference_reps)
    result={"protocol_version":protocol_version or PROTOCOL_VERSION,"dataset_id":dataset,"task_family":store.meta["task_family"],"target_metric":store.meta["target_metric"],"status":"running","split_type":split_type,"seed":int(seed),"variant":variant,"method":method,"selection_mode":"validation_tuned_reuse_selected_model","best_params":best,"n_features":int(Xtr.shape[1]),"n_train":len(tr),"n_val":len(va),"n_test":len(te),"search_time_s":float(search_s),"total_case_runtime_s":float(time.perf_counter()-tcase),"test_used_after_selection_only":True,"train_validation_refit":False,**met,**comp}
    save_case(output,dataset,result,errs,pred,rows); log(f"DONE {dataset} {split_type} seed={seed} {variant} {method} mean={met['mean_error_m']:.3f} m")
    del Xtr,Xv,Xte,ytr,yv,yte,pred,errs,selected; gc.collect(); return result


def core_anchor_params(output,dataset,seed,method,base_split_type="random"):
    p=checkpoint_path(output,dataset,base_split_type,seed,"COMMON_FULL",method); d=load_checkpoint(p)
    if d is None: raise RuntimeError(f"Missing COMMON_FULL {base_split_type} anchor: {p}")
    return d["best_params"]


def run_ablation_case(store,output,dataset,seed,variant,method,split,base_split_type,threads,inference_reps,force=False):
    ablation_split_type=f"ablation_{base_split_type}"
    cp=checkpoint_path(output,dataset,ablation_split_type,seed,variant,method)
    if not force:
        old=load_checkpoint(cp)
        if old is not None: log(f"SKIP {dataset} {ablation_split_type} seed={seed} {variant} {method}"); return old
    params=core_anchor_params(output,dataset,seed,method,base_split_type); tr,va,te=split["train"],split["val"],split["test"]
    Xtr=store.get_x(variant,tr); Xte=store.get_x(variant,te); ytr=store.get_y(tr); yte=store.get_y(te); t=time.perf_counter()
    pred,errs,met,comp=fit_fixed_ablation(method,params,Xtr,ytr,Xte,yte,threads,store.meta["target_metric"],inference_reps)
    result={"protocol_version":PROTOCOL_VERSION,"dataset_id":dataset,"task_family":store.meta["task_family"],"target_metric":store.meta["target_metric"],"status":"running","split_type":ablation_split_type,"base_split_type":base_split_type,"seed":int(seed),"variant":variant,"method":method,"selection_mode":"fixed_from_corresponding_common_full_train_only","anchor_variant":"COMMON_FULL","anchor_split_type":base_split_type,"best_params":params,"n_features":int(Xtr.shape[1]),"n_train":len(tr),"n_val":len(va),"n_test":len(te),"search_time_s":0.0,"total_case_runtime_s":float(time.perf_counter()-t),"validation_samples_used_for_ablation_fit":False,"train_validation_refit":False,**met,**comp}
    save_case(output,dataset,result,errs,None,None); log(f"DONE {dataset} {ablation_split_type} seed={seed} {variant} {method} mean={met['mean_error_m']:.3f} m")
    del Xtr,Xte,ytr,yte,pred,errs; gc.collect(); return result


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p=np.asarray(pvals,dtype=float); n=len(p); order=np.argsort(p); adj=np.empty(n,float); running=0.0
    for rank,idx in enumerate(order):
        val=(n-rank)*p[idx]; running=max(running,val); adj[idx]=min(1.0,running)
    return adj


def rank_biserial(d: np.ndarray) -> float:
    d=np.asarray(d,float); d=d[d!=0]
    if len(d)==0: return 0.0
    r=rankdata(np.abs(d)); pos=r[d>0].sum(); neg=r[d<0].sum(); den=pos+neg
    return float((pos-neg)/den) if den else 0.0


def bootstrap_ci(x: np.ndarray, nboot=5000, seed=20260830):
    x=np.asarray(x,float)
    if len(x)==0: return (np.nan,np.nan)
    if len(x)==1: return (float(x[0]),float(x[0]))
    rng=np.random.default_rng(seed); means=np.empty(nboot,float)
    for i in range(nboot): means[i]=rng.choice(x,size=len(x),replace=True).mean()
    return tuple(np.percentile(means,[2.5,97.5]).tolist())


def paired_stats(df: pd.DataFrame, split_type: str) -> pd.DataFrame:
    rows=[]
    for m in METHODS:
        a=df[(df.split_type==split_type)&(df.method==m)&(df.variant=="RSSI_ONLY")].set_index("seed")["mean_error_m"]
        b=df[(df.split_type==split_type)&(df.method==m)&(df.variant=="COMMON_FULL")].set_index("seed")["mean_error_m"]
        common=sorted(set(a.index)&set(b.index)); d=np.array([a.loc[s]-b.loc[s] for s in common],float)
        if len(d)>=2 and np.any(d!=0):
            try: p=float(wilcoxon(d,alternative="two-sided",method="auto").pvalue)
            except Exception: p=np.nan
        else: p=np.nan
        lo,hi=bootstrap_ci(d)
        rows.append({"method":m,"n_paired_seeds":len(d),"mean_rssi_error_m":float(a.loc[common].mean()) if common else np.nan,"mean_multifeature_error_m":float(b.loc[common].mean()) if common else np.nan,"mean_improvement_rssi_minus_multifeature_m":float(d.mean()) if len(d) else np.nan,"median_improvement_m":float(np.median(d)) if len(d) else np.nan,"improvement_ci95_low_m":lo,"improvement_ci95_high_m":hi,"multifeature_win_fraction":float(np.mean(d>0)) if len(d) else np.nan,"rank_biserial":rank_biserial(d) if len(d) else np.nan,"wilcoxon_p_raw":p})
    out=pd.DataFrame(rows); valid=out.wilcoxon_p_raw.notna(); out["wilcoxon_p_holm_6methods"]=1.0
    if valid.any(): out.loc[valid,"wilcoxon_p_holm_6methods"]=holm_adjust(out.loc[valid,"wilcoxon_p_raw"].to_numpy())
    out["significant_holm_0_05"]=out.wilcoxon_p_holm_6methods<0.05
    return out


def collect_cases(dataset_root: Path) -> pd.DataFrame:
    rows=[]
    for p in dataset_root.glob("checkpoints/*/seed_*/*/*.json"):
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            if d.get("status")=="complete": rows.append(d)
        except Exception: pass
    return pd.DataFrame(rows)


def aggregate_dataset(output: Path, dataset: str, store: FeatureStore) -> None:
    root=output/dataset; agg=root/"aggregate"; agg.mkdir(parents=True,exist_ok=True)
    df=collect_cases(root)
    if df.empty:
        (agg/"VALIDATION_REPORT.txt").write_text("No completed cases.\n",encoding="utf-8"); return
    df.to_csv(agg/"all_case_summaries.csv",index=False)
    main=df[df.variant.isin(MAIN_VARIANTS)].copy()
    if not main.empty:
        summary=(main.groupby(["split_type","variant","method"],dropna=False).agg(n_seeds=("seed","nunique"),mean_error_m=("mean_error_m","mean"),std_across_seeds_m=("mean_error_m","std"),median_error_m=("median_error_m","mean"),p90_error_m=("p90_error_m","mean"),p95_error_m=("p95_error_m","mean"),p99_error_m=("p99_error_m","mean"),model_size_mb=("model_size_mb","median"),inference_ms_per_sample=("inference_ms_per_sample","median"),end_to_end_inference_ms_per_sample=("end_to_end_inference_ms_per_sample","median"),search_time_s=("search_time_s","median"),final_fit_time_s=("final_fit_time_s","median")).reset_index())
        summary.to_csv(agg/"accuracy_computational_summary_across_seeds.csv",index=False)
        for st in ["random","spatial"]:
            if st in set(main.split_type): paired_stats(main,st).to_csv(agg/f"paired_{st}_statistical_comparison.csv",index=False)
        # Computational paired ratios, random only.
        rr=[]
        for m in METHODS:
            a=main[(main.split_type=="random")&(main.variant=="RSSI_ONLY")&(main.method==m)].set_index("seed")
            b=main[(main.split_type=="random")&(main.variant=="COMMON_FULL")&(main.method==m)].set_index("seed")
            for s in sorted(set(a.index)&set(b.index)):
                ar=a.loc[s]; br=b.loc[s]
                rr.append({"method":m,"seed":int(s),"inference_ratio_mf_over_rssi":br.inference_ms_per_sample/ar.inference_ms_per_sample,"end_to_end_inference_ratio_mf_over_rssi":br.end_to_end_inference_ms_per_sample/ar.end_to_end_inference_ms_per_sample,"storage_ratio_mf_over_rssi":br.model_size_mb/ar.model_size_mb,"fit_ratio_mf_over_rssi":br.final_fit_time_s/ar.final_fit_time_s if ar.final_fit_time_s>0 else np.nan,"search_ratio_mf_over_rssi":br.search_time_s/ar.search_time_s if ar.search_time_s>0 else np.nan})
        cr=pd.DataFrame(rr); cr.to_csv(agg/"computational_ratios_by_seed.csv",index=False)
        if not cr.empty:
            nums=[c for c in cr.columns if c not in ("method","seed")]
            across=cr.groupby("method")[nums].agg(["median",lambda x:np.percentile(x,25),lambda x:np.percentile(x,75)])
            across.columns=[f"{a}_{'median' if b=='median' else ('q25' if b=='<lambda_0>' else 'q75')}" for a,b in across.columns]
            across.reset_index().to_csv(agg/"computational_ratios_across_seeds.csv",index=False)
        # Generalization gap
        if "spatial" in set(main.split_type):
            g=[]
            for m in METHODS:
                for v in MAIN_VARIANTS:
                    a=main[(main.split_type=="random")&(main.method==m)&(main.variant==v)].set_index("seed")["mean_error_m"]
                    b=main[(main.split_type=="spatial")&(main.method==m)&(main.variant==v)].set_index("seed")["mean_error_m"]
                    c=sorted(set(a.index)&set(b.index)); d=np.array([b.loc[s]-a.loc[s] for s in c],float)
                    g.append({"method":m,"variant":v,"n_paired_seeds":len(c),"mean_spatial_minus_random_m":float(d.mean()) if len(d) else np.nan,"median_gap_m":float(np.median(d)) if len(d) else np.nan})
            pd.DataFrame(g).to_csv(agg/"random_vs_spatial_generalization_gap.csv",index=False)
    # Ablation under both random and spatial holdout.
    for base_st in ["random","spatial"]:
        ab_st=f"ablation_{base_st}"
        ab=df[df.split_type==ab_st].copy()
        if ab.empty: continue
        anchor=df[(df.split_type==base_st)&(df.variant=="COMMON_FULL")].copy(); rows=[]
        for m in METHODS:
            aa=anchor[anchor.method==m].set_index("seed")["mean_error_m"]
            for v in sorted(ab.variant.unique()):
                bb=ab[(ab.method==m)&(ab.variant==v)].set_index("seed")["mean_error_m"]
                common=sorted(set(aa.index)&set(bb.index)); full=np.array([aa.loc[s] for s in common],float); drop=np.array([bb.loc[s] for s in common],float); d=drop-full
                rel=np.divide(100*d,full,out=np.full_like(d,np.nan),where=full!=0)
                if len(d)>=2 and np.any(d!=0):
                    try:p=float(wilcoxon(d,alternative="two-sided",method="auto").pvalue)
                    except:p=np.nan
                else:p=np.nan
                lo,hi=bootstrap_ci(d); rlo,rhi=bootstrap_ci(rel[np.isfinite(rel)])
                rows.append({"base_split_type":base_st,"method":m,"variant":v,"n_paired_seeds":len(d),"mean_attribute_impact_m":float(d.mean()) if len(d) else np.nan,
                             "median_attribute_impact_m":float(np.median(d)) if len(d) else np.nan,"impact_ci95_low_m":lo,"impact_ci95_high_m":hi,
                             "mean_relative_attribute_impact_pct":float(np.nanmean(rel)) if len(rel) else np.nan,"median_relative_attribute_impact_pct":float(np.nanmedian(rel)) if len(rel) else np.nan,
                             "relative_impact_ci95_low_pct":rlo,"relative_impact_ci95_high_pct":rhi,"positive_impact_fraction":float(np.mean(d>0)) if len(d) else np.nan,
                             "rank_biserial":rank_biserial(d) if len(d) else np.nan,"wilcoxon_p_raw":p})
        ast=pd.DataFrame(rows); ast["wilcoxon_p_holm_global_30"]=1.0; valid=ast.wilcoxon_p_raw.notna()
        if valid.any(): ast.loc[valid,"wilcoxon_p_holm_global_30"]=holm_adjust(ast.loc[valid,"wilcoxon_p_raw"].to_numpy())
        ast["wilcoxon_p_holm_within_method_5"]=1.0
        for m in METHODS:
            ix=(ast.method==m)&ast.wilcoxon_p_raw.notna()
            if ix.any(): ast.loc[ix,"wilcoxon_p_holm_within_method_5"]=holm_adjust(ast.loc[ix,"wilcoxon_p_raw"].to_numpy())
        ast.to_csv(agg/f"ablation_{base_st}_paired_statistical_comparison.csv",index=False)
    # Validation report
    seeds_present=sorted(df.seed.unique().tolist())
    counts_by_split={st:int((df.split_type==st).sum()) for st in ["random","spatial","ablation_random","ablation_spatial"]}
    expected_per_seed={"random":12,"spatial":12,"ablation_random":30,"ablation_spatial":30}
    lines=[f"Protocol: {PROTOCOL_VERSION}",f"Dataset: {dataset}",f"Deployment: {store.meta.get('deployment_id')}",f"Task family: {store.meta['task_family']}",f"Target metric: {store.meta['target_metric']}",
           f"Completed cases: {len(df)}",f"Completed seeds: {seeds_present}",f"Feature groups: {store.meta['feature_groups']}",f"Main representations: {store.meta['representations']}",
           f"Ablation variants: {store.ablation_variants()}",f"Counts by split type: {counts_by_split}",f"Expected cases per present seed when that task is complete: {expected_per_seed}"]
    dtype_ok=True; pred_ok=True; errors_checked=preds_checked=0
    for _,r in df.iterrows():
        ep=root/str(r.errors_file).replace("\\","/"); a=np.load(ep,mmap_mode="r"); errors_checked+=1; dtype_ok &= (a.dtype==np.float64)
        pf=r.get("predictions_file")
        if isinstance(pf,str) and pf:
            a=np.load(root/pf.replace("\\","/"),mmap_mode="r"); preds_checked+=1; pred_ok &= (a.dtype==np.float64)
    main2=df[df.split_type.isin(["random","spatial"])]
    reuse_ok=bool(main2.selected_validation_model_reused.fillna(False).astype(bool).all()) if not main2.empty and "selected_validation_model_reused" in main2 else False
    protocol_ok=bool((df.protocol_version==PROTOCOL_VERSION).all()) if "protocol_version" in df else False
    spatial_disjoint=True; spatial_checked=0
    split_root=root/"splits"/"spatial"
    if split_root.exists():
        for sp in split_root.glob("seed_*.npz"):
            z=np.load(sp); spatial_checked+=1
            gt=set(z["group_train"].tolist()); gv=set(z["group_val"].tolist()); ge=set(z["group_test"].tolist())
            spatial_disjoint &= not bool((gt&gv)|(gt&ge)|(gv&ge))
    lines += [f"Protocol version consistent: {protocol_ok}",f"Error arrays float64: {dtype_ok} ({errors_checked} checked)",f"Prediction arrays float64: {pred_ok} ({preds_checked} checked)",
              f"Main selected validation models reused without refit: {reuse_ok}",f"Spatial holdout supported: {store.meta.get('spatial_holdout_supported')}",f"Spatial group sets disjoint: {spatial_disjoint} ({spatial_checked} splits checked)"]
    (agg/"VALIDATION_REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (agg/"dataset_metadata.json").write_text(json.dumps(store.meta,indent=2,ensure_ascii=False),encoding="utf-8")


def _wilcoxon_one_sample(values: Sequence[float]) -> Tuple[float,float,float,float,float,float]:
    d=np.asarray(values,dtype=float); d=d[np.isfinite(d)]
    if len(d)>=2 and np.any(d!=0):
        try: p=float(wilcoxon(d,alternative="two-sided",method="auto").pvalue)
        except Exception: p=np.nan
    else: p=np.nan
    lo,hi=bootstrap_ci(d)
    return (float(d.mean()) if len(d) else np.nan,float(np.median(d)) if len(d) else np.nan,
            float(np.mean(d>0)) if len(d) else np.nan,rank_biserial(d) if len(d) else np.nan,p,lo,hi)


def _scenario_seed_main_effects(output: Path, datasets: Sequence[str], stores: Dict[str,FeatureStore], split_type: str) -> pd.DataFrame:
    rows=[]
    for d in datasets:
        cases=collect_cases(output/d)
        if cases.empty: continue
        sub=cases[(cases.split_type==split_type)&(cases.variant.isin(MAIN_VARIANTS))]
        for m in METHODS:
            a=sub[(sub.method==m)&(sub.variant=="RSSI_ONLY")].set_index("seed")["mean_error_m"]
            b=sub[(sub.method==m)&(sub.variant=="COMMON_FULL")].set_index("seed")["mean_error_m"]
            for seed in sorted(set(a.index)&set(b.index)):
                r=float(a.loc[seed]); c=float(b.loc[seed]); imp=r-c
                rows.append({"deployment_id":stores[d].meta["deployment_id"],"scenario_id":d,"split_type":split_type,"method":m,"seed":int(seed),
                             "rssi_error_m":r,"common_full_error_m":c,"improvement_m":imp,"relative_improvement_pct":100.0*imp/r if r!=0 else np.nan})
    return pd.DataFrame(rows)


def _deployment_main_stats(seed_effects: pd.DataFrame) -> pd.DataFrame:
    if seed_effects.empty: return pd.DataFrame()
    depseed=(seed_effects.groupby(["deployment_id","method","seed"],as_index=False)
             .agg(n_scenarios=("scenario_id","nunique"),relative_improvement_pct=("relative_improvement_pct","mean")))
    rows=[]
    for dep in DEPLOYMENTS:
        for m in METHODS:
            z=depseed[(depseed.deployment_id==dep)&(depseed.method==m)]
            d=z.relative_improvement_pct.to_numpy(float)
            mean,med,win,rb,p,lo,hi=_wilcoxon_one_sample(d)
            rows.append({"deployment_id":dep,"method":m,"n_paired_seeds":len(d),"n_scenarios":int(z.n_scenarios.max()) if len(z) else 0,
                         "mean_relative_improvement_pct":mean,"median_relative_improvement_pct":med,"relative_ci95_low_pct":lo,"relative_ci95_high_pct":hi,
                         "common_full_win_fraction":win,"rank_biserial":rb,"wilcoxon_p_raw":p})
    out=pd.DataFrame(rows); out["wilcoxon_p_holm_6methods"]=1.0
    for dep in DEPLOYMENTS:
        ix=(out.deployment_id==dep)&out.wilcoxon_p_raw.notna()
        if ix.any(): out.loc[ix,"wilcoxon_p_holm_6methods"]=holm_adjust(out.loc[ix,"wilcoxon_p_raw"].to_numpy())
    out["significant_holm_0_05"]=out.wilcoxon_p_holm_6methods<0.05
    out["significant_common_better"]=out.significant_holm_0_05&(out.mean_relative_improvement_pct>0)
    out["significant_common_worse"]=out.significant_holm_0_05&(out.mean_relative_improvement_pct<0)
    return depseed,out


def _scenario_seed_ablation_effects(output:Path,datasets:Sequence[str],stores:Dict[str,FeatureStore],base_split_type:str)->pd.DataFrame:
    rows=[]
    for d in datasets:
        cases=collect_cases(output/d)
        if cases.empty: continue
        anchor=cases[(cases.split_type==base_split_type)&(cases.variant=="COMMON_FULL")]
        ab=cases[cases.split_type==f"ablation_{base_split_type}"]
        for m in METHODS:
            aa=anchor[anchor.method==m].set_index("seed")["mean_error_m"]
            for v in sorted(ab.variant.unique()):
                bb=ab[(ab.method==m)&(ab.variant==v)].set_index("seed")["mean_error_m"]
                for seed in sorted(set(aa.index)&set(bb.index)):
                    full=float(aa.loc[seed]); drop=float(bb.loc[seed]); impact=drop-full
                    rows.append({"deployment_id":stores[d].meta["deployment_id"],"scenario_id":d,"base_split_type":base_split_type,"method":m,"variant":v,"seed":int(seed),
                                 "common_full_error_m":full,"drop_error_m":drop,"impact_m":impact,"relative_attribute_impact_pct":100.0*impact/full if full!=0 else np.nan})
    return pd.DataFrame(rows)


def _deployment_ablation_stats(seed_effects:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    if seed_effects.empty: return pd.DataFrame(),pd.DataFrame()
    depseed=(seed_effects.groupby(["deployment_id","method","variant","seed"],as_index=False)
             .agg(n_scenarios=("scenario_id","nunique"),relative_attribute_impact_pct=("relative_attribute_impact_pct","mean")))
    rows=[]
    for dep in DEPLOYMENTS:
        for m in METHODS:
            for v in sorted(seed_effects.variant.unique()):
                z=depseed[(depseed.deployment_id==dep)&(depseed.method==m)&(depseed.variant==v)]
                d=z.relative_attribute_impact_pct.to_numpy(float); mean,med,win,rb,p,lo,hi=_wilcoxon_one_sample(d)
                rows.append({"deployment_id":dep,"method":m,"variant":v,"n_paired_seeds":len(d),"n_scenarios":int(z.n_scenarios.max()) if len(z) else 0,
                             "mean_relative_attribute_impact_pct":mean,"median_relative_attribute_impact_pct":med,"impact_ci95_low_pct":lo,"impact_ci95_high_pct":hi,
                             "positive_impact_fraction":win,"rank_biserial":rb,"wilcoxon_p_raw":p})
    out=pd.DataFrame(rows); out["wilcoxon_p_holm_global_30"]=1.0; out["wilcoxon_p_holm_within_method_5"]=1.0
    for dep in DEPLOYMENTS:
        ix=(out.deployment_id==dep)&out.wilcoxon_p_raw.notna()
        if ix.any(): out.loc[ix,"wilcoxon_p_holm_global_30"]=holm_adjust(out.loc[ix,"wilcoxon_p_raw"].to_numpy())
        for m in METHODS:
            ix=(out.deployment_id==dep)&(out.method==m)&out.wilcoxon_p_raw.notna()
            if ix.any(): out.loc[ix,"wilcoxon_p_holm_within_method_5"]=holm_adjust(out.loc[ix,"wilcoxon_p_raw"].to_numpy())
    return depseed,out


def _cross_deployment_summary(depstats:pd.DataFrame)->pd.DataFrame:
    if depstats.empty: return pd.DataFrame()
    rows=[]
    for m in METHODS:
        z=depstats[depstats.method==m]
        vals=z.mean_relative_improvement_pct.to_numpy(float)
        rows.append({"method":m,"n_deployments":int(z.deployment_id.nunique()),"mean_equal_weight_relative_improvement_pct":float(np.mean(vals)),
                     "median_deployment_relative_improvement_pct":float(np.median(vals)),"deployments_with_positive_mean_improvement":int(np.sum(vals>0)),
                     "deployments_significant_common_better":int(z.significant_common_better.sum()),"deployments_significant_common_worse":int(z.significant_common_worse.sum()),
                     "deployments_not_significant":int((~z.significant_holm_0_05).sum())})
    return pd.DataFrame(rows)


def aggregate_cross_dataset(output: Path, datasets: Sequence[str], stores: Dict[str,FeatureStore]) -> None:
    """Equal-weight synthesis over the four source deployments.

    Brno and Bucharest contain two evaluation scenarios each. Scenario-level effects are first converted
    to relative percentages and averaged within deployment *for the same seed*. Statistical testing then
    uses 20 deployment-level paired seed effects. Antwerp, Bonn, Brno and Bucharest therefore each receive
    exactly one unit of weight in the cross-deployment interpretation.
    """
    agg=output/"aggregate_equal_weight"; agg.mkdir(parents=True,exist_ok=True)
    protocol={"protocol_version":PROTOCOL_VERSION,"deployments":DEPLOYMENTS,"scenarios_by_deployment":SCENARIOS_BY_DEPLOYMENT,
              "main_variants":MAIN_VARIANTS,"common_feature_groups":["rssi","snr","esp","rx_mask","aggregates"],
              "ablation_variants":["DROP_SNR","DROP_ESP","DROP_SNR_ESP","DROP_RX_MASK","DROP_AGGREGATES"],
              "deployment_weighting":"1:1:1:1; scenario relative effects averaged within deployment for each seed"}
    (agg/"protocol_metadata.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf-8")
    for st in ["random","spatial"]:
        se=_scenario_seed_main_effects(output,datasets,stores,st)
        if se.empty: continue
        se.to_csv(agg/f"scenario_seed_{st}_relative_effects.csv",index=False)
        depseed,depstats=_deployment_main_stats(se); depseed.to_csv(agg/f"deployment_seed_{st}_relative_effects.csv",index=False); depstats.to_csv(agg/f"deployment_{st}_statistical_comparison.csv",index=False)
        _cross_deployment_summary(depstats).to_csv(agg/f"cross_deployment_{st}_method_summary.csv",index=False)
    for base_st in ["random","spatial"]:
        ab=_scenario_seed_ablation_effects(output,datasets,stores,base_st)
        if ab.empty: continue
        ab.to_csv(agg/f"scenario_seed_ablation_{base_st}_relative_effects.csv",index=False)
        depseed,depstats=_deployment_ablation_stats(ab); depseed.to_csv(agg/f"deployment_seed_ablation_{base_st}_relative_effects.csv",index=False); depstats.to_csv(agg/f"deployment_ablation_{base_st}_statistical_comparison.csv",index=False)
        summ=(depstats.groupby(["method","variant"],as_index=False)
              .agg(n_deployments=("deployment_id","nunique"),mean_equal_weight_relative_attribute_impact_pct=("mean_relative_attribute_impact_pct","mean"),
                   median_deployment_relative_attribute_impact_pct=("mean_relative_attribute_impact_pct","median"),
                   deployments_positive_impact=("mean_relative_attribute_impact_pct",lambda x:int(np.sum(np.asarray(x)>0)))))
        summ.to_csv(agg/f"cross_deployment_ablation_{base_st}_summary.csv",index=False)
    # Number of distinct spatial test-block sets actually exercised across the seeds.
    diagnostics=[]
    for d in datasets:
        root=output/d/"splits"/"spatial"; combos=[]
        if root.exists():
            for p in sorted(root.glob("seed_*.npz")):
                x=np.load(p); combos.append(tuple(sorted(map(int,x["group_test"].tolist()))))
        diagnostics.append({"deployment_id":stores[d].meta["deployment_id"],"scenario_id":d,"n_spatial_blocks":int(len(np.unique(stores[d].spatial_groups))),
                            "n_saved_spatial_seeds":len(combos),"n_unique_test_block_sets":len(set(combos))})
    pd.DataFrame(diagnostics).to_csv(agg/"spatial_partition_diversity.csv",index=False)
    readme=["Equal-footing cross-deployment aggregation", "",
            "Primary comparison uses exactly the same feature groups in every scenario: RSSI, SNR, derived ESP, reception mask, and 15 radio aggregates.",
            "Packet metadata and gateway geometry are excluded from the primary comparison because they are not available comparably in all four source deployments.",
            "Antwerp, Bonn, Brno and Bucharest receive equal weight. Brno and Bucharest subscenarios are averaged within deployment at the seed level before inference.",
            "Raw localization errors are never pooled across deployments. Relative paired effects are used for cross-deployment synthesis.",
            "Spatial holdout uses the same deterministic 20-block sample-balanced Morton-order construction in every scenario; entire blocks are assigned to train/validation/test."]
    (agg/"README.txt").write_text("\n".join(readme)+"\n",encoding="utf-8")

def system_metadata(threads:int,protocol_version:Optional[str]=None) -> Dict[str,Any]:
    # thread_pools: BLAS and OpenMP libraries in memory and their thread limit (see lorabench/threads.py)
    return {"protocol_version":protocol_version or PROTOCOL_VERSION,"platform":platform.platform(),"python_version":sys.version,"physical_cpus":psutil.cpu_count(logical=False),"logical_cpus":psutil.cpu_count(logical=True),"ram_gb":psutil.virtual_memory().total/(1024**3),"threads_requested":threads,"numpy_version":np.__version__,"pandas_version":pd.__version__,"thread_pools":thread_pools()}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--task",choices=["cache","core","spatial","ablation_random","ablation_spatial","ablation","aggregate","all"],default="all")
    ap.add_argument("--datasets",default="primary",help="primary/all, deployment names (antwerp,bonn,brno,bucharest), or comma-separated scenario IDs")
    ap.add_argument("--profile",choices=["quick","desktop"],default="desktop")
    ap.add_argument("--seeds",nargs="*",type=int,default=SEEDS_20)
    ap.add_argument("--threads",type=int,default=6)
    ap.add_argument("--data-dir",default="datasets")
    ap.add_argument("--cache-dir",default="cache_equal_footing_v5")
    ap.add_argument("--output-dir",default="results_equal_footing_v5")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--force-cache",action="store_true")
    ap.add_argument("--max-samples",type=int,default=0,help="Optional deterministic sample cap; intended for smoke/development runs only")


def run(args) -> None:
    data_dir=Path(args.data_dir); cache_root=Path(args.cache_dir); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    datasets=parse_dataset_list(args.datasets); inference_reps=2 if args.profile=="quick" else 5
    (output/"system_metadata.json").write_text(json.dumps(system_metadata(args.threads),indent=2),encoding="utf-8")
    if args.task=="cache":
        build_many(datasets,data_dir,cache_root,args.force_cache)
        for d in datasets:
            m=json.loads((cache_root/d/"metadata.json").read_text()); log(f"CACHE {d}: n={m['n_samples']} gw={m['n_gateways']} groups={m['feature_groups']} metric={m['target_metric']}")
        return
    stores={}
    for d in datasets:
        build_dataset(d,data_dir,cache_root,args.force_cache); stores[d]=FeatureStore(cache_root/d)
        log(f"DATASET {d}: n={len(stores[d].y)}, metric={stores[d].meta['target_metric']}, features={stores[d].meta['feature_groups']}")
    if args.task in ("core","spatial","ablation_random","ablation_spatial","ablation","all"):
        for d,store in stores.items():
            maxs=args.max_samples if args.max_samples>0 else None
            indices=deterministic_subset(store,d,maxs)
            for seed in args.seeds:
                rs=split_random(indices,seed); save_split(output,d,"random",seed,rs)
                if args.task in ("core","all"):
                    for v in MAIN_VARIANTS:
                        for m in METHODS: run_tuned_case(store,output,d,"random",seed,v,m,rs,args.profile,args.threads,inference_reps,args.force)
                if args.task in ("spatial","ablation_spatial","ablation","all") and store.meta.get("spatial_holdout_supported"):
                    ss=split_spatial(indices,np.asarray(store.spatial_groups),seed); save_split(output,d,"spatial",seed,ss)
                    for v in MAIN_VARIANTS:
                        for m in METHODS: run_tuned_case(store,output,d,"spatial",seed,v,m,ss,args.profile,args.threads,inference_reps,args.force)
                if args.task in ("ablation_random","ablation_spatial","ablation","all"):
                    # Fixed-hyperparameter ablation uses the COMMON_FULL anchor from the matching split type.
                    jobs=[]
                    if args.task in ("ablation_random","ablation","all"): jobs.append(("random",rs))
                    if args.task in ("ablation_spatial","ablation","all"): jobs.append(("spatial",ss))
                    for base_st,split_obj in jobs:
                        for v in store.ablation_variants():
                            for m in METHODS: run_ablation_case(store,output,d,seed,v,m,split_obj,base_st,args.threads,inference_reps,args.force)
    if args.task in ("aggregate","all","core","spatial","ablation_random","ablation_spatial","ablation"):
        for d,store in stores.items(): aggregate_dataset(output,d,store)
        aggregate_cross_dataset(output,datasets,stores)
    log("FINISHED")


def main():
    ap=argparse.ArgumentParser(); add_arguments(ap); run(ap.parse_args())


if __name__=="__main__": main()
