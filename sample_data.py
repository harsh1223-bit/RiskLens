from __future__ import annotations
import numpy as np
import pandas as pd

def make_demo_data(n=500, seed=42):
    rng=np.random.default_rng(seed)
    gender=rng.choice(["Female","Male"],size=n,p=[0.49,0.51])
    age=rng.integers(21,66,size=n)
    income=np.clip(rng.normal(62000,18000,size=n),15000,150000).round(0)
    credit_history=rng.choice([0,1],size=n,p=[0.22,0.78])
    age_band=pd.cut(age,[20,30,40,50,70],labels=["21-30","31-40","41-50","51+"]).astype(str)
    # Synthetic target with a modest group-related signal, strictly for demo.
    logit=-2.1 + income/35000 + credit_history*1.25 + (age-40)*0.012
    prob=1/(1+np.exp(-logit))
    y_true=rng.binomial(1,prob)
    # Simulated imperfect predictions, independent of actual protected group.
    pred_prob=np.clip(prob+rng.normal(0,0.18,size=n),0.02,0.98)
    y_pred=(pred_prob>=0.5).astype(int)
    return pd.DataFrame({"y_true":y_true,"y_pred":y_pred,"y_prob":pred_prob.round(4),"income":income,"age":age,"credit_history":credit_history,"gender":gender,"age_band":age_band})

def make_demo_drift_data(reference_df, seed=7):
    rng=np.random.default_rng(seed)
    new=reference_df.copy()
    if "income" in new:
        new["income"]=(pd.to_numeric(new["income"],errors="coerce")*1.18+rng.normal(0,5000,len(new))).clip(lower=10000)
    if "age" in new:
        new["age"]=(pd.to_numeric(new["age"],errors="coerce")+rng.integers(0,7,len(new))).clip(18,90)
    if "gender" in new:
        new["gender"]=rng.choice(["Female","Male"],size=len(new),p=[0.42,0.58])
    return new
