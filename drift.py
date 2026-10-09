from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

def population_stability_index(reference, new, bins=10):
    ref=pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    cur=pd.to_numeric(pd.Series(new), errors="coerce").dropna().to_numpy()
    if len(ref)==0 or len(cur)==0:
        return np.nan
    edges=np.unique(np.quantile(ref, np.linspace(0,1,bins+1)))
    if len(edges)<2:
        edges=np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts,_=np.histogram(ref,bins=edges)
    cur_counts,_=np.histogram(cur,bins=edges)
    # Add a small smoothing constant to avoid log(0).
    ref_pct=(ref_counts+0.5)/(ref_counts.sum()+0.5*len(ref_counts))
    cur_pct=(cur_counts+0.5)/(cur_counts.sum()+0.5*len(cur_counts))
    return float(np.sum((cur_pct-ref_pct)*np.log(cur_pct/ref_pct)))

def drift_report(reference_df, new_df, psi_some=0.1, psi_significant=0.25):
    common=[c for c in reference_df.columns if c in new_df.columns]
    rows=[]
    for col in common:
        ref=reference_df[col]
        cur=new_df[col]
        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            psi=population_stability_index(ref,cur)
            a=pd.to_numeric(ref,errors="coerce").dropna()
            b=pd.to_numeric(cur,errors="coerce").dropna()
            ks_p=float(ks_2samp(a,b).pvalue) if len(a)>0 and len(b)>0 else np.nan
            test="KS"
            p_value=ks_p
        else:
            psi=np.nan
            a=ref.fillna("Missing").astype(str).value_counts()
            b=cur.fillna("Missing").astype(str).value_counts()
            categories=sorted(set(a.index)|set(b.index))
            table=np.array([[a.get(k,0) for k in categories],[b.get(k,0) for k in categories]])
            try:
                p_value=float(chi2_contingency(table).pvalue) if table.shape[1]>1 else np.nan
            except ValueError:
                p_value=np.nan
            test="Chi-square"
        if pd.notna(psi) and psi > psi_significant:
            status="Red"
        elif pd.notna(psi) and psi > psi_some:
            status="Amber"
        elif pd.notna(psi):
            status="Green"
        elif pd.notna(p_value) and p_value < 0.01:
            status="Amber"
        else:
            status="Green" if pd.notna(p_value) else "Unavailable"
        rows.append({"feature":col,"psi":float(psi) if pd.notna(psi) else np.nan,"test":test,"p_value":float(p_value) if pd.notna(p_value) else np.nan,"status":status,"reference_n":int(ref.notna().sum()),"new_n":int(cur.notna().sum())})
    return {"rows":rows}
