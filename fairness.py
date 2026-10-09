from __future__ import annotations
import numpy as np
import pandas as pd

def fairness_report(y_true, y_pred, groups, amber_threshold=0.05, red_threshold=0.10, positive_label=None):
    yt = pd.Series(y_true).reset_index(drop=True)
    yp = pd.Series(y_pred).reset_index(drop=True)
    gr = pd.Series(groups).reset_index(drop=True)
    if not (len(yt) == len(yp) == len(gr)):
        raise ValueError("Labels, predictions, and groups must have the same length.")
    frame = pd.DataFrame({"y_true":yt, "y_pred":yp, "group":gr}).dropna()
    if frame.empty:
        return {"available":False, "message":"No complete rows are available for fairness analysis.", "rows":[], "gaps":{}}
    classes = sorted(pd.unique(pd.concat([frame.y_true, frame.y_pred]).astype(str)).tolist())
    if len(classes) < 2:
        return {"available":False, "message":"At least two label classes are needed for fairness metrics.", "rows":[], "gaps":{}}
    pos = str(positive_label) if positive_label is not None else ("1" if "1" in classes else classes[-1])
    frame["actual_pos"] = frame.y_true.astype(str) == pos
    frame["pred_pos"] = frame.y_pred.astype(str) == pos
    rows=[]
    for group, part in frame.groupby("group", dropna=True):
        tp = int((part.actual_pos & part.pred_pos).sum())
        fp = int((~part.actual_pos & part.pred_pos).sum())
        fn = int((part.actual_pos & ~part.pred_pos).sum())
        tn = int((~part.actual_pos & ~part.pred_pos).sum())
        rows.append({
            "group":str(group), "n":int(len(part)),
            "selection_rate":float(part.pred_pos.mean()),
            "tpr":float(tp/(tp+fn)) if tp+fn else np.nan,
            "fpr":float(fp/(fp+tn)) if fp+tn else np.nan,
            "precision":float(tp/(tp+fp)) if tp+fp else np.nan
        })
    df=pd.DataFrame(rows)
    metrics=["selection_rate","tpr","fpr","precision"]
    gaps={}
    for metric in metrics:
        vals=df[metric].dropna()
        gaps[metric]=float(vals.max()-vals.min()) if len(vals)>=2 else None
    def status(gap):
        if gap is None or pd.isna(gap): return "Unavailable"
        if gap < amber_threshold: return "Green"
        if gap <= red_threshold: return "Amber"
        return "Red"
    for metric in metrics:
        gaps[metric+"_status"]=status(gaps[metric])
    return {"available":True, "rows":rows, "gaps":gaps, "positive_label":pos}
