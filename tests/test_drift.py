import pytest
import pandas as pd
from drift import population_stability_index, drift_report

def test_psi_identical_distributions_is_near_zero():
    values=[1,2,3,4,5,6,7,8,9,10]
    assert population_stability_index(values,values) == pytest.approx(0.0, abs=1e-9)

def test_drift_report_has_numeric_and_categorical_checks():
    reference=pd.DataFrame({"x":[1,2,3,4,5,6],"category":["a","a","b","b","a","b"]})
    new=pd.DataFrame({"x":[10,11,12,13,14,15],"category":["c","c","c","c","a","c"]})
    result=drift_report(reference,new)
    assert len(result["rows"]) == 2
    by_feature={r["feature"]:r for r in result["rows"]}
    assert by_feature["x"]["test"] == "KS"
    assert by_feature["category"]["test"] == "Chi-square"
