import pytest
from fairness import fairness_report

def test_fairness_group_rates_and_gap():
    result=fairness_report([1,0,1,0],[1,0,0,1],["A","A","B","B"])
    assert result["available"]
    rows={r["group"]:r for r in result["rows"]}
    assert rows["A"]["selection_rate"] == pytest.approx(0.5)
    assert rows["B"]["selection_rate"] == pytest.approx(0.5)
    assert result["gaps"]["selection_rate"] == pytest.approx(0.0)
    assert result["gaps"]["tpr"] == pytest.approx(1.0)

def test_fairness_rejects_length_mismatch():
    with pytest.raises(ValueError):
        fairness_report([1,0],[1,0],["A"])
