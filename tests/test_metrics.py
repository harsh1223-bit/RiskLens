import pytest
from metrics import evaluate_predictions

def test_metrics_hand_computed_binary():
    result=evaluate_predictions([1,0,1,0],[1,0,0,0],[0.9,0.2,0.4,0.1])
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2/3)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[2,0],[1,1]]

def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_predictions([1,0],[1])
