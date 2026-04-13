"""Purpose: Verify the end-to-end backend response contract expected by Label Studio clients"""
from __future__ import annotations

from birdsys.ml_backend.app.predictors.model_a_yolo import Detection
from birdsys.ml_backend.app.predictors.model_b_attributes import AttributePrediction
from birdsys.ml_backend.app.response_contract import extract_predictions, format_predict_response
from birdsys.ml_backend.app.serializers import to_label_studio_prediction


def test_serializer_shapes_prediction() -> None:
    det = Detection(x=0.1, y=0.2, w=0.3, h=0.4, score=0.88)
    attr = AttributePrediction(
        readability="readable",
        readability_conf=0.9,
        specie="correct",
        specie_conf=0.85,
        behavior="resting",
        behavior_conf=0.8,
        substrate="bare_ground",
        substrate_conf=0.8,
        stance="bipedal",
        stance_conf=0.7,
    )

    out = to_label_studio_prediction(
        task_id=123,
        detections=[det],
        attributes=[attr],
        model_version="test-v1",
    )

    assert out["task"] == 123
    assert out["model_version"] == "test-v1"
    assert isinstance(out["result"], list)
    assert any(item["type"] == "rectanglelabels" for item in out["result"])
    assert any(item["from_name"] == "specie" for item in out["result"])
    assert any(item["from_name"] == "isbird" for item in out["result"])
    assert any(item["from_name"] == "behavior" for item in out["result"])
    assert any(item["from_name"] == "substrate" for item in out["result"])
    assert any(item["from_name"] == "stance" for item in out["result"])
    assert not any(item["from_name"] == "image_status" for item in out["result"])


def test_serializer_hides_irrelevant_conditional_fields() -> None:
    det = Detection(x=0.1, y=0.2, w=0.3, h=0.4, score=0.88)
    attr = AttributePrediction(
        readability="unreadable",
        readability_conf=0.6,
        specie="correct",
        specie_conf=0.8,
        behavior="unsure",
        behavior_conf=0.5,
        substrate="unsure",
        substrate_conf=0.5,
        stance="unsure",
        stance_conf=0.5,
    )

    out = to_label_studio_prediction(
        task_id=321,
        detections=[det],
        attributes=[attr],
        model_version="test-v2",
    )

    from_names = [item["from_name"] for item in out["result"]]
    assert "readability" in from_names
    assert "specie" in from_names
    assert "isbird" in from_names
    assert "behavior" not in from_names
    assert "substrate" not in from_names
    assert "stance" not in from_names


def test_serializer_allows_stance_on_unsure_substrate() -> None:
    det = Detection(x=0.1, y=0.2, w=0.3, h=0.4, score=0.88)
    attr = AttributePrediction(
        readability="readable",
        readability_conf=0.9,
        specie="correct",
        specie_conf=0.8,
        behavior="resting",
        behavior_conf=0.6,
        substrate="unsure",
        substrate_conf=0.5,
        stance="unsure",
        stance_conf=0.5,
    )

    out = to_label_studio_prediction(
        task_id=777,
        detections=[det],
        attributes=[attr],
        model_version="test-v3",
    )

    from_names = [item["from_name"] for item in out["result"]]
    assert "stance" in from_names


def test_predict_response_contract_supports_both_keys() -> None:
    predictions = [{"task": 1, "result": [], "score": 0.5, "model_version": "v1"}]
    payload = format_predict_response(predictions)

    assert payload["results"] == predictions
    assert payload["predictions"] == predictions


def test_extract_predictions_supports_results_and_predictions() -> None:
    predictions = [{"task": 9, "result": [{"id": "a"}]}]

    assert extract_predictions({"results": predictions}) == predictions
    assert extract_predictions({"predictions": predictions}) == predictions
    assert extract_predictions({"predictions": "bad"}) == []
