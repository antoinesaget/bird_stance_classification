"""Purpose: Verify prediction prefill request shaping and import behavior"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from birdsys.labelstudio import prefill_predictions as mod


def test_task_is_untouched_and_drafts_excluded() -> None:
    assert mod.task_is_untouched({"total_annotations": 0, "is_labeled": False, "drafts": []}) is True
    assert mod.task_is_untouched({"total_annotations": 1, "is_labeled": False, "drafts": []}) is False
    assert mod.task_is_untouched({"total_annotations": 0, "is_labeled": True, "drafts": []}) is False
    assert mod.task_is_untouched({"total_annotations": 0, "is_labeled": False, "drafts": [{"id": 1}]}) is False
    assert mod.task_has_drafts({"drafts": [{"id": 1}]}) is True


def test_main_replace_existing_imports_before_delete(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    events: list[tuple[str, int]] = []
    detail_calls: dict[int, int] = {10: 0}

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv=None: Namespace(
            base_url="http://ls",
            api_token="token",
            project_id=7,
            ml_backend_url="http://ml",
            task_page_size=100,
            predict_batch_size=16,
            import_batch_size=100,
            only_missing=False,
            untouched_only=True,
            replace_existing=True,
            store_empty=True,
            limit=0,
            report_out=str(report_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "task_pages",
        lambda base_url, project_id, api_token, page_size: iter(
            [{"id": 10, "total_annotations": 0, "is_labeled": False, "drafts": [], "total_predictions": 1}]
        ),
    )

    def fake_get_task_detail(base_url: str, api_token: str, task_id: int) -> dict:
        detail_calls[task_id] += 1
        if detail_calls[task_id] == 1:
            return {"predictions": [{"id": 501}]}
        return {"predictions": [{"id": 501}, {"id": 777}]}

    monkeypatch.setattr(mod, "get_task_detail", fake_get_task_detail)
    monkeypatch.setattr(
        mod,
        "ml_predict",
        lambda ml_backend_url, tasks: [{"task": 10, "model_version": "new", "score": 0.9, "result": [{"id": "x"}]}],
    )

    def fake_import_predictions(base_url: str, project_id: int, api_token: str, predictions: list[dict]) -> dict:
        events.append(("import", int(predictions[0]["task"])))
        return {"created": len(predictions)}

    monkeypatch.setattr(mod, "import_predictions", fake_import_predictions)

    def fake_delete_prediction(base_url: str, api_token: str, prediction_id: int) -> None:
        events.append(("delete", prediction_id))

    monkeypatch.setattr(mod, "delete_prediction", fake_delete_prediction)

    assert mod.main() == 0
    assert events == [("import", 10), ("delete", 501)]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tasks_successfully_replaced"] == 1
    assert report["old_prediction_rows_deleted"] == 1
    assert report["tasks_left_on_old_predictions"] == 0


def test_main_keeps_old_prediction_on_backend_error(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    deleted: list[int] = []

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv=None: Namespace(
            base_url="http://ls",
            api_token="token",
            project_id=7,
            ml_backend_url="http://ml",
            task_page_size=100,
            predict_batch_size=16,
            import_batch_size=100,
            only_missing=False,
            untouched_only=True,
            replace_existing=True,
            store_empty=True,
            limit=0,
            report_out=str(report_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "task_pages",
        lambda base_url, project_id, api_token, page_size: iter(
            [{"id": 11, "total_annotations": 0, "is_labeled": False, "drafts": [], "total_predictions": 1}]
        ),
    )
    monkeypatch.setattr(mod, "get_task_detail", lambda base_url, api_token, task_id: {"predictions": [{"id": 601}]})
    monkeypatch.setattr(
        mod,
        "ml_predict",
        lambda ml_backend_url, tasks: [{"task": 11, "error": "backend exploded", "result": []}],
    )
    monkeypatch.setattr(mod, "delete_prediction", lambda base_url, api_token, prediction_id: deleted.append(prediction_id))

    assert mod.main() == 1
    assert deleted == []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["backend_failures"] == 1
    assert report["tasks_left_on_old_predictions"] == 1
    assert report["tasks_successfully_replaced"] == 0


def test_main_replaces_with_explicit_empty_prediction(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    deleted: list[int] = []
    detail_calls: dict[int, int] = {12: 0}

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv=None: Namespace(
            base_url="http://ls",
            api_token="token",
            project_id=7,
            ml_backend_url="http://ml",
            task_page_size=100,
            predict_batch_size=16,
            import_batch_size=100,
            only_missing=False,
            untouched_only=True,
            replace_existing=True,
            store_empty=True,
            limit=0,
            report_out=str(report_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "task_pages",
        lambda base_url, project_id, api_token, page_size: iter(
            [{"id": 12, "total_annotations": 0, "is_labeled": False, "drafts": [], "total_predictions": 1}]
        ),
    )

    def fake_get_task_detail(base_url: str, api_token: str, task_id: int) -> dict:
        detail_calls[task_id] += 1
        if detail_calls[task_id] == 1:
            return {"predictions": [{"id": 701}]}
        return {"predictions": [{"id": 701}, {"id": 702}]}

    monkeypatch.setattr(mod, "get_task_detail", fake_get_task_detail)
    monkeypatch.setattr(
        mod,
        "ml_predict",
        lambda ml_backend_url, tasks: [{"task": 12, "model_version": "new", "score": 0.0, "result": []}],
    )
    monkeypatch.setattr(mod, "import_predictions", lambda base_url, project_id, api_token, predictions: {"created": 1})
    monkeypatch.setattr(mod, "delete_prediction", lambda base_url, api_token, prediction_id: deleted.append(prediction_id))

    assert mod.main() == 0
    assert deleted == [701]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["empty_predictions"] == 1
    assert report["stored_empty_predictions"] == 1
    assert report["tasks_successfully_replaced"] == 1


def test_main_counts_skips_and_partial_failures(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    detail_calls: dict[int, int] = {20: 0, 21: 0}

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda argv=None: Namespace(
            base_url="http://ls",
            api_token="token",
            project_id=7,
            ml_backend_url="http://ml",
            task_page_size=100,
            predict_batch_size=2,
            import_batch_size=2,
            only_missing=False,
            untouched_only=True,
            replace_existing=True,
            store_empty=True,
            limit=0,
            report_out=str(report_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "task_pages",
        lambda base_url, project_id, api_token, page_size: iter(
            [
                {"id": 30, "total_annotations": 1, "is_labeled": True, "drafts": [], "total_predictions": 1},
                {"id": 31, "total_annotations": 0, "is_labeled": False, "drafts": [{"id": 1}], "total_predictions": 1},
                {"id": 20, "total_annotations": 0, "is_labeled": False, "drafts": [], "total_predictions": 1},
                {"id": 21, "total_annotations": 0, "is_labeled": False, "drafts": [], "total_predictions": 1},
            ]
        ),
    )

    def fake_get_task_detail(base_url: str, api_token: str, task_id: int) -> dict:
        detail_calls[task_id] += 1
        if detail_calls[task_id] == 1:
            return {"predictions": [{"id": task_id + 1000}]}
        if task_id == 20:
            return {"predictions": [{"id": 1020}, {"id": 2020}]}
        return {"predictions": [{"id": 1021}, {"id": 2021}]}

    monkeypatch.setattr(mod, "get_task_detail", fake_get_task_detail)
    monkeypatch.setattr(
        mod,
        "ml_predict",
        lambda ml_backend_url, tasks: [
            {"task": 20, "model_version": "new", "score": 0.9, "result": [{"id": "a"}]},
            {"task": 21, "model_version": "new", "score": 0.9, "result": [{"id": "b"}]},
        ],
    )
    monkeypatch.setattr(mod, "import_predictions", lambda base_url, project_id, api_token, predictions: {"created": len(predictions)})

    def fake_delete_prediction(base_url: str, api_token: str, prediction_id: int) -> None:
        if prediction_id == 1021:
            raise RuntimeError("cannot delete")

    monkeypatch.setattr(mod, "delete_prediction", fake_delete_prediction)

    assert mod.main() == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scanned_tasks"] == 4
    assert report["eligible_tasks"] == 2
    assert report["skipped_annotated"] == 1
    assert report["skipped_drafts"] == 1
    assert report["tasks_successfully_replaced"] == 1
    assert report["old_prediction_rows_deleted"] == 1
    assert report["delete_failures"] == 1
    assert report["tasks_left_on_old_predictions"] == 1
