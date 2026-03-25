from __future__ import annotations

from scripts import create_labelstudio_project


def test_list_paginated_supports_tasks_responses(monkeypatch) -> None:
    calls: list[str] = []

    def fake_request_json(base_url: str, path: str, api_token: str):  # type: ignore[no-untyped-def]
        calls.append(path)
        if path == "/api/tasks?project=4&page_size=2":
            return {"total": 3, "tasks": [{"id": 1}, {"id": 2}]}
        if path == "/api/tasks?project=4&page_size=2&page=2":
            return {"total": 3, "tasks": [{"id": 3}]}
        raise AssertionError(path)

    monkeypatch.setattr(create_labelstudio_project, "request_json", fake_request_json)

    items = create_labelstudio_project.list_paginated("http://labelstudio.local", "/api/tasks?project=4&page_size=2", "token")

    assert [item["id"] for item in items] == [1, 2, 3]
    assert calls == ["/api/tasks?project=4&page_size=2", "/api/tasks?project=4&page_size=2&page=2"]
