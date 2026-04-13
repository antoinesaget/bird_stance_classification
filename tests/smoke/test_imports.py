from __future__ import annotations


def test_core_imports() -> None:
    import birdsys  # noqa: F401
    from birdsys.core import attributes, config, logging, models, paths  # noqa: F401
    from birdsys.ml_experiments import metrics  # noqa: F401


def test_backend_imports() -> None:
    from birdsys.ml_backend.app.predictors import model_a_yolo, model_b_attributes  # noqa: F401
    from birdsys.ml_backend.app import serializers  # noqa: F401
