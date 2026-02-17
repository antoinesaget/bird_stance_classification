from __future__ import annotations


def test_core_imports() -> None:
    import birdsys  # noqa: F401
    from birdsys import config, logging, paths  # noqa: F401
    from birdsys.training import attributes, metrics, models  # noqa: F401


def test_backend_imports() -> None:
    from services.ml_backend.app.predictors import model_a_yolo, model_b_attributes, model_c_image_status  # noqa: F401
    from services.ml_backend.app import serializers  # noqa: F401
