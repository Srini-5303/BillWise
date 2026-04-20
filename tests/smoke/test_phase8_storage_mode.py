from billwise.common.config import get_config


def test_phase8_storage_backend_value():
    cfg = get_config()
    assert cfg.gcs.storage_backend in {"local", "hybrid", "gcs", "gcs_only"}