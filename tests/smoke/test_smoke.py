from billwise.common.config import get_config
from billwise.common.db import init_db
from billwise.common.storage import ensure_directories


def test_phase1_bootstrap():
    cfg = get_config()
    ensure_directories()
    init_db()

    assert cfg.project_name == "BillWise"
    assert cfg.paths.raw_dir.exists()
    assert cfg.paths.processed_dir.exists()
    assert cfg.paths.reviewed_dir.exists()
    assert cfg.paths.exports_dir.exists()
    assert cfg.paths.logs_dir.exists()
    assert cfg.paths.database_path.exists()