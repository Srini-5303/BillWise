from billwise.common.gcs_storage import build_blob_path


def test_phase8_gcs_blob_paths():
    path = build_blob_path("processed", "receipts", "abc.json")
    assert "processed/receipts/abc.json" in path