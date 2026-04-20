from billwise.dashboard.exports import export_dashboard_views


def test_phase8_dashboard_exports_returns_shape():
    result = export_dashboard_views()

    assert "receipts_path" in result
    assert "items_path" in result
    assert "joined_path" in result
    assert "receipts_rows" in result
    assert "items_rows" in result
    assert "joined_rows" in result
    assert "gcs_paths" in result