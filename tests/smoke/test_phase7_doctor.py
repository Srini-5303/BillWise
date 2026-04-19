from billwise.common.doctor import collect_doctor_report


def test_phase7_doctor_report_shape():
    report = collect_doctor_report()

    assert "project_root" in report
    assert "checks" in report
    assert "summary" in report

    assert isinstance(report["checks"], list)
    assert isinstance(report["summary"], dict)

    assert "ok" in report["summary"]
    assert "warn" in report["summary"]
    assert "fail" in report["summary"]

    assert any(check["name"] == "project_root" for check in report["checks"])