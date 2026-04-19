from billwise.pipeline.orchestrator import determine_final_status


def test_phase5_determine_final_status_no_review():
    processing_status, review_status, requires_review = determine_final_status(False, False)

    assert processing_status == "ready_for_dashboard"
    assert review_status == "approved"
    assert requires_review is False


def test_phase5_determine_final_status_with_review():
    processing_status, review_status, requires_review = determine_final_status(True, False)

    assert processing_status == "ready_for_dashboard"
    assert review_status == "pending"
    assert requires_review is True