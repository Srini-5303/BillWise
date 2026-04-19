from billwise.common.db import init_db
from billwise.common.ids import new_id
from billwise.common.repositories import CategorizationRepository, ReceiptRepository
from billwise.common.schemas import CategoryPrediction, LineItem, ReceiptRecord
from billwise.common.storage import ensure_directories
from billwise.dashboard.bridge import load_all_data


def test_phase6_dashboard_bridge_reads_shared_db():
    ensure_directories()
    init_db()

    receipt_id = new_id("rcpt")
    item_id = new_id("item")

    receipt = ReceiptRecord(
        receipt_id=receipt_id,
        image_path="data/raw/test_dashboard.jpg",
        vendor_name="Dashboard Test Mart",
        receipt_date="2026-04-19",
        total=9.99,
        tax=0.50,
        payment_method="visa",
        extraction_method="hybrid",
        items=[
            LineItem(
                item_id=item_id,
                raw_name="BANANA",
                normalized_name="banana",
                quantity=2,
                unit_price=0.75,
                item_total=1.50,
                item_confidence=0.91,
                item_source="hybrid-selected",
            )
        ],
    )
    ReceiptRepository.upsert_receipt(receipt)

    CategorizationRepository.replace_for_items(
        [
            CategoryPrediction(
                item_id=item_id,
                predicted_category="Fruits",
                category_confidence=0.97,
                top_k_scores={"Fruits": 0.97, "Vegetables": 0.03},
                categorizer_model="distilbert_receipt_classifier",
                needs_human_review=False,
                final_category="Fruits",
            )
        ]
    )

    df_receipts, df_items, df_joined = load_all_data()

    r = df_receipts[df_receipts["receipt_id"] == receipt_id]
    i = df_items[df_items["receipt_id"] == receipt_id]
    j = df_joined[df_joined["receipt_id"] == receipt_id]

    assert not r.empty
    assert not i.empty
    assert not j.empty
    assert r.iloc[0]["vendor_name"] == "Dashboard Test Mart"
    assert float(r.iloc[0]["receipt_total"]) == 9.99
    assert i.iloc[0]["raw_item_text"] == "BANANA"
    assert i.iloc[0]["category"] == "Fruits"