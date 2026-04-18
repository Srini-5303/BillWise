from __future__ import annotations

LABELS = [
    "merchant_name",
    "date",
    "time",
    "subtotal",
    "tax",
    "total",
    "payment_method",
    "card_last4",
    "receipt_number",
    "item_description",
    "item_quantity",
    "item_unit_price",
    "item_total",
    "other",
    "ignore",
]

ANNOTATION_GUIDELINES = """
Receipt Annotation Guidelines

Goal:
Create consistent token/box-level labels for future LayoutLM fine-tuning.

General rules:
- Label visible text conservatively.
- Do not guess missing fields.
- Keep labels consistent across receipts.
- Use 'other' for visible text that is not part of a target field.
- Use 'ignore' only for noise/artifacts that should not be learned.

Field definitions:
- merchant_name: main store/merchant name only, not slogan/address
- date: transaction date only
- time: transaction time only
- subtotal: pre-tax total
- tax: tax amount
- total: final amount paid
- payment_method: cash/card/mobile_wallet/contactless/gift_card/ebt/check/other payment text
- card_last4: only 4 digits clearly tied to tender/card context
- receipt_number: transaction/receipt/reference number
- item_description: purchased product name text
- item_quantity: explicit quantity text like '2 @' or '3 x'
- item_unit_price: explicit per-unit pricing
- item_total: line total for the purchased item
- other: visible text not belonging to target labels
- ignore: OCR garbage / non-meaningful artifacts

Do not label as items:
- coupon lines
- savings/discount lines
- subtotal/tax/total lines
- headers/footers
- phone/address/web lines
- thank-you text
- loyalty/member/reward text unless explicitly part of payment context

Card_last4 rules:
- Only label if clearly tied to card context
- Do not label naked 4-digit numbers
- Do not label tax percentages, reference fragments, or unrelated codes

Box rules:
- Keep boxes tight around the text
- Do not merge neighboring unrelated text
- If OCR split a phrase into multiple boxes, label each box consistently
- Under-labeling is better than incorrect labeling
"""