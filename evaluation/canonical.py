from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class CanonicalFields(BaseModel):
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[
        Literal["cash", "card", "mobile_wallet", "contactless", "gift_card", "ebt", "check", "other"]
    ] = None
    card_last4: Optional[str] = None
    receipt_number: Optional[str] = None


class CanonicalLineItem(BaseModel):
    line_id: Optional[int] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    item_total: Optional[float] = None


class CanonicalReceipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    receipt_id: str
    image_file: str
    fields: CanonicalFields
    items: List[CanonicalLineItem] = Field(default_factory=list)