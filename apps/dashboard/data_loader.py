from __future__ import annotations

import streamlit as st

from billwise.dashboard.bridge import (
    get_data_source_label as _get_data_source_label,
    load_all_data as _load_all_data,
    load_all_ocr_corrections as _load_all_ocr_corrections,
    load_all_validations as _load_all_validations,
    load_validated_item_ids as _load_validated_item_ids,
    save_category_validation as _save_category_validation,
    save_ocr_correction as _save_ocr_correction,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_data():
    return _load_all_data()


def get_data_source_label():
    return _get_data_source_label()


def save_category_validation(
    item_id: str,
    receipt_id: str,
    raw_item_text: str,
    original_category: str,
    validated_category: str,
    validator_note: str = "",
) -> None:
    _save_category_validation(
        item_id=item_id,
        receipt_id=receipt_id,
        raw_item_text=raw_item_text,
        original_category=original_category,
        validated_category=validated_category,
        validator_note=validator_note,
    )
    st.cache_data.clear()


def save_ocr_correction(
    receipt_id: str,
    field_name: str,
    original_value: str,
    corrected_value: str,
) -> None:
    _save_ocr_correction(
        receipt_id=receipt_id,
        field_name=field_name,
        original_value=original_value,
        corrected_value=corrected_value,
    )
    st.cache_data.clear()


def load_validated_item_ids():
    return _load_validated_item_ids()


def load_all_validations():
    return _load_all_validations()


def load_all_ocr_corrections():
    return _load_all_ocr_corrections()