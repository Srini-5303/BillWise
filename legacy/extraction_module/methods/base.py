from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation.canonical import CanonicalReceipt


class BaseExtractionMethod(ABC):
    name: str

    @abstractmethod
    def extract(self, image_path: str, receipt_id: str) -> CanonicalReceipt:
        raise NotImplementedError