"""Module for matching time differences."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from fuzzy_matching.storage import EncryptedStore


class TimedeltaMatcher:
    """Class for matching time differences."""

    def __init__(
        self,
        field: str,
        weight: float,
        encryption_key: bytes,
        storage_path: Path,
        date_format: str,
    ):
        self._field = field
        self._weight = weight
        self._format = date_format
        self._storage = EncryptedStore(field, encryption_key, storage_path)

    def create(self, data) -> None:
        """Store encrypted datetime values."""
        data = data.assign(
            **{self._field: pd.to_datetime(data[self._field], format=self._format)}
        )

        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Search names in the vector space."""
        data = self._storage.load()
        if data is None:
            return None

        target = pd.to_datetime(target, format=self._format)

        # Compute absolute time differences and normalize.
        deltas = (data[self._field] - target).abs()
        deltas = (deltas.max() - deltas) / deltas.max()

        data = data.assign(**{f"similarity_{self._field}": deltas * self._weight})
        return data.set_index("id")

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._vector_store.delete()
        self._storage.delete()
