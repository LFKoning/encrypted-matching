"""Module for matching time differences."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from fuzzy_matching.storage import EncryptedStore


class TimedeltaMatcher:
    """Class for matching time differences."""

    def __init__(
        self,
        field,
        encryption_key: bytes,
        storage_path: Path,
        date_format: str,
    ):
        self._field = field
        self._format = date_format
        self._storage = EncryptedStore(field, encryption_key, storage_path)

    def create(self, uuids: pd.Series, values: pd.Series) -> None:
        """Store encrypted datetime values."""
        values = pd.to_datetime(values, format=self._format)

        # Store datetime values.
        values.index = uuids
        values.name = self._field
        self._storage.store(values)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Search names in the vector space."""
        # Load the encrypted values.
        values = self._storage.retrieve()
        if values is None:
            return None, None

        target = pd.to_datetime(target, format=self._format)

        # Compute absolute time differences and normalize.
        deltas = (values - target).abs()
        deltas = (deltas.max() - deltas) / deltas.max()
        deltas = pd.Series(deltas, index=values.index)

        return values, deltas

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._vector_store.delete()
        self._storage.delete()
