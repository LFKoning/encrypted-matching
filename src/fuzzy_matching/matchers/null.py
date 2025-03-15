"""Module for fields not used in matching."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from fuzzy_matching.storage import EncryptedStore


class NullMatcher:
    """Module for fields not used in matching."""

    def __init__(
        self,
        field: str,
        encryption_key: bytes,
        storage_path: Path,
    ) -> None:
        self._field = field
        self._storage = EncryptedStore(field, encryption_key, storage_path)

    def create(self, uuids: pd.Series, values: pd.Series) -> None:
        """Add values to the matching set, return a list of UUIDs."""
        values.index = uuids
        values.name = self._field
        self._storage.store(values)

    def get(self, _: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Match the target, return scores for the matching set."""
        values = self._storage.retrieve()
        if values is None:
            return None, None

        similarities = pd.Series(0, index=values.index)

        return values, similarities

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
