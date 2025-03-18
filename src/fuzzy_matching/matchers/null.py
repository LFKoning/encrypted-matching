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

    def create(self, data: pd.DataFrame) -> None:
        """Add values to the matching set, return a list of UUIDs."""
        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

    def get(self, _: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Match the target, return scores for the matching set."""
        data = self._storage.load()
        if data is None:
            return None

        data = data.assign(**{f"similarity_{self._field}": 0})
        return data.set_index("id")

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
