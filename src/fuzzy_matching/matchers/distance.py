"""Module for fuzzy matching using edit distances."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from rapidfuzz.distance.OSA import normalized_similarity
from rapidfuzz.process import cdist

from fuzzy_matching.storage import EncryptedStore


class DistanceMatcher:
    """Module for fuzzy matching using edit distances."""

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

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Match the target, return scores for the matching set."""
        values = self._storage.retrieve()
        if values is None:
            return None, None

        similarities = cdist(
            [target],
            values,
            scorer=normalized_similarity,
            workers=-1,
        )
        similarities = pd.Series(similarities[0], index=values.index)

        return values, similarities

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
