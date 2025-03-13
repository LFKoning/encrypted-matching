"""Module for fuzzy matching using edit distances."""

from pathlib import Path

import pandas as pd
from rapidfuzz.distance.OSA import normalized_similarity
from rapidfuzz.process import cdist

from fuzzy_matching.storage import EncryptedStore


class DistanceMatcher:
    """Module for fuzzy matching using edit distances."""

    ENCODING = "utf8"

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
        data = pd.DataFrame(
            {
                "uuid": uuids,
                self._field: values,
            }
        )
        self._storage.store(data)

    def get(self, target: str) -> pd.DataFrame:
        """Match the target, return scores for the matching set."""
        data = self._storage.retrieve()
        if data is None:
            return None

        similarities = cdist(
            [target],
            data[self._field],
            scorer=normalized_similarity,
            workers=-1,
        )
        return data.assign(**{f"similarity_{self._field}": similarities[0]})
