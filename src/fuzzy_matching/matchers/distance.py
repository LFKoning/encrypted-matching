"""Module for fuzzy matching using edit distances."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein
from rapidfuzz.distance.DamerauLevenshtein import normalized_similarity as damerau
from rapidfuzz.distance.OSA import normalized_similarity as optimal_alignment
from rapidfuzz.process import cdist

from .base_string import StringMatcher
from fuzzy_matching.storage import EncryptedStore


class DistanceMatcher(StringMatcher):
    """Module for fuzzy matching using edit distances."""

    ALGORITMS = {
        "levenshtein": levenshtein,
        "damerau": damerau,
        "alignment": optimal_alignment,
    }

    def __init__(
        self,
        field: str,
        encryption_key: bytes,
        storage_path: Path,
        algoritm: str,
    ) -> None:
        if algoritm.lower() not in self.ALGORITMS:
            raise ValueError(
                f"Unknown distance algoritm: {algoritm}."
                "Valid options are: " + ", ".join(self.ALGORITMS)
            )

        self._algoritm = self.ALGORITMS[algoritm]
        self._field = field
        self._storage = EncryptedStore(field, encryption_key, storage_path)

    def create(self, uuids: pd.Series, values: pd.Series) -> None:
        """Add values to the matching set, return a list of UUIDs."""
        # Perform basic data preprocessing.
        values = values.map(self._preprocess)

        # Store values with UUIDs and name.
        values.index = uuids
        values.name = self._field
        self._storage.store(values)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Match the target, return scores for the matching set."""
        values = self._storage.retrieve()
        if values is None:
            return None, None

        target = self._preprocess(target)

        similarities = cdist(
            [target],
            values,
            scorer=self._algoritm,
            workers=-1,
        )
        similarities = pd.Series(similarities[0], index=values.index)

        return values, similarities

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
