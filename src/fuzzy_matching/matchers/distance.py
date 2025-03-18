"""Module for fuzzy matching using edit distances."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from rapidfuzz.distance.DamerauLevenshtein import normalized_similarity as damerau
from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein
from rapidfuzz.distance.OSA import normalized_similarity as optimal_alignment
from rapidfuzz.process import cdist

from .bases import BaseMatcher, StringMixin


class DistanceMatcher(BaseMatcher, StringMixin):
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
        settings: dict = None,
    ) -> None:
        super().__init__(field, encryption_key, storage_path, settings)
        self._algoritm = self.ALGORITMS[settings["algoritm"].lower()]

    def create(self, data: pd.DataFrame) -> None:
        """Add identifiers and values to the matching set."""
        # Perform basic data preprocessing.
        data = data.assign(**{self._field: data[self._field].map(self._preprocess)})

        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Match the target, return scores for the matching set."""
        data = self._storage.load()
        if data is None:
            return None

        target = self._preprocess(target)
        similarities = cdist(
            [target],
            data[self._field],
            scorer=self._algoritm,
            workers=-1,
        )

        data = data.assign(
            **{f"similarity_{self._field}": similarities[0] * self._weight}
        )
        return data.set_index("id")

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
