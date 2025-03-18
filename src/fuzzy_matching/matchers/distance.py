"""Module for fuzzy matching using edit distances."""

from pathlib import Path

import pandas as pd
from rapidfuzz.distance.DamerauLevenshtein import normalized_similarity as damerau
from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein
from rapidfuzz.distance.OSA import normalized_similarity as optimal_alignment
from rapidfuzz.process import cdist

from .bases import BaseMatcher, StringMixin


class DistanceMatcher(BaseMatcher, StringMixin):
    """Module for fuzzy matching using edit distances.

    Parameters
    ----------
    field : str
        Name of the field used in matching.
    encryption_key : bytes
        Encryption key for storing data, provided as bytes.
    storage_path : pathlib.Path
        Path to a file to store the data in.
    settings : dict, optional
        Additional settings for the algoritm.
    """

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
        """Add entities to the matching set.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with entities to add to the matching set.
        """
        # Perform basic data preprocessing.
        data = data.assign(**{self._field: data[self._field].map(self._preprocess)})

        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

    def get(self, target: str) -> pd.DataFrame:
        """Return all entities and their similarity to the target.

        Parameters
        ----------
        target : str
            Target string to match against.

        Returns
        -------
        pandas.DataFrame
            DataFrame of entities and their similarity scores.
        """
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
