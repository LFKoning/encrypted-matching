"""Module for matching time differences."""

from pathlib import Path

import pandas as pd

from .bases import BaseMatcher


class TimedeltaMatcher(BaseMatcher):
    """Class for matching time differences.

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

    def __init__(
        self,
        field: str,
        encryption_key: bytes,
        storage_path: Path,
        settings: dict = None,
    ):
        super().__init__(field, encryption_key, storage_path, settings)
        self._format = settings.get("date_format", "%d-%m-%Y")

    def create(self, data) -> None:
        """Add entities to the matching set.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with entities to add to the matching set.
        """
        data = data.assign(
            **{self._field: pd.to_datetime(data[self._field], format=self._format)}
        )

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
