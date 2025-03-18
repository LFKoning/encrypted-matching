"""Module for fields not used in matching."""

import pandas as pd

from .bases import BaseMatcher


class NullMatcher(BaseMatcher):
    """Module for fields not used in matching."""

    def create(self, data: pd.DataFrame) -> None:
        """Add entities to the matching set.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with entities to add to the matching set.
        """
        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

    def get(self, _: str) -> pd.DataFrame:
        """Return all entities and their similarity to the target.

        Returns
        -------
        pandas.DataFrame
            DataFrame of entities and their similarity scores.
        """
        data = self._storage.load()
        if data is None:
            return None

        data = data.assign(**{f"similarity_{self._field}": 0})
        return data.set_index("id")

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
