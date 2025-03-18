"""Module for fuzzy matching on multiple characteristics."""

from pathlib import Path

import pandas as pd

from fuzzy_matching.matchers import (
    DistanceMatcher,
    NullMatcher,
    TimedeltaMatcher,
    VectorMatcher,
)


class MultiMatcher:
    """Fuzzy matching on multiple characteristics.

    Parameters
    ----------
    top_n : int
        Number of results to return.
    config : dict
        Dict of field names and matching settings.
    encryption_key : bytes
        Encryption key for storing data, provided as bytes.
    storage_path : str, default="storage"
        Folder to store data in.
    """

    def __init__(
        self, top_n, config, encryption_key: bytes, storage_path="storage"
    ) -> None:
        # Create the storage path if needed.
        storage_path = Path(storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        self._top_n = top_n

        # Define available matching algoritms.
        matchers = {algo: DistanceMatcher for algo in DistanceMatcher.ALGORITMS}
        matchers |= {
            "vector": VectorMatcher,
            "timedelta": TimedeltaMatcher,
            "null": NullMatcher,
        }

        self._matchers = {}
        for field, settings in config.items():
            algoritm = settings.get("algoritm").lower()
            if algoritm not in matchers:
                raise ValueError(
                    f"Unknown matching algoritm: {algoritm}."
                    "Available matchers: " + ", ".join(matchers)
                )

            self._matchers[field] = matchers[algoritm](
                field, encryption_key, storage_path, settings
            )

    def create(self, data: pd.DataFrame, id_column: str) -> None:
        """Add data to the matching set.

        Parameters
        ----------
        data : pandas.DataFrame
            Pandas DataFrame with data to add to the matching set.
        id_column : str
            Name of the column with entity identifiers.
            Note: Entitity dentifiers must be unique!
        """
        if id_column not in data.columns:
            raise RuntimeError(f"Missing ID column {id_column!r} in the data")
        data = data.rename(columns={id_column: "id"})

        missing = set(self._matchers) - set(data.columns)
        if missing:
            raise RuntimeError("Missing columns in the data: " + ".".join(missing))

        for field, matcher in self._matchers.items():
            matcher.create(data[["id", field]])

    def get(self, target: dict) -> pd.DataFrame:
        """Match records from the matching set.

        Parameters
        ----------
        target : dict
            Search query as dict of field : value pairs.
        """
        results = []

        # Get similarity scores from the individual matchers.
        for field, matcher in self._matchers.items():
            result = matcher.get(target[field])
            if result is None:
                raise RuntimeError(f"No results for {field}; aborting...")

            results.append(result)

        results = pd.concat(results, axis=1, join="inner")
        columns = [c for c in results.columns if c.startswith("similarity")]
        results["similarity"] = results[columns].sum(axis=1)
        results = results.nlargest(self._top_n, columns="similarity")

        return results.sort_values(by="similarity", ascending=False)

    def delete(self) -> None:
        """Delete all matching data."""
        for field, matcher in self._matchers.items():
            matcher.delete(field)
