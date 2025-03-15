"""Module for fuzzy matching on multiple characteristics."""

import uuid
from pathlib import Path

import pandas as pd

from fuzzy_matching.matchers import DistanceMatcher, VectorMatcher, NullMatcher


class MultiMatcher:
    """Fuzzy matching on multiple characteristics.

    Parameters
    ----------
    top_n : int
        Number of results to return.
    config : dict
        Dict specifying field name and matching algoritm.
    encryption_key : str
        Encryption key to use for storing data.
    storage_path : str, default="storage"
        Folder to store data in.
    """

    def __init__(
        self, top_n, config, encryption_key: str, storage_path="storage"
    ) -> None:
        # Create the storage path if needed.
        storage_path = Path(storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        self._top_n = top_n
        self._weights = {
            field: settings.get("weight", 1.0) for field, settings in config.items()
        }

        self._matchers = {}
        for field, settings in config.items():
            algoritm = settings["algoritm"]
            if algoritm == "distance":
                self._matchers[field] = DistanceMatcher(
                    field, encryption_key, storage_path
                )

            elif algoritm == "vector":
                self._matchers[field] = VectorMatcher(
                    field, encryption_key, storage_path
                )
            elif algoritm == "null":
                self._matchers[field] = NullMatcher(field, encryption_key, storage_path)

            else:
                raise TypeError(f"Unknown matching algoritm: {algoritm}")

    def create(self, data) -> None:
        """Add data to the matching set."""
        missing = set(self._matchers) - set(data.columns)
        if missing:
            raise RuntimeError("Missing fields in the data: " + ".".join(missing))

        uuids = pd.Series([self._make_id() for _ in range(len(data))])

        for field, matcher in self._matchers.items():
            matcher.create(uuids, data[field])

    def get(self, target: dict) -> pd.DataFrame:
        """Match records from the matching set."""
        results = []
        similarities = None

        # Get similarity scores from the individual matchers.
        for field, matcher in self._matchers.items():
            values, similarity = matcher.get(target[field])

            if values is None:
                raise RuntimeError(f"No results for {field}; aborting...")

            results.append(values)

            if similarities is None:
                similarities = similarity * self._weights[field]
            else:
                similarities += similarity * self._weights[field]

        top_similar = similarities.nlargest(self._top_n)
        results = [result[top_similar.index] for result in results]
        results = pd.concat(results, axis=1)

        return results.assign(similarity=top_similar)

    def delete(self) -> None:
        """Delete all matching data."""
        for field, matcher in self._matchers.items():
            matcher.delete(field)

    @staticmethod
    def _make_id() -> str:
        """Generate a UUID4 identifier."""
        return uuid.uuid4().hex
