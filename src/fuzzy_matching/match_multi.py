"""Module for fuzzy matching on multiple characteristics."""

import re
import string
import unicodedata
import uuid
from pathlib import Path

import pandas as pd

from fuzzy_matching.match_distance import DistanceMatcher
from fuzzy_matching.match_vector import VectorMatcher


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
            field: settings["weight"] for field, settings in config.items()
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

            else:
                raise TypeError(f"Unknown matching algoritm: {algoritm}")

    def create(self, data) -> None:
        """Add data to the matching set."""
        missing = set(self._matchers) - set(data.columns)
        if missing:
            raise RuntimeError("Missing fields in the data: " + ".".join(missing))

        uuids = pd.Series([self._make_id() for _ in range(len(data))])

        for field, matcher in self._matchers.items():
            prep_data = data[field].map(self._preprocess)
            matcher.create(uuids, prep_data)

    def get(self, target: dict) -> pd.DataFrame:
        """Match records from the matching set."""
        results = None

        # Get similarity scores from the individual matchers.
        for field, matcher in self._matchers.items():
            prep_target = self._preprocess(target[field])
            result = matcher.get(prep_target)
            if results is None:
                results = result
            else:
                results = results.merge(result, on="uuid", how="left")

        # Aggragate similarity scores across fields.
        similarity = pd.Series(0, index=results.index)
        for field, weight in self._weights.items():
            similarity += results[f"similarity_{field}"] * weight

        similarity = similarity.nlargest(self._top_n)
        results = results.loc[similarity.index, :]

        results = results.assign(similarity=similarity)

        return results

    def delete(self) -> None:
        """Delete all matching data."""
        for field, matcher in self._matchers.items():
            matcher.delete(field)

    def _preprocess(self, value: str) -> str:
        """Preprocess text values."""
        functions = [
            str.lower,
            self._string_normalize,
            self._strip_puntuation,
            self._clean_whitespace,
            str.strip,
        ]

        for function in functions:
            value = function(value)

        return value

    @staticmethod
    def _clean_whitespace(value: str) -> str:
        """Replace whitespace for a single space."""
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _string_normalize(value: str) -> str:
        """Normalize special unicode characters."""
        return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()

    @staticmethod
    def _strip_puntuation(value: str) -> str:
        """Replace punctuation by whitespace."""
        return "".join(c if c not in string.punctuation else " " for c in value)

    @staticmethod
    def _make_id() -> str:
        """Generate a UUID4 identifier."""
        return uuid.uuid4().hex
