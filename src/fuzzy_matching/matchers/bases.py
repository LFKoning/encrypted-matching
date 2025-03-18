"""Module with base class for matchers using string values."""

import re
import string
import unicodedata

import pandas as pd


class BaseMatcher:
    """Base class for matchers."""

    def _flatten(self, items):
        for item in items:
            if isinstance(item, (list, tuple)):
                yield from self._flatten(item)
            else:
                yield item

    def _group_ids(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Group identifiers that share the same value."""
        return data.groupby(field, as_index=False).agg(
            lambda ids: list(self._flatten(ids))
        )

    def _ungroup_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        """Explode grouped identifiers into rows."""
        return data.explode(column="id")


class StringMixin:
    """Base class for matchers using string values."""

    def _preprocess(self, value: str) -> str:
        """Preprocess string values."""
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
