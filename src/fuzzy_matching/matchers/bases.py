"""Module with base class for matchers using string values."""

import re
import string
import unicodedata
from pathlib import Path

from fuzzy_matching.storage import EncryptedStore


class BaseMatcher:
    """Base class for matchers."""

    def __init__(
        self,
        field: str,
        encryption_key: bytes,
        storage_path: Path,
        settings: dict = None,
    ) -> None:
        self._field = field
        self._weight = settings.get("weight", 1.0)
        self._settings = settings or {}

        storage_path = storage_path / self._make_filename()
        self._storage = EncryptedStore(encryption_key, storage_path)

    def _make_filename(self, extension="dat") -> str:
        """Create a filename from a field name."""
        field = self._field.lower().strip().replace(" ", "_")
        field = "".join(
            [char for char in field if char in string.ascii_lowercase + "_-"]
        )

        return f"{self.__class__.__name__.lower()}_{field}.{extension}"


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
