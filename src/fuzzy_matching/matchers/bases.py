"""Module with base classes for matchering algoritms."""

import re
import string
import unicodedata
from pathlib import Path

from fuzzy_matching.storage import EncryptedStore


class BaseMatcher:
    """Base class for matching algoritms.

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
    ) -> None:
        self._field = field
        self._settings = settings or {}
        self._weight = settings.get("weight", 1.0)

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
        """Preprocess string values.

        Preprocessing steps:

        - Convert to lowercase.
        - Normalize special characters.
        - Replace punctuation characters with spaces.
        - Remove excess whitespace.

        Parameters
        ----------
        values : str
            String value to preprocess.

        Returns
        -------
        str
            Preprocessed string value.
        """
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
        """Replace all whitespace with a single space.

        Parameters
        ----------
        values : str
            String value to process.

        Returns
        -------
        str
            Processed string value.
        """
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _string_normalize(value: str) -> str:
        """Normalize special unicode characters.

        Parameters
        ----------
        values : str
            String value to process.

        Returns
        -------
        str
            Processed string value.
        """
        return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()

    @staticmethod
    def _strip_puntuation(value: str) -> str:
        """Replace punctuation characters with whitespace."

        Parameters
        ----------
        values : str
            String value to process.

        Returns
        -------
        str
            Processed string value.
        """
        return "".join(c if c not in string.punctuation else " " for c in value)
