"""Module for encrypted value storage"""

import io
import string
from pathlib import Path

import pandas as pd
from scipy import sparse

from fuzzy_matching.encryption import AESGCM4Encryptor


class Storage:
    """Storage base class."""

    @staticmethod
    def _make_filename(field, extension="dat") -> str:
        """Create a filename from a field name."""
        field = field.lower().strip().replace(" ", "_")
        field = "".join(
            [char for char in field if char in string.ascii_lowercase + "_-"]
        )
        return f"{field}.{extension}"


class EncryptedStore(Storage):
    """Class for encrypted value storage."""

    def __init__(self, field: str, encryption_key: bytes, storage_path: Path) -> None:
        self._storage_file = storage_path / self._make_filename(field)
        self._encryptor = AESGCM4Encryptor(encryption_key)
        self._data = self._load()

    def retrieve(self) -> pd.DataFrame | None:
        """Return data as a DataFrame."""
        return self._data

    def store(self, values: pd.DataFrame) -> None:
        """Encrypt and store the data."""
        if self._data is None:
            self._data = values
        else:
            self._data = pd.concat([self._data, values])

        data = io.BytesIO()
        self._data.to_pickle(data)

        data = self._encryptor.encrypt(data.getbuffer())

        with open(self._storage_file, "wb") as data_file:
            data_file.write(data)

    def _load(self) -> pd.DataFrame | None:
        """Load and decrypt the data."""
        try:
            with open(self._storage_file, "rb") as data_file:
                raw_data = io.BytesIO(data_file.read())

            raw_data = self._encryptor.decrypt(raw_data)

            return pd.read_pickle(raw_data)

        except FileNotFoundError:
            return None


class VectorStore(Storage):
    """Class for storing sparse vector matrices."""

    def __init__(self, field: str, storage_path: Path) -> None:
        self._storage_file = storage_path / self._make_filename(field, "npz")
        self._data = self._load()

    def retrieve(self) -> sparse.csr_matrix | None:
        """Return vectors as a sparse matrix."""
        return self._data

    def store(self, vectors):
        """Store vectors to disk."""
        if self._data is not None:
            self._data = sparse.vstack([self._data, vectors])
        else:
            self._data = vectors
        sparse.save_npz(self._storage_file, self._data)

    def _load(self) -> sparse.csr_matrix | None:
        """Load vectors from disk."""
        try:
            return sparse.load_npz(self._storage_file)
        except FileNotFoundError:
            return None
