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
        return f"storage_{field}.{extension}"


class EncryptedStore(Storage):
    """Class for encrypted value storage."""

    def __init__(self, field: str, encryption_key: bytes, storage_path: Path) -> None:
        self._storage_file = storage_path / self._make_filename(field)
        self._encryptor = AESGCM4Encryptor(encryption_key)

    def store(self, data: pd.Series | pd.DataFrame) -> None:
        """Encrypt and store pandas data structures."""
        byte_data = io.BytesIO()
        data.to_pickle(byte_data)

        byte_data = self._encryptor.encrypt(byte_data.getbuffer())

        with open(self._storage_file, "wb") as data_file:
            data_file.write(byte_data)

    def delete(self) -> None:
        """Delete all stored data."""
        try:
            self._storage_file.unlink()
        except FileNotFoundError:
            pass

    def load(self) -> pd.Series | pd.DataFrame | None:
        """Load and decrypt pandas data structures."""
        try:
            with open(self._storage_file, "rb") as data_file:
                raw_data = data_file.read()

            raw_data = self._encryptor.decrypt(raw_data)

            raw_data = io.BytesIO(raw_data)
            return pd.read_pickle(raw_data)

        except FileNotFoundError:
            print(f"Warning: Cannot find file {self._storage_file}")
            return None


class VectorStore(Storage):
    """Class for storing sparse vector matrices."""

    def __init__(self, field: str, storage_path: Path) -> None:
        self._storage_file = storage_path / self._make_filename(field, "npz")

    def store(self, vectors):
        """Store vectors to disk."""
        sparse.save_npz(self._storage_file, vectors)

    def delete(self) -> None:
        """Delete all stored data."""
        try:
            self._storage_file.unlink()
        except FileNotFoundError:
            pass

    def load(self) -> sparse.csr_matrix | None:
        """Load vectors from disk."""
        try:
            return sparse.load_npz(self._storage_file)
        except FileNotFoundError:
            print(f"Warning: Cannot find file {self._storage_file}")
            return None
