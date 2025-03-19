"""Module for encrypted storage of pandas data structures."""

import io
from pathlib import Path

import pandas as pd
from scipy import sparse

from fuzzy_matching.encryption import AESGCM4Encryptor


class EncryptedStore:
    """Class for encrypted storage of pandas data structures.

    Parameters
    ----------
    encryption_key : bytes
        Encryption key for storing data, provided as bytes.
    storage_path : str, default="storage"
        Folder to store data in.
    """

    def __init__(self, encryption_key: bytes, storage_path: Path) -> None:
        self._data = None
        self._storage_path = storage_path
        self._encryptor = AESGCM4Encryptor(encryption_key)

    def store(self, data: pd.Series | pd.DataFrame) -> None:
        """Encrypt and store a pandas data structure.

        Parameters
        ----------
        data : pandas.DataFrame or pandas.Series
            Data structure to store to file.
        """
        self._data = data

        byte_data = io.BytesIO()
        data.to_pickle(byte_data)

        byte_data = self._encryptor.encrypt(byte_data.getbuffer())

        with open(self._storage_path, "wb") as data_file:
            data_file.write(byte_data)

    def load(self) -> pd.Series | pd.DataFrame | None:
        """Load and decrypt a pandas data structure.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            The decrypted data structure.
        """
        if self._data is not None:
            return self._data

        try:
            with open(self._storage_path, "rb") as data_file:
                raw_data = data_file.read()

            raw_data = self._encryptor.decrypt(raw_data)

            raw_data = io.BytesIO(raw_data)
            self._data = pd.read_pickle(raw_data)
            return self._data

        except FileNotFoundError:
            print(f"Warning: Cannot find file: {self._storage_path}")
            return None

    def delete(self) -> None:
        """Delete all stored data."""
        try:
            self._storage_path.unlink()
        except FileNotFoundError:
            pass
        self._data = None


class VectorStore:
    """Class for storing sparse vector matrices.

    Parameters
    ----------
    storage_path : pathlib.Path
        Path to a file to store the data in.
    """

    def __init__(self, storage_path: Path) -> None:
        self._vectors = None
        self._storage_path = storage_path

    def store(self, vectors: sparse.csr_matrix) -> None:
        """Store vectors to disk.

        Parameters
        ----------
        vectors : scipy.sparse.csr_matrix
            Sparse matrix of vectors.
        """
        sparse.save_npz(self._storage_path, vectors)

    def load(self) -> sparse.csr_matrix | None:
        """Load vectors from disk.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of vectors.
        """
        if self._vectors is not None:
            return self._vectors

        try:
            self._vectors = sparse.load_npz(self._storage_path)
            return self._vectors

        except FileNotFoundError:
            print(f"Warning: Cannot find file: {self._storage_path}")
            return None

    def delete(self) -> None:
        """Delete all stored vectors."""
        try:
            self._storage_path.unlink()
        except FileNotFoundError:
            pass
        self._vectors = None
