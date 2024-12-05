"""Module for fuzzy matching large sets of encypted data."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EncryptedMatcher:
    """Fuzzy matching for large sets of encrypted data."""

    ENCODING = "utf8"

    def __init__(
        self,
        encryption_key,
        storage_path: str,
        topn: int = 10,
        n_features: int = 2**20,
    ):
        self._encryptor = Fernet(encryption_key)
        self._topn = -topn

        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(exist_ok=True)
        self._database = self._setup_database()
        self._vectors = self._load_vectors()

        self._vectorizer = HashingVectorizer(
            encoding=self.ENCODING,
            n_features=n_features,
            ngram_range=(3, 3),
            analyzer="char_wb",
            lowercase=True,
            strip_accents="ascii",
        )

    def add_names(self, names: pd.Series):
        """Store encrypted and vectorized names."""
        self._store_data(names)
        vectors = self._vectorizer.fit_transform(names)
        self._store_vectors(vectors)

    def search(self, targets: str):
        """Search names in the vector space."""
        target_vectors = self._vectorizer.fit_transform(targets)

        similarities = cosine_similarity(target_vectors, self._vectors)
        top_matches = np.argpartition(similarities, self._topn)[:, self._topn :]

        results = []
        for target_index, matched_indices in enumerate(top_matches):
            matched_rows = self._search_data(matched_indices)
            for match_index, match_data in matched_rows:
                results.append(
                    {
                        "name": targets[target_index],
                        "target": self._decrypt_data(match_data),
                        "encrypted": match_data,
                        "similarity": float(similarities[target_index, match_index]),
                    }
                )

        results = sorted(results, key=lambda m: m["similarity"], reverse=True)
        return results

    def _search_data(self, indices):
        """Retrieve encrypted data from SQLite."""
        query = """
            SELECT
                MatchIndex - 1 AS MatchedIndex,
                MatchData
            FROM EncryptedData
            WHERE MatchIndex IN (%s)
        """
        query = query % ", ".join("?" * len(indices))

        # MatchIndex starts at 1 and need to convert to list.
        indices = [int(i + 1) for i in indices]
        result = self._database.execute(query, indices)
        return result.fetchall()

    def _load_vectors(self):
        """Load stored vectors."""
        try:
            return sparse.load_npz(self._storage_path / "vectors.npz")
        except FileNotFoundError:
            return None

    def _store_data(self, names):
        """Store encrypted data to SQLite."""
        encrypted = names.map(self._encrypt_data).values.reshape(-1, 1)
        self._database.executemany(
            "INSERT INTO EncryptedData (MatchData) VALUES (?)", encrypted
        )
        self._database.commit()

    def _store_vectors(self, vectors):
        """Store vectors to disk."""
        if self._vectors is not None:
            self._vectors = sparse.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors
        sparse.save_npz(self._storage_path / "vectors.npz", vectors)

    def _encrypt_data(self, data):
        """Encrypts data using Fernet encryption."""
        return self._encryptor.encrypt(data.encode(self.ENCODING))

    def _decrypt_data(self, data):
        """Decrypts data using Fernet encryption."""
        # Decode to string or leave as bytes?
        return self._encryptor.decrypt(data).decode(self.ENCODING)

    def _setup_database(self):
        """Sets up the SQLite database."""
        database = sqlite3.connect(self._storage_path / "encrypted.db")
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS EncryptedData (
                MatchIndex INTEGER PRIMARY KEY,
                MatchData BLOB
            );
            """
        )
        database.commit()
        return database
