"""Module for fuzzy matching using cosine similarity between vectors."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fuzzy_matching.storage import EncryptedStore, VectorStore


class VectorMatcher:
    """Fuzzy matching using cosine similarity between vectors."""

    def __init__(
        self,
        field,
        encryption_key: bytes,
        storage_path: Path,
    ):
        self._field = field
        self._vector_store = VectorStore(field, storage_path)
        self._value_store = EncryptedStore(field, encryption_key, storage_path)

        self._vectorizer = HashingVectorizer(
            encoding="utf8",
            n_features=2**10,
            ngram_range=(3, 3),
            analyzer="char_wb",
        )

    def create(self, uuids: pd.Series, values: pd.Series) -> None:
        """Store encrypted and vectorized names."""
        # Store values
        values.index = uuids
        values.name = self._field
        self._value_store.store(values)

        # Convert to vectors and store.
        vectors = self._vectorizer.fit_transform(values)
        self._vector_store.store(vectors)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Search names in the vector space."""
        # Load the encrypted values.
        values = self._value_store.retrieve()
        if values is None:
            return None, None

        # Compute vector similarities.
        target_vector = self._vectorizer.fit_transform([target])
        vectors = self._vector_store.retrieve()
        similarities = cosine_similarity(target_vector, vectors)[0]
        similarities = pd.Series(similarities, index=values.index)

        return values, similarities

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._vector_store.delete()
        self._value_store.delete()
