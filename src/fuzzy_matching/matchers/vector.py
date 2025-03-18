"""Module for fuzzy matching using cosine similarity between vectors."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fuzzy_matching.storage import VectorStore

from .bases import BaseMatcher, StringMixin


class VectorMatcher(BaseMatcher, StringMixin):
    """Fuzzy matching using cosine similarity between vectors."""

    def __init__(
        self,
        field: str,
        encryption_key: bytes,
        storage_path: Path,
        settings: dict = None,
    ):
        super().__init__(field, encryption_key, storage_path, settings)

        storage_path = storage_path / self._make_filename("npz")
        self._vector_storage = VectorStore(storage_path)

        self._vectorizer = HashingVectorizer(
            encoding="utf8",
            n_features=2**10,
            ngram_range=(3, 3),
            analyzer="char_wb",
        )

    def create(self, data: pd.DataFrame) -> None:
        """Store encrypted and vectorized values."""
        # Perform basic data preprocessing.
        data = data.assign(**{self._field: data[self._field].map(self._preprocess)})

        # Store values with UUIDs and name.
        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

        # Convert to vectors and store.
        vectors = self._vectorizer.fit_transform(data[self._field])
        self._vector_storage.store(vectors)

    def get(self, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Search values in the vector space."""
        # Load the encrypted values.
        data = self._storage.load()
        if data is None:
            return None

        target = self._preprocess(target)

        # Compute vector similarities.
        target_vector = self._vectorizer.fit_transform([target])
        vectors = self._vector_storage.load()
        similarities = cosine_similarity(target_vector, vectors)[0]

        data = data.assign(**{f"similarity_{self._field}": similarities * self._weight})
        return data.set_index("id")

    def delete(self) -> None:
        """Delete all matching data for the field."""
        self._storage.delete()
        self._vector_storage.delete()
