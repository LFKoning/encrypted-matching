"""Module for fuzzy matching using cosine similarity between vectors."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fuzzy_matching.storage import VectorStore

from .bases import BaseMatcher, StringMixin


class VectorMatcher(BaseMatcher, StringMixin):
    """Fuzzy matching using cosine similarity between vectors.

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
        """Add entities to the matching set.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with entities to add to the matching set.
        """
        # Perform basic data preprocessing.
        data = data.assign(**{self._field: data[self._field].map(self._preprocess)})

        # Store values with UUIDs and name.
        existing = self._storage.load()
        data = pd.concat([existing, data])
        self._storage.store(data)

        # Convert to vectors and store.
        vectors = self._vectorizer.fit_transform(data[self._field])
        self._vector_storage.store(vectors)

    def get(self, target: str) -> pd.DataFrame:
        """Return all entities and their similarity to the target.

        Parameters
        ----------
        target : str
            Target string to match against.

        Returns
        -------
        pandas.DataFrame
            DataFrame of entities and their similarity scores.
        """
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
