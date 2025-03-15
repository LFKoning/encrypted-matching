"""Test retrieving a record."""

from fuzzy_matching.match_multi import MultiMatcher

KEY = b"\x0e\x84\xa1\x01\xd0\xed\x932\xb5\x1dt\x11\x05\xe5j\xf8"
STORAGE = "storage"
CONFIG = {
    "name": {
        "algoritm": "vector",
        "weight": 0.2,
    },
    "birthdate": {
        "algoritm": "distance",
        "weight": 0.2,
    },
    "national_id": {"algoritm": "distance", "weight": 0.6},
}


def main():
    """Main program routine."""
    matcher = MultiMatcher(10, CONFIG, KEY, STORAGE)
    results = matcher.get(
        {
            "name": "Weijters, Hanna",
            "birthdate": "16-11-2000",
            "national_id": "nld6625447m7080511883201356",
        }
    )
    print(results)


if __name__ == "__main__":
    main()
