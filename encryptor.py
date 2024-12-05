"""Module with encryption helper classes."""

from secrets import token_bytes

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESGCMSIV


class BaseEncryptor:
    """Encryptor base class."""

    _encrypt_class = NotImplemented

    def __init__(self, key: bytes, encoding="utf8") -> None:
        # pylint: disable=not-callable
        self._encryptor = self._encrypt_class(key)
        self._encoding = encoding

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a compatible encryption key."""
        # pylint: disable=no-member
        return cls._encrypt_class.generate_key()

    def encrypt(self, value: str) -> bytes:
        """Encrypt a series of values."""
        return value.encode(self._encoding)

    def decrypt(self, value: bytes) -> str:
        """Decrypt a series of values."""
        return value.decode(self._encoding)


class FernetEncryptor(BaseEncryptor):
    """Class for encrypting with Fernet encryption.

    Parameters:
    -----------
    key : bytes
        Fernet compatible encryption key.
    """

    _encrypt_class = Fernet

    def encrypt(self, value: str) -> bytes:
        """Encrypt a value using Fernet encryption."""
        return self._encryptor.encrypt(value.encode(self._encoding))

    def decrypt(self, value: bytes) -> str:
        """Encrypt a value using Fernet encryption."""
        return self._encryptor.decrypt(value).decode(self._encoding)


class AESGCM4Encryptor(BaseEncryptor):
    """Class for encrypting with AES-GCM IV.

    AES-GCM IV is slower, but more robust against nonce reuse.

    Parameters
    ----------
    key : bytes
        An AES-GCM compatible encryption key.
    """

    _encrypt_class = AESGCMSIV

    @classmethod
    def generate_key(cls, bit_length: int = 128) -> bytes:
        """Generate an AES-GCM compatitble encryption key.

        Parameters
        ----------
        bit_length : int, default=128
            Length of the key in bytes.

        Returns
        -------
        bytes
            The generated AES-GCM key.
        """
        return AESGCM.generate_key(bit_length=bit_length)

    def encrypt(self, value: str) -> bytes:
        """Encrypt a string value using AES-GCM IV encryption."""
        # Generate a 12 byte nonce.
        # Not garantueed to be unique!
        nonce = token_bytes(12)
        cypher = self._encryptor.encrypt(nonce, value.encode(self._encoding), b"")
        return nonce + cypher

    def decrypt(self, value: bytes) -> str:
        """Decrypt a string value using AES-GCM IV encryption."""
        # Get the nonce and then decrypt.
        nonce = value[0:12]
        return self._encryptor.decrypt(nonce, value[12:], b"").decode(self._encoding)
