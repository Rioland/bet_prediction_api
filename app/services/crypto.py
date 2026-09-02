import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet

from app.core.config import settings

# A single application-level key, so the salt is fixed; the iteration count is
# what makes a low-entropy passphrase expensive to attack.
_KDF_SALT = b"football-ai-settings-v1"
_KDF_ITERATIONS = 600_000


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    raw = settings.settings_encryption_key.encode("utf-8")
    key = hashlib.pbkdf2_hmac("sha256", raw, _KDF_SALT, _KDF_ITERATIONS, dklen=32)
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str) -> str:
    return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")
