"""Generate a VAPID key pair for Web Push notifications.

Usage:
    uv run python tools/generate_vapid_keys.py

Prints base64url-encoded public/private keys ready to paste into
VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY env vars. Never commit the private key —
this script only prints to stdout, it never writes a file.

Keys are raw EC (P-256 / secp256r1) values in the format Web Push expects:
  - public:  uncompressed point (0x04 || X || Y), 65 bytes
  - private: raw scalar, 32 bytes
"""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric import ec


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def main() -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_numbers = private_key.public_key().public_numbers()
    private_value = private_key.private_numbers().private_value

    public_raw = (
        b"\x04" + public_numbers.x.to_bytes(32, "big") + public_numbers.y.to_bytes(32, "big")
    )
    private_raw = private_value.to_bytes(32, "big")

    print(f"VAPID_PUBLIC_KEY={_b64url(public_raw)}")
    print(f"VAPID_PRIVATE_KEY={_b64url(private_raw)}")
    print("VAPID_SUBJECT=mailto:ops@apex.ai  # replace with a real contact")


if __name__ == "__main__":
    main()
