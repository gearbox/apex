from enum import StrEnum


class ResponseImageFormat(StrEnum):
    """Image formats for API responses."""

    URL = "url"
    BASE64 = "base64"
