from enum import Enum


class ResponseImageFormat(str, Enum):
    """Image formats for API responses."""

    URL = "url"
    BASE64 = "base64"
