from typing import IO, Any

import fsspec

from vera.config.settings import settings
from vera.ports.blob import BlobStore


class FsspecBlobStore(BlobStore):
    """BlobStore implemented with fsspec to uniformly support various filesystems.

    Currently using file:// for prototyping, with plans to use gcs:// in the future.

    base_uri: blob storage location
    prefix:
    key: relative path of object, stable across storage backends
    """

    def __init__(self, base_uri: str, prefix: str):
        self._base_uri = base_uri.rstrip("/")
        self._prefix = prefix.strip("/")

    def _url(self, key: str) -> str:
        """
        Url constructed like <base_uri> / <prefix> / <key>
            file:///foo/bar/vera/.data / vera / papers/PMC12345/jats/article.xml
        """
        key = key.lstrip("/")
        if self._prefix:
            key = f"{self._prefix}/{key}"
        return f"{self._base_uri}/{key}"

    def open(self, key: str, mode: str = "rb") -> IO[Any]:
        return fsspec.open(self._url(key), mode).open()

    def exists(self, key: str) -> bool:
        fs, path = fsspec.core.url_to_fs(self._url(key))
        return fs.exists(path)

    def put_bytes(self, key: str, data: bytes) -> None:
        with self.open(key, "wb") as f:
            f.write(data)


blobstore: BlobStore = FsspecBlobStore(settings.blob_base_uri, settings.blob_prefix)
