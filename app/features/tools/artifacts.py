from datetime import timedelta

from app.contracts.channels import ArtifactKind, ReplyArtifact
from app.features.blobs.contracts import BlobServiceContract


# NOTE: a generated file is stored before the reader sees it, so a channel that hands
# out links can name it at once; it lapses on `ttl` unless a recorded turn adopts it
class ArtifactWriter:
    blobs: BlobServiceContract
    ttl: timedelta

    def __init__(self, blobs: BlobServiceContract, ttl: timedelta) -> None:
        self.blobs = blobs
        self.ttl = ttl

    async def write(
        self,
        mime: str,
        data: bytes,
        *,
        kind: ArtifactKind,
        name: str,
    ) -> ReplyArtifact:
        blob = await self.blobs.save(
            mime,
            data,
            kind=kind,
            filename=name,
            ttl=self.ttl,
        )
        return ReplyArtifact(
            blob=str(blob.id),
            kind=kind,
            mime=mime,
            size=len(data),
            data=data,
            name=name,
        )
