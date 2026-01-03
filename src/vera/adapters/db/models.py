import uuid
from datetime import datetime, date

from sqlalchemy import Column, DateTime, Integer, UniqueConstraint, func
from sqlmodel import SQLModel, Field


def uuid_pk() -> uuid.UUID:
    return uuid.uuid4()


def created_at_column() -> Column:
    return Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: uuid.UUID = Field(default_factory=uuid_pk, primary_key=True)
    kind: str = Field(index=True)  # ingest_jats, chunk_v1, etc
    status: str = "success"
    detail: str | None = None

    created_at: datetime = Field(sa_column=created_at_column())


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid_pk, primary_key=True)
    source: str = Field(index=True)  # pmc, arxiv, biorxiv, etc

    title: str | None = None
    journal: str | None = None
    published_date: date | None = None
    license: str | None = None

    created_at: datetime = Field(sa_column=created_at_column())


class DocumentIdentifier(SQLModel, table=True):
    __tablename__ = "document_identifiers"
    __table_args__ = (UniqueConstraint("scheme", "value", name="uq_identifier_scheme_value"),)

    id: uuid.UUID = Field(default_factory=uuid_pk, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)

    scheme: str = Field(index=True)  # pmcid, doi, arxiv, etc
    value: str = Field(index=True)  # PMC123456, 10.1101/..., 2401.01234, etc

    created_at: datetime = Field(sa_column=created_at_column())


class Artifact(SQLModel, table=True):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("blob_key", name="uq_artifacts_blob_key"),)

    id: uuid.UUID = Field(default_factory=uuid_pk, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)

    blob_key: str = Field(index=True)  # key within blob store
    kind: str = Field(index=True)  # jats_xml, pdf, extracted_json, etc

    mime_type: str | None = None
    sha256: str | None = Field(default=None, index=True)
    bytes: int | None = None

    created_at: datetime = Field(sa_column=created_at_column())


class TextView(SQLModel, table=True):
    __tablename__ = "text_views"

    id: uuid.UUID = Field(default_factory=uuid_pk, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    artifact_id: uuid.UUID = Field(foreign_key="artifacts.id", index=True)

    view_kind: str = Field(index=True)  # jats_body, jats_captions, bioc, etc

    body_word_count: int = 0
    paragraph_count: int = 0
    is_primary: bool = False

    extractor_version: str = Field(default="v1", index=True)

    created_at: datetime = Field(sa_column=created_at_column())


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "text_view_id",
            "chunk_version",
            "ordinal",
            name="uq_chunk_view_ver_ord",
        ),
    )

    id: str = Field(primary_key=True)  # deterministic hash
    text_view_id: uuid.UUID = Field(foreign_key="text_views.id", index=True)

    ordinal: int = Field(default=0, sa_column=Column(Integer, nullable=False))
    section: str | None = None
    text: str

    chunk_version: str = Field(default="v1", index=True)
    created_at: datetime = Field(sa_column=created_at_column())
