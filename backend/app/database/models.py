from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.database import Base
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(255), unique=True)
    file_type: Mapped[str] = mapped_column(String(10))
    text: Mapped[str] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    analyses_a: Mapped[list] = relationship("Analysis", foreign_keys="Analysis.document_a_id", back_populates="document_a", cascade="all, delete-orphan")
    analyses_b: Mapped[list] = relationship("Analysis", foreign_keys="Analysis.document_b_id", back_populates="document_b", cascade="all, delete-orphan")
class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_a_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document_b_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    similarity: Mapped[float] = mapped_column(Float)
    matches_json: Mapped[str] = mapped_column(Text)
    audit_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    document_a: Mapped[Document] = relationship(foreign_keys=[document_a_id], back_populates="analyses_a")
    document_b: Mapped[Document] = relationship(foreign_keys=[document_b_id], back_populates="analyses_b")
