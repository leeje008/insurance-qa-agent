"""api.db.models - ORM 모델 정의 (6 테이블)."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.database import Base
from core.constants import EMBEDDING_DIM


class InsuranceProduct(Base):
    """보험 상품 테이블."""

    __tablename__ = "insurance_products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="상품명")
    insurance_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="보험 유형")
    insurer: Mapped[str] = mapped_column(String(100), nullable=False, comment="보험사")
    version: Mapped[str] = mapped_column(String(50), nullable=True, comment="약관 버전")
    effective_date: Mapped[date | None] = mapped_column(comment="시행일")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    articles: Mapped[list["PolicyArticle"]] = relationship(back_populates="product")
    glossary_terms: Mapped[list["InsuranceGlossary"]] = relationship(back_populates="product")


class PolicyArticle(Base):
    """약관 조항 테이블 (관/조/항/호 계층 구조)."""

    __tablename__ = "policy_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("insurance_products.id"), nullable=False)
    level: Mapped[str] = mapped_column(String(20), nullable=False, comment="계층: part/article/...")
    number: Mapped[str] = mapped_column(String(20), nullable=False, comment="조항 번호 (예: 제1조)")
    title: Mapped[str | None] = mapped_column(String(300), comment="조항 제목")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="조항 본문")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("policy_articles.id"), comment="상위 조항 ID (자기참조)"
    )
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True, comment="조항 임베딩 벡터")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    product: Mapped["InsuranceProduct"] = relationship(back_populates="articles")
    parent: Mapped["PolicyArticle | None"] = relationship(
        remote_side="PolicyArticle.id", back_populates="children"
    )
    children: Mapped[list["PolicyArticle"]] = relationship(back_populates="parent")
    sub_chunks: Mapped[list["ArticleSubChunk"]] = relationship(back_populates="article")
    source_references: Mapped[list["ArticleReference"]] = relationship(
        foreign_keys="ArticleReference.source_id", back_populates="source"
    )
    target_references: Mapped[list["ArticleReference"]] = relationship(
        foreign_keys="ArticleReference.target_id", back_populates="target"
    )


class ArticleReference(Base):
    """조항 간 참조 관계 테이블."""

    __tablename__ = "article_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("policy_articles.id"), nullable=False, comment="참조하는 조항"
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("policy_articles.id"), nullable=False, comment="참조되는 조항"
    )

    source: Mapped["PolicyArticle"] = relationship(
        foreign_keys=[source_id], back_populates="source_references"
    )
    target: Mapped["PolicyArticle"] = relationship(
        foreign_keys=[target_id], back_populates="target_references"
    )


class ArticleSubChunk(Base):
    """조항 하위 청크 테이블 (긴 조항을 분할)."""

    __tablename__ = "article_sub_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("policy_articles.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="청크 순서")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="청크 본문")
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True, comment="청크 임베딩 벡터")

    article: Mapped["PolicyArticle"] = relationship(back_populates="sub_chunks")


class InsuranceGlossary(Base):
    """보험 용어 사전 테이블."""

    __tablename__ = "insurance_glossary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    term: Mapped[str] = mapped_column(String(200), nullable=False, comment="용어")
    definition: Mapped[str] = mapped_column(Text, nullable=False, comment="정의")
    product_id: Mapped[int | None] = mapped_column(
        ForeignKey("insurance_products.id"), nullable=True, comment="특정 상품 전용 (NULL=공통)"
    )

    product: Mapped["InsuranceProduct | None"] = relationship(back_populates="glossary_terms")


class QALog(Base):
    """Q&A 로그 테이블."""

    __tablename__ = "qa_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="사용자 질문")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="생성된 답변")
    sources_json: Mapped[str | None] = mapped_column(Text, comment="참조 조항 JSON")
    confidence: Mapped[float | None] = mapped_column(Float, comment="신뢰도 점수")
    feedback_score: Mapped[int | None] = mapped_column(Integer, comment="사용자 피드백 (1-5)")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
