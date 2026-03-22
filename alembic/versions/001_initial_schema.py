"""initial schema - 6 tables with pgvector.

Revision ID: 001
Revises:
Create Date: 2026-03-22
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers
revision = "001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    # pgvector 확장 활성화
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. insurance_products
    op.create_table(
        "insurance_products",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, comment="상품명"),
        sa.Column("insurance_type", sa.String(50), nullable=False, comment="보험 유형"),
        sa.Column("insurer", sa.String(100), nullable=False, comment="보험사"),
        sa.Column("version", sa.String(50), nullable=True, comment="약관 버전"),
        sa.Column("effective_date", sa.Date(), nullable=True, comment="시행일"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # 2. policy_articles
    op.create_table(
        "policy_articles",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "product_id",
            sa.Integer(),
            sa.ForeignKey("insurance_products.id"),
            nullable=False,
        ),
        sa.Column("level", sa.String(20), nullable=False, comment="계층: part/article/..."),
        sa.Column("number", sa.String(20), nullable=False, comment="조항 번호 (예: 제1조)"),
        sa.Column("title", sa.String(300), nullable=True, comment="조항 제목"),
        sa.Column("content", sa.Text(), nullable=False, comment="조항 본문"),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("policy_articles.id"),
            nullable=True,
            comment="상위 조항 ID (자기참조)",
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True, comment="조항 임베딩 벡터"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # 3. article_references
    op.create_table(
        "article_references",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("policy_articles.id"),
            nullable=False,
            comment="참조하는 조항",
        ),
        sa.Column(
            "target_id",
            sa.Integer(),
            sa.ForeignKey("policy_articles.id"),
            nullable=False,
            comment="참조되는 조항",
        ),
    )

    # 4. article_sub_chunks
    op.create_table(
        "article_sub_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column(
            "article_id",
            sa.Integer(),
            sa.ForeignKey("policy_articles.id"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False, comment="청크 순서"),
        sa.Column("content", sa.Text(), nullable=False, comment="청크 본문"),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True, comment="청크 임베딩 벡터"),
    )

    # 5. insurance_glossary
    op.create_table(
        "insurance_glossary",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("term", sa.String(200), nullable=False, comment="용어"),
        sa.Column("definition", sa.Text(), nullable=False, comment="정의"),
        sa.Column(
            "product_id",
            sa.Integer(),
            sa.ForeignKey("insurance_products.id"),
            nullable=True,
            comment="특정 상품 전용 (NULL=공통)",
        ),
    )

    # 6. qa_logs
    op.create_table(
        "qa_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("question", sa.Text(), nullable=False, comment="사용자 질문"),
        sa.Column("answer", sa.Text(), nullable=False, comment="생성된 답변"),
        sa.Column("sources_json", sa.Text(), nullable=True, comment="참조 조항 JSON"),
        sa.Column("confidence", sa.Float(), nullable=True, comment="신뢰도 점수"),
        sa.Column("feedback_score", sa.Integer(), nullable=True, comment="사용자 피드백 (1-5)"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # 인덱스
    op.create_index("ix_policy_articles_product_id", "policy_articles", ["product_id"])
    op.create_index("ix_policy_articles_number", "policy_articles", ["number"])
    op.create_index("ix_article_sub_chunks_article_id", "article_sub_chunks", ["article_id"])
    op.create_index("ix_insurance_glossary_product_id", "insurance_glossary", ["product_id"])
    op.create_index("ix_qa_logs_created_at", "qa_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("qa_logs")
    op.drop_table("insurance_glossary")
    op.drop_table("article_sub_chunks")
    op.drop_table("article_references")
    op.drop_table("policy_articles")
    op.drop_table("insurance_products")
    op.execute("DROP EXTENSION IF EXISTS vector")
