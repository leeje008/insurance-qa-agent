"""검색 결과 로깅 테이블 추가 (리랭킹 전후 점수 추적).

Revision ID: 002
Revises: 001
Create Date: 2026-03-25
"""

import sqlalchemy as sa

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("qa_log_id", sa.Integer, sa.ForeignKey("qa_logs.id"), nullable=False),
        sa.Column("article_id", sa.Integer, sa.ForeignKey("policy_articles.id"), nullable=False),
        sa.Column("rank_before_rerank", sa.Integer),
        sa.Column("rank_after_rerank", sa.Integer, nullable=True),
        sa.Column("semantic_score", sa.Float, nullable=True),
        sa.Column("keyword_score", sa.Float, nullable=True),
        sa.Column("rrf_score", sa.Float, nullable=True),
        sa.Column("rerank_score", sa.Float, nullable=True),
        sa.Column("rerank_strategy", sa.String(50), nullable=True),
        sa.Column("was_used_in_answer", sa.Boolean, default=False),
    )
    op.create_index("ix_retrieval_logs_qa_log_id", "retrieval_logs", ["qa_log_id"])
    op.create_index("ix_retrieval_logs_article_id", "retrieval_logs", ["article_id"])


def downgrade() -> None:
    op.drop_index("ix_retrieval_logs_article_id", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_qa_log_id", table_name="retrieval_logs")
    op.drop_table("retrieval_logs")
