# Insurance QA Agent - 프로젝트 기획서 요약

## 개요

보험약관 Q&A 에이전트 — RAG(Retrieval-Augmented Generation) 기반 보험약관 질의응답 시스템

## 핵심 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | FastAPI + Gunicorn |
| LLM | Ollama (qwen2.5:14b / 7b) |
| Embedding | nomic-embed-text (768 dim) |
| DB | PostgreSQL 16 + PGVector |
| 검색 | Hybrid Search (Semantic + Keyword + RRF) |
| Pipeline | LangGraph 4-Node (query_processor → retriever → answer_generator → answer_validator) |
| PDF 파싱 | PyMuPDF + pdfplumber |
| Frontend | Streamlit (MVP) |

## DB 테이블 (6개)

1. **insurance_products** — 보험 상품
2. **policy_articles** — 약관 조항 (관/조/항/호 계층, 임베딩 벡터)
3. **article_references** — 조항 간 참조 관계
4. **article_sub_chunks** — 긴 조항 분할 청크 (임베딩 벡터)
5. **insurance_glossary** — 보험 용어 사전
6. **qa_logs** — Q&A 로그 (질문, 답변, 신뢰도, 피드백)

## RAG 파이프라인

1. **Query Processor** — 질문 분석, 키워드/의도 추출
2. **Retriever** — Hybrid Search (PGVector cosine + 키워드 매칭 + RRF 융합)
3. **Answer Generator** — 검색 결과 기반 답변 생성 (조항 인용 포함)
4. **Answer Validator** — 환각 검증, 신뢰도 평가 (실패 시 재생성 루프, 최대 2회)

## 개발 로드맵

- **1주차**: 데이터 + PDF 파싱 + DB 구축
- **2주차**: RAG 파이프라인 + 검색 최적화
- **3주차**: UI + 고급 기능 (다중 상품, 비교)
- **4주차**: 평가 + 최적화 + 배포
