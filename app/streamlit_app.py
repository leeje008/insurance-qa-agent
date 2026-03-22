"""app.streamlit_app - Streamlit 메인 앱 엔트리포인트."""

from __future__ import annotations

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="보험약관 Q&A",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("보험약관 Q&A")
    st.markdown("RAG 기반 보험약관 질의응답 시스템")
    st.divider()

    st.subheader("설정")
    st.session_state.api_base_url = st.text_input(
        "API 서버 URL",
        value=st.session_state.api_base_url,
    )

    st.divider()
    st.markdown("### 페이지")
    st.page_link("app/streamlit_app.py", label="홈", icon="🏠")
    st.page_link("app/pages/01_qa.py", label="Q&A 질의응답", icon="💬")

    st.divider()
    st.caption("Insurance QA Agent v0.1.0")

# ---------------------------------------------------------------------------
# 메인 페이지
# ---------------------------------------------------------------------------

st.title("보험약관 Q&A 시스템")
st.markdown("""
보험약관의 내용을 AI가 분석하여 정확한 답변을 제공합니다.

### 주요 기능
- **약관 질의응답**: 보험약관에 대한 질문을 자연어로 입력하면 관련 조항을 찾아 답변
- **조항 인용**: 답변의 근거가 되는 약관 조항 번호를 함께 제공
- **신뢰도 표시**: 답변의 정확도를 수치로 확인

### 사용 방법
1. 왼쪽 사이드바에서 **Q&A 질의응답** 페이지로 이동
2. 질문을 입력하고 **질문하기** 버튼 클릭
3. AI가 약관을 분석하여 답변을 생성

---
""")

# 시스템 상태 표시
try:
    resp = httpx.get(f"{st.session_state.api_base_url}/health", timeout=5)
    if resp.status_code == 200:
        st.success("API 서버 연결됨")

        # 통계 표시
        try:
            stats_resp = httpx.get(
                f"{st.session_state.api_base_url}/admin/stats", timeout=5,
            )
            if stats_resp.status_code == 200:
                stats = stats_resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("등록 상품", stats.get("total_products", 0))
                col2.metric("약관 조항", stats.get("total_articles", 0))
                col3.metric("Q&A 이력", stats.get("total_qa_logs", 0))
        except Exception:
            pass
    else:
        st.warning("API 서버 응답 이상")
except Exception:
    st.error(
        f"API 서버에 연결할 수 없습니다: {st.session_state.api_base_url}\n\n"
        "서버가 실행 중인지 확인하세요."
    )
