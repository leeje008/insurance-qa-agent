"""app.pages.01_qa - Q&A 페이지 (질문 입력 및 답변 표시)."""

from __future__ import annotations

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Q&A 질의응답", page_icon="💬", layout="wide")

# 세션 상태
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

API_BASE = st.session_state.api_base_url

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Q&A 질의응답")

    st.session_state.api_base_url = st.text_input(
        "API 서버 URL",
        value=st.session_state.api_base_url,
        key="qa_api_url",
    )

    # 상품 선택
    product_id = None
    try:
        resp = httpx.get(f"{API_BASE}/admin/products", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            products = data.get("products", [])
            if products:
                options = [{"id": None, "label": "전체 상품"}] + [
                    {"id": p["id"], "label": f"{p['insurer']} - {p['name']}"}
                    for p in products
                ]
                selected = st.selectbox(
                    "상품 선택",
                    options=options,
                    format_func=lambda x: x["label"],
                )
                product_id = selected["id"] if selected else None
    except Exception:
        st.caption("상품 목록 로드 실패")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------

st.title("💬 보험약관 Q&A")

# 대화 히스토리 표시
for msg in st.session_state.chat_history:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        if role == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # 신뢰도 표시
            confidence = meta.get("confidence", 0)
            if confidence > 0:
                color = "green" if confidence >= 0.7 else (
                    "orange" if confidence >= 0.4 else "red"
                )
                st.markdown(
                    f"**신뢰도**: :{color}[{confidence:.0%}]"
                )

            # 근거 조항 표시
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📌 근거 조항 ({len(sources)}건)"):
                    for src in sources:
                        title = src.get("title", "")
                        label = src["number"]
                        if title:
                            label += f" ({title})"
                        st.markdown(f"- {label}")

            # 피드백 버튼
            log_id = meta.get("log_id")
            if log_id:
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    score = i + 1
                    emoji = ["😞", "😐", "🙂", "😊", "🤩"][i]
                    if col.button(
                        emoji, key=f"fb_{log_id}_{score}",
                        help=f"{score}점",
                    ):
                        try:
                            httpx.post(
                                f"{API_BASE}/qa/{log_id}/feedback",
                                json={"score": score},
                                timeout=5,
                            )
                            st.toast(f"피드백 감사합니다! ({score}점)")
                        except Exception:
                            st.toast("피드백 전송 실패")

# ---------------------------------------------------------------------------
# 질문 입력
# ---------------------------------------------------------------------------

question = st.chat_input("보험약관에 대해 질문하세요...")

if question:
    # 사용자 메시지 추가
    st.session_state.chat_history.append({
        "role": "user", "content": question,
    })
    with st.chat_message("user"):
        st.markdown(question)

    # API 호출
    with st.chat_message("assistant"):
        with st.spinner("약관을 분석하고 있습니다..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/qa/ask",
                    json={
                        "question": question,
                        "product_id": product_id,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "답변을 생성하지 못했습니다.")
                    confidence = data.get("confidence", 0)
                    sources = data.get("sources", [])
                    log_id = data.get("log_id")

                    st.markdown(answer)

                    # 신뢰도
                    if confidence > 0:
                        color = "green" if confidence >= 0.7 else (
                            "orange" if confidence >= 0.4 else "red"
                        )
                        st.markdown(
                            f"**신뢰도**: :{color}[{confidence:.0%}]"
                        )

                    # 근거 조항
                    if sources:
                        with st.expander(
                            f"📌 근거 조항 ({len(sources)}건)"
                        ):
                            for src in sources:
                                title = src.get("title", "")
                                label = src["number"]
                                if title:
                                    label += f" ({title})"
                                st.markdown(f"- {label}")

                    # 히스토리에 추가
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "confidence": confidence,
                            "sources": sources,
                            "log_id": log_id,
                        },
                    })
                else:
                    error_msg = f"API 오류 ({resp.status_code})"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": error_msg,
                    })

            except httpx.TimeoutException:
                msg = "요청 시간이 초과되었습니다. 다시 시도해 주세요."
                st.error(msg)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": msg,
                })
            except Exception as exc:
                msg = f"오류 발생: {exc}"
                st.error(msg)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": msg,
                })
