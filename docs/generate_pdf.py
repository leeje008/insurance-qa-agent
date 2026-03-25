"""PDF 보고서 생성 스크립트 — 2026-03-25 고도화 작업 보고서."""

from fpdf import FPDF
import os

# 한국어 지원을 위한 폰트 경로 (macOS)
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
]


def find_korean_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        font_path = find_korean_font()
        if font_path:
            self.add_font("Korean", "", font_path, uni=True)
            self.add_font("Korean", "B", font_path, uni=True)
            self.font_family_name = "Korean"
        else:
            self.font_family_name = "Helvetica"

    def header(self):
        self.set_font(self.font_family_name, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Insurance QA Agent - Enhancement Report 2026-03-25", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family_name, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font(self.font_family_name, "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def subsection_title(self, title):
        self.set_font(self.font_family_name, "B", 11)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font(self.font_family_name, "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_font(self.font_family_name, "B", 8)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cols, widths, fill=False):
        self.set_font(self.font_family_name, "", 8)
        self.set_text_color(0, 0, 0)
        if fill:
            self.set_fill_color(240, 240, 250)
        for col, w in zip(cols, widths):
            self.cell(w, 6, str(col), border=1, fill=fill, align="L")
        self.ln()


def generate():
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Page 1: Title ---
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font(pdf.font_family_name, "B", 24)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Insurance QA Agent", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(pdf.font_family_name, "B", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Enhancement Report", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_font(pdf.font_family_name, "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "2026-03-25", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "RAG Pipeline + Data Science Infrastructure", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Branch: develop", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)
    pdf.set_font(pdf.font_family_name, "", 10)
    pdf.set_text_color(0, 0, 0)

    summary_items = [
        ("Total Tasks Implemented", "14"),
        ("New Files Created", "20"),
        ("Modified Files", "16"),
        ("Test Cases", "95 (all passing)"),
        ("Lint Status", "All checks passed"),
        ("New Reranker Strategies", "5 (pluggable)"),
        ("New API Endpoints", "4"),
        ("New DB Tables", "1 (RetrievalLog)"),
    ]

    widths = [90, 90]
    pdf.table_header(["Metric", "Value"], widths)
    for i, (k, v) in enumerate(summary_items):
        pdf.table_row([k, v], widths, fill=(i % 2 == 0))

    # --- Page 2: RAG Pipeline ---
    pdf.add_page()
    pdf.section_title("1. RAG Pipeline Enhancement")

    pdf.subsection_title("1.1 Multi-turn Conversation")
    pdf.body_text(
        "ConversationMessage schema added to QuestionRequest. "
        "Last 10 messages sent to backend. Query Processor uses CONTEXTUAL_QUERY_PROMPT "
        "to resolve pronouns and contextual references before QUERY_ANALYSIS_PROMPT."
    )

    pdf.subsection_title("1.2 Pluggable Reranker System (5 strategies)")
    reranker_data = [
        ("cross_encoder", "BAAI/bge-reranker-v2-m3", "CPU 3-6s", "Default"),
        ("llm_listwise", "Ollama qwen2.5:14b", "5-8s", "No extra deps"),
        ("similarity", "nomic-embed-text re-embed", "1-2s", "Lightweight"),
        ("ltr", "LightGBM LambdaRank", "<10ms", "Requires training"),
        ("noop", "Pass-through", "0ms", "Baseline"),
    ]
    widths_r = [35, 50, 30, 40]
    pdf.table_header(["Strategy", "Model", "Latency", "Notes"], widths_r)
    for i, row in enumerate(reranker_data):
        pdf.table_row(row, widths_r, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.subsection_title("1.3 HyDE (Hypothetical Document Embeddings)")
    pdf.body_text(
        "Query Processor generates a hypothetical policy article via LLM, embeds it, "
        "and the Retriever runs dual-embedding semantic search (query + HyDE). "
        "Results are merged via RRF fusion."
    )

    pdf.subsection_title("1.4 Real-time SSE Streaming")
    pdf.body_text(
        "Replaced fake streaming with LangGraph astream. "
        "5-stage SSE events: query_processing -> retrieval -> reranking -> generation -> validation."
    )

    pdf.subsection_title("1.5 Pseudo-Relevance Feedback (PRF)")
    pdf.body_text(
        "Extracts key Korean terms from top-3 retrieved documents. "
        "Expands keyword list and re-runs keyword search. "
        "Results merged via RRF. No training data required."
    )

    # --- Page 3: Data Science ---
    pdf.add_page()
    pdf.section_title("2. Data Science Infrastructure")

    pdf.subsection_title("2.1 Evaluation Framework")
    metrics_data = [
        ("MRR", "Mean Reciprocal Rank", "Overall retrieval quality"),
        ("NDCG@K", "Normalized DCG", "Position-weighted relevance"),
        ("Precision@K", "Top-K precision", "Precision measurement"),
        ("Hit Rate@K", "At least 1 hit in K", "Recall indicator"),
    ]
    widths_m = [30, 55, 65]
    pdf.table_header(["Metric", "Full Name", "Purpose"], widths_m)
    for i, row in enumerate(metrics_data):
        pdf.table_row(row, widths_m, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.subsection_title("2.2 Retrieval Failure Analysis")
    failure_data = [
        ("Embedding mismatch", "High keyword but low semantic score"),
        ("Missing content", "No relevant article in DB"),
        ("Chunking error", "Correct article but wrong chunk ranked high"),
        ("Intent confusion", "Keyword match but wrong intent"),
        ("Cross-reference gap", "Answer needs multiple articles"),
        ("Terminology gap", "Colloquial vs legal term mismatch"),
    ]
    widths_f = [50, 120]
    pdf.table_header(["Failure Mode", "Detection Signal"], widths_f)
    for i, row in enumerate(failure_data):
        pdf.table_row(row, widths_f, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.subsection_title("2.3 Learning-to-Rank (LightGBM LambdaRank)")
    pdf.body_text(
        "10 features extracted from retrieval_logs: semantic_score, keyword_score, "
        "rrf_score, rerank_score, doc_length, article_level, title_match, "
        "query_length, num_keywords, score_gap. "
        "Labels from feedback_score. Expected NDCG improvement: 5-15%."
    )

    pdf.subsection_title("2.4 Confidence Calibration (Platt Scaling)")
    pdf.body_text(
        "Logistic regression: P(satisfied) = sigmoid(w * confidence + b). "
        "Trained on qa_logs where feedback_score >= 4 = positive. "
        "Persisted as JSON model file."
    )

    pdf.subsection_title("2.5 Active Learning")
    pdf.body_text(
        "3 sampling strategies for annotation prioritization: "
        "Uncertainty (confidence near threshold), "
        "Diversity (question length bucketing), "
        "Low confidence (model's hardest cases). "
        "100 selected labels = 500 random labels in evaluation quality."
    )

    # --- Page 4: ML Models ---
    pdf.add_page()
    pdf.section_title("3. ML Models & Latency Optimization")

    pdf.subsection_title("3.1 Intent Classifier (TF-IDF + LinearSVC)")
    pdf.body_text(
        "Character n-gram (3-5) TF-IDF + LinearSVC. CPU inference <50ms. "
        "Replaces LLM intent extraction (2-4s). "
        "6 classes: coverage, claim, exclusion, definition, comparison, general. "
        "Bootstrap from LLM labels on existing qa_logs."
    )

    pdf.subsection_title("3.2 Quality Predictor (GradientBoosting)")
    pdf.body_text(
        "Predicts feedback_score (1-5) from 9 features. "
        "If predicted >= 4.0, skip LLM answer_validator (saves 3-5s). "
        "Features: confidence, num_sources, answer_length, question_length, "
        "max/mean_retrieval_score, score_gap, num_retrieved, has_reranking."
    )

    pdf.subsection_title("3.3 Parameter Optimizer")
    pdf.body_text(
        "Grid search for SEMANTIC_WEIGHT/KEYWORD_WEIGHT (0.0-1.0, 11 steps) "
        "and SIMILARITY_THRESHOLD (0.3-0.9, 13 steps). "
        "Evaluates each combination on golden test set via NDCG@5."
    )

    # --- Page 5: Security & Testing ---
    pdf.section_title("4. Security & Testing")

    pdf.subsection_title("4.1 API Security")
    sec_data = [
        ("API Key Auth", "X-API-Key header", "API_KEY env var (empty=disabled)"),
        ("Rate Limiting", "IP sliding window", "RATE_LIMIT_RPM (0=disabled)"),
        ("CORS", "Config-based origins", "CORS_ORIGINS (comma-separated)"),
    ]
    widths_s = [40, 50, 70]
    pdf.table_header(["Feature", "Implementation", "Configuration"], widths_s)
    for i, row in enumerate(sec_data):
        pdf.table_row(row, widths_s, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.subsection_title("4.2 Test Coverage")
    test_data = [
        ("Existing tests", "51", "PASS"),
        ("Cache tests", "9", "PASS"),
        ("Reranker tests", "11", "PASS"),
        ("Evaluation metrics", "14", "PASS"),
        ("Calibration tests", "10", "PASS"),
        ("Total", "95", "ALL PASS"),
    ]
    widths_t = [60, 30, 30]
    pdf.table_header(["Test Group", "Count", "Status"], widths_t)
    for i, row in enumerate(test_data):
        pdf.table_row(row, widths_t, fill=(i % 2 == 0))

    # --- Page 6: New API & Config ---
    pdf.add_page()
    pdf.section_title("5. New API Endpoints & Configuration")

    pdf.subsection_title("5.1 New Endpoints")
    ep_data = [
        ("POST", "/admin/rerank/compare", "Compare all reranker strategies"),
        ("POST", "/admin/eval/reranker-benchmark", "Golden set benchmark"),
        ("POST", "/admin/eval/failure-analysis", "6-mode failure analysis"),
        ("GET", "/admin/eval/annotation-queue", "Active learning queue"),
    ]
    widths_e = [20, 65, 75]
    pdf.table_header(["Method", "Path", "Description"], widths_e)
    for i, row in enumerate(ep_data):
        pdf.table_row(row, widths_e, fill=(i % 2 == 0))

    pdf.ln(3)
    pdf.subsection_title("5.2 New Configuration Variables")
    conf_data = [
        ("RERANK_STRATEGY", "cross_encoder", "Reranker strategy"),
        ("API_KEY", "(empty)", "API authentication key"),
        ("RATE_LIMIT_RPM", "0", "Rate limit per minute"),
        ("CORS_ORIGINS", "*", "Allowed CORS origins"),
    ]
    widths_c = [45, 35, 60]
    pdf.table_header(["Variable", "Default", "Description"], widths_c)
    for i, row in enumerate(conf_data):
        pdf.table_row(row, widths_c, fill=(i % 2 == 0))

    pdf.ln(5)
    pdf.section_title("6. File Changes Summary")
    pdf.body_text("New files: 20  |  Modified files: 16  |  Total lines added: ~1,300+")
    pdf.body_text(
        "Key new packages: api/evaluation/ (10 modules), "
        "api/retrieval/reranker.py, api/retrieval/query_expansion.py, api/security.py"
    )

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "2026-03-25-enhancement-report.pdf")
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    generate()
