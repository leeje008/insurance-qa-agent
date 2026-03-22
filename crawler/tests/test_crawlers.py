"""crawler 모듈 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crawler.config import CrawlerSettings
from crawler.storage import PolicyStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """임시 데이터 디렉토리."""
    return tmp_path / "data"


@pytest.fixture()
def settings(tmp_data_dir: Path) -> CrawlerSettings:
    """테스트용 크롤러 설정."""
    return CrawlerSettings(
        data_dir=tmp_data_dir,
        request_delay_min=0.0,
        request_delay_max=0.0,
    )


@pytest.fixture()
def storage(settings: CrawlerSettings) -> PolicyStorage:
    """테스트용 PolicyStorage."""
    return PolicyStorage(settings)


# ---------------------------------------------------------------------------
# CrawlerSettings 테스트
# ---------------------------------------------------------------------------

class TestCrawlerSettings:
    """CrawlerSettings 테스트."""

    def test_default_settings(self) -> None:
        s = CrawlerSettings()
        assert s.request_delay_min == 5.0
        assert s.request_delay_max == 10.0
        assert s.max_retries == 3
        assert "InsuranceQABot" in s.user_agent
        assert s.headless is True

    def test_custom_settings(self, tmp_data_dir: Path) -> None:
        s = CrawlerSettings(
            data_dir=tmp_data_dir,
            request_delay_min=1.0,
            max_retries=5,
        )
        assert s.data_dir == tmp_data_dir
        assert s.request_delay_min == 1.0
        assert s.max_retries == 5


# ---------------------------------------------------------------------------
# PolicyStorage 테스트
# ---------------------------------------------------------------------------

class TestPolicyStorage:
    """PolicyStorage 테스트."""

    def test_save_pdf(self, storage: PolicyStorage, tmp_data_dir: Path) -> None:
        pdf_data = b"%PDF-1.4 fake pdf content"

        entry = storage.save_pdf(
            pdf_data,
            source="test_source",
            insurer="테스트보험",
            product_name="테스트상품",
            version="2024-01",
            effective_date="2024-01-01",
            source_url="https://example.com/test.pdf",
        )

        assert entry is not None
        assert entry["source"] == "test_source"
        assert entry["insurer"] == "테스트보험"
        assert entry["product_name"] == "테스트상품"
        assert entry["version"] == "2024-01"
        assert entry["sha256"] is not None
        assert entry["size_bytes"] == len(pdf_data)

        # 파일이 실제로 저장되었는지 확인
        saved_path = tmp_data_dir / entry["file_path"]
        assert saved_path.exists()
        assert saved_path.read_bytes() == pdf_data

        # manifest.json이 생성되었는지 확인
        manifest_path = tmp_data_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest) == 1

    def test_duplicate_detection(self, storage: PolicyStorage) -> None:
        pdf_data = b"%PDF-1.4 duplicate test"

        entry1 = storage.save_pdf(
            pdf_data, source="s", insurer="i", product_name="p", version="v1",
        )
        entry2 = storage.save_pdf(
            pdf_data, source="s", insurer="i", product_name="p2", version="v2",
        )

        assert entry1 is not None
        assert entry2 is None  # 동일 해시로 건너뜀

    def test_find_by_source(self, storage: PolicyStorage) -> None:
        storage.save_pdf(
            b"%PDF-1.4 content a",
            source="src_a", insurer="ins", product_name="prod", version="v1",
        )
        storage.save_pdf(
            b"%PDF-1.4 content b",
            source="src_b", insurer="ins", product_name="prod", version="v1",
        )

        found = storage.find_by_source("src_a", "ins", "prod", "v1")
        assert found is not None
        assert found["source"] == "src_a"

        not_found = storage.find_by_source("src_c", "ins", "prod", "v1")
        assert not_found is None

    def test_list_entries(self, storage: PolicyStorage) -> None:
        storage.save_pdf(
            b"%PDF-1.4 aaa", source="s1", insurer="i", product_name="p1", version="v",
        )
        storage.save_pdf(
            b"%PDF-1.4 bbb", source="s2", insurer="i", product_name="p2", version="v",
        )

        all_entries = storage.list_entries()
        assert len(all_entries) == 2

        s1_entries = storage.list_entries(source="s1")
        assert len(s1_entries) == 1
        assert s1_entries[0]["source"] == "s1"

    def test_manifest_persistence(
        self, settings: CrawlerSettings, tmp_data_dir: Path
    ) -> None:
        # 첫 번째 인스턴스로 저장
        storage1 = PolicyStorage(settings)
        storage1.save_pdf(
            b"%PDF-1.4 persist",
            source="s", insurer="i", product_name="p", version="v",
        )

        # 두 번째 인스턴스로 로드
        storage2 = PolicyStorage(settings)
        assert len(storage2.manifest) == 1
        assert storage2.manifest[0]["product_name"] == "p"


# ---------------------------------------------------------------------------
# Scheduler 테스트
# ---------------------------------------------------------------------------

class TestScheduleState:
    """ScheduleState 테스트."""

    def test_needs_update_no_history(self, settings: CrawlerSettings) -> None:
        from crawler.scheduler import ScheduleState

        schedule = ScheduleState(settings)
        assert schedule.needs_update("fss_standard") is True

    def test_record_and_check(self, settings: CrawlerSettings) -> None:
        from crawler.scheduler import ScheduleState

        schedule = ScheduleState(settings)
        schedule.record_run("fss_standard", count=5)

        # 방금 기록했으므로 갱신 불필요
        assert schedule.needs_update("fss_standard") is False

    def test_list_due_sources(self, settings: CrawlerSettings) -> None:
        from crawler.scheduler import ScheduleState

        schedule = ScheduleState(settings)
        due = schedule.list_due_sources()
        # 히스토리 없으므로 모든 소스가 갱신 필요
        assert len(due) > 0

    def test_summary(self, settings: CrawlerSettings) -> None:
        from crawler.scheduler import ScheduleState

        schedule = ScheduleState(settings)
        summary = schedule.summary()
        assert "fss_standard" in summary
        assert "knia_auto" in summary
        assert summary["fss_standard"]["interval_days"] == 90
