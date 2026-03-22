"""crawler.storage - PDF 저장 및 버전 관리."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crawler.config import CrawlerSettings, get_crawler_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest 항목 타입
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"


def _sha256(data: bytes) -> str:
    """바이트 데이터의 SHA-256 해시 반환."""
    return hashlib.sha256(data).hexdigest()


class PolicyStorage:
    """약관 PDF 저장소 — manifest.json 기반 중복 방지 및 버전 관리."""

    def __init__(self, settings: CrawlerSettings | None = None) -> None:
        self._settings = settings or get_crawler_settings()
        self._data_dir = self._settings.data_dir
        self._manifest_path = self._data_dir / MANIFEST_FILENAME
        self._manifest: list[dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict[str, Any]]:
        """manifest.json 로드. 파일이 없으면 빈 리스트 반환."""
        if self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_manifest(self) -> None:
        """manifest.json 저장."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------
    # 중복 검사
    # ------------------------------------------------------------------

    def find_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """SHA-256 해시로 기존 항목 조회."""
        for entry in self._manifest:
            if entry.get("sha256") == sha256:
                return entry
        return None

    def find_by_source(
        self, source: str, insurer: str, product_name: str, version: str
    ) -> dict[str, Any] | None:
        """소스/보험사/상품명/버전으로 기존 항목 조회."""
        for entry in self._manifest:
            if (
                entry.get("source") == source
                and entry.get("insurer") == insurer
                and entry.get("product_name") == product_name
                and entry.get("version") == version
            ):
                return entry
        return None

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        pdf_data: bytes,
        *,
        source: str,
        insurer: str,
        product_name: str,
        version: str,
        effective_date: str | None = None,
        source_url: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """PDF 바이트를 저장하고 manifest에 기록.

        이미 동일한 파일(해시 기준)이 존재하면 None을 반환하고 건너뛴다.

        Returns:
            저장된 manifest 항목 dict, 또는 중복 시 None.
        """
        sha256 = _sha256(pdf_data)

        if self.find_by_hash(sha256):
            logger.info(
                "중복 PDF 건너뜀: %s / %s / %s (hash=%s)",
                source, insurer, product_name, sha256[:12],
            )
            return None

        # 저장 경로 결정
        safe_insurer = insurer.replace("/", "_").replace(" ", "_")
        safe_product = product_name.replace("/", "_").replace(" ", "_")
        sub_dir = self._data_dir / source / safe_insurer
        sub_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{safe_product}_{version}.pdf"
        file_path = sub_dir / filename

        # 파일명 충돌 시 숫자 접미사 추가
        counter = 1
        while file_path.exists():
            filename = f"{safe_product}_{version}_{counter}.pdf"
            file_path = sub_dir / filename
            counter += 1

        file_path.write_bytes(pdf_data)
        logger.info("PDF 저장: %s (%d bytes)", file_path, len(pdf_data))

        # manifest 항목 작성
        entry: dict[str, Any] = {
            "source": source,
            "insurer": insurer,
            "product_name": product_name,
            "version": version,
            "effective_date": effective_date,
            "source_url": source_url,
            "file_path": str(file_path.relative_to(self._data_dir)),
            "sha256": sha256,
            "size_bytes": len(pdf_data),
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        if extra_metadata:
            entry["metadata"] = extra_metadata

        self._manifest.append(entry)
        self._save_manifest()
        return entry

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def list_entries(self, source: str | None = None) -> list[dict[str, Any]]:
        """manifest 항목 목록 반환. source로 필터링 가능."""
        if source is None:
            return list(self._manifest)
        return [e for e in self._manifest if e.get("source") == source]

    @property
    def manifest(self) -> list[dict[str, Any]]:
        """전체 manifest 반환."""
        return list(self._manifest)

    def resolve_path(self, entry: dict[str, Any]) -> Path:
        """manifest 항목의 절대 파일 경로 반환."""
        return self._data_dir / entry["file_path"]
