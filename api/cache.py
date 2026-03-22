"""api.cache - 인메모리 LRU 캐시 (답변 캐싱)."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE = 256
DEFAULT_TTL_SECONDS = 3600  # 1시간


@dataclass
class CacheEntry:
    """캐시 항목."""

    value: Any
    created_at: float


class LRUCache:
    """TTL 기반 LRU 캐시.

    Q&A 답변을 캐싱하여 동일 질문에 대한 반복 LLM 호출을 방지한다.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(question: str, product_id: int | None = None) -> str:
        """질문 + 상품 ID로 캐시 키 생성."""
        raw = f"{question.strip().lower()}:{product_id or 'all'}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """캐시 조회. TTL 만료된 항목은 제거."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """캐시 저장. 용량 초과 시 가장 오래된 항목 제거."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(value=value, created_at=time.time())
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(value=value, created_at=time.time())

    def invalidate(self, key: str) -> bool:
        """특정 키 캐시 무효화."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """전체 캐시 초기화."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """캐시 통계."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }


# 글로벌 캐시 인스턴스
qa_cache = LRUCache()
