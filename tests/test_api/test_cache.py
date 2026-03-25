"""tests.test_api.test_cache - LRU 캐시 테스트."""

import time

from api.cache import LRUCache


class TestLRUCache:
    """LRUCache 동작 테스트."""

    def test_set_get(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_miss(self):
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # 'a'가 evict됨
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        cache = LRUCache(max_size=10, ttl_seconds=1)
        cache.set("key", "value")
        assert cache.get("key") == "value"
        time.sleep(1.1)
        assert cache.get("key") is None

    def test_make_key_deterministic(self):
        cache = LRUCache()
        key1 = cache.make_key("테스트 질문", 1)
        key2 = cache.make_key("테스트 질문", 1)
        key3 = cache.make_key("다른 질문", 1)
        assert key1 == key2
        assert key1 != key3

    def test_make_key_case_insensitive(self):
        cache = LRUCache()
        key1 = cache.make_key("Test Question", None)
        key2 = cache.make_key("test question", None)
        assert key1 == key2

    def test_invalidate(self):
        cache = LRUCache(max_size=10)
        cache.set("key", "value")
        cache.invalidate("key")
        assert cache.get("key") is None

    def test_clear(self):
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        cache = LRUCache(max_size=10)
        cache.set("key", "value")
        cache.get("key")      # hit
        cache.get("missing")  # miss
        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
