"""Caching layer for prompt-response pairs using SQLite."""

import sqlite3
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached prompt-response pair."""
    cache_key: str
    prompt_hash: str
    model: str
    system_prompt: str
    user_prompt: str
    response: str
    score: Optional[float]
    created_at: str
    expires_at: Optional[str] = None


class Cache:
    """SQLite-based cache for prompt-response pairs."""

    def __init__(
        self,
        db_path: str = ".prompt_protect_cache.db",
        ttl_hours: int = 168,  # 1 week default
    ):
        self._db_path = db_path
        self._ttl_hours = ttl_hours
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                system_prompt TEXT,
                user_prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                score REAL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                hit_count INTEGER DEFAULT 1,
                last_hit TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_key ON prompt_cache(cache_key)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_hash ON prompt_cache(prompt_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model ON prompt_cache(model)
        """)

        conn.commit()
        conn.close()

    def _hash_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        """Create a hash for the prompt combination."""
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _make_cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> str:
        """Create a cache key for the prompt."""
        prompt_hash = self._hash_prompt(system_prompt, user_prompt, model)
        return f"{model}:{prompt_hash}"

    def get(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> Optional[CacheEntry]:
        """Get a cached response if available."""
        cache_key = self._make_cache_key(system_prompt, user_prompt, model)

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM prompt_cache
                WHERE cache_key = ?
                AND (expires_at IS NULL OR expires_at > ?)
            """, (cache_key, datetime.utcnow().isoformat()))

            row = cursor.fetchone()

            if row:
                # Update hit count
                cursor.execute("""
                    UPDATE prompt_cache
                    SET hit_count = hit_count + 1,
                        last_hit = ?
                    WHERE cache_key = ?
                """, (datetime.utcnow().isoformat(), cache_key))

                conn.commit()

                return CacheEntry(
                    cache_key=row["cache_key"],
                    prompt_hash=row["prompt_hash"],
                    model=row["model"],
                    system_prompt=row["system_prompt"],
                    user_prompt=row["user_prompt"],
                    response=row["response"],
                    score=row["score"],
                    created_at=row["created_at"],
                    expires_at=row["expires_at"],
                )

            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
        finally:
            conn.close()

    def put(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        response: str,
        score: Optional[float] = None,
        ttl_hours: Optional[int] = None,
    ) -> None:
        """Store a prompt-response pair in cache."""
        cache_key = self._make_cache_key(system_prompt, user_prompt, model)
        prompt_hash = self._hash_prompt(system_prompt, user_prompt, model)

        ttl = ttl_hours if ttl_hours is not None else self._ttl_hours
        expires_at = (datetime.utcnow() + timedelta(hours=ttl)).isoformat() if ttl > 0 else None

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO prompt_cache
                (cache_key, prompt_hash, model, system_prompt, user_prompt,
                 response, score, created_at, expires_at, hit_count, last_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                cache_key,
                prompt_hash,
                model,
                system_prompt[:5000],  # Limit stored length
                user_prompt[:5000],
                response[:50000],
                score,
                datetime.utcnow().isoformat(),
                expires_at,
                datetime.utcnow().isoformat(),
            ))

            conn.commit()
            logger.debug(f"Cached response for {cache_key}")

        except Exception as e:
            logger.warning(f"Cache put error: {e}")
        finally:
            conn.close()

    def invalidate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> bool:
        """Invalidate a specific cache entry."""
        cache_key = self._make_cache_key(system_prompt, user_prompt, model)

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM prompt_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return False
        finally:
            conn.close()

    def clear_expired(self) -> int:
        """Remove expired cache entries."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                DELETE FROM prompt_cache
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.utcnow().isoformat(),))

            count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {count} expired cache entries")
            return count
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0
        finally:
            conn.close()

    def clear_all(self) -> None:
        """Clear all cache entries."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM prompt_cache")
            conn.commit()
            logger.info("Cleared all cache entries")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    COUNT(CASE WHEN expires_at > ? THEN 1 END) as active_entries
                FROM prompt_cache
            """, (datetime.utcnow().isoformat(),))

            row = cursor.fetchone()

            return {
                "total_entries": row[0] or 0,
                "total_hits": row[1] or 0,
                "active_entries": row[2] or 0,
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {}
        finally:
            conn.close()

    def get_top_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most cached models."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT model, COUNT(*) as count, SUM(hit_count) as hits
                FROM prompt_cache
                GROUP BY model
                ORDER BY hits DESC
                LIMIT ?
            """, (limit,))

            return [
                {"model": row[0], "entries": row[1], "hits": row[2]}
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning(f"Cache top models error: {e}")
            return []
        finally:
            conn.close()


class CachedClient:
    """Wrapper that adds caching to any LLM client."""

    def __init__(self, client, cache: Cache):
        self._client = client
        self._cache = cache

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat with caching."""
        # Extract system prompt and user message
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]

        model = getattr(self._client, "_config", None)
        if model and hasattr(model, "model"):
            model = model.model
        else:
            model = "default"

        # Try cache first
        cached = self._cache.get(system_prompt, user_prompt, model)
        if cached:
            logger.debug(f"Cache hit for {model}")
            return cached.response

        # Call the actual client
        response = await self._client.chat(messages, temperature, max_tokens)

        # Cache the response
        self._cache.put(system_prompt, user_prompt, model, response)

        return response
