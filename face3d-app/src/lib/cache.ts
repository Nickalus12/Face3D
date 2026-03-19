/**
 * IPC result caching layer with TTL support.
 *
 * Avoids redundant Rust calls for data that changes infrequently.
 * Every cache entry has a time-to-live; stale entries are evicted on read.
 */

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

class IPCCache {
  private store = new Map<string, CacheEntry<unknown>>();

  /** Return cached value or null if missing / expired. */
  get<T>(key: string): T | null {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return null;
    }
    return entry.value as T;
  }

  /** Store a value with the given TTL (milliseconds). */
  set<T>(key: string, value: T, ttlMs: number): void {
    this.store.set(key, { value, expiresAt: Date.now() + ttlMs });
  }

  /**
   * Remove all entries whose key starts with `prefix`.
   * Pass an empty string to clear everything.
   */
  invalidate(prefix: string): void {
    if (prefix === "") {
      this.store.clear();
      return;
    }
    for (const key of Array.from(this.store.keys())) {
      if (key.startsWith(prefix)) {
        this.store.delete(key);
      }
    }
  }

  /** Number of live entries (for debugging). */
  get size(): number {
    return this.store.size;
  }
}

/** Singleton cache shared across the whole app. */
export const ipcCache = new IPCCache();

// ── TTL presets (ms) ────────────────────────────────────────────

/** GPU info refreshes often — short TTL. */
export const TTL_GPU = 2_000;

/** Session list changes on pipeline complete. */
export const TTL_SESSION_LIST = 10_000;

/** Sensor data is static once recorded. */
export const TTL_SENSOR = 60_000;

/** Model path is stable within a session. */
export const TTL_MODEL_PATH = 120_000;

/** Session files / image counts — moderate. */
export const TTL_SESSION_FILES = 15_000;

/** System info rarely changes. */
export const TTL_SYSTEM_INFO = 300_000;

export default IPCCache;
