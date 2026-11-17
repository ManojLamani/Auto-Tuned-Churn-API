"""In-memory service metrics and a per-client rate limiter (enough for a single-instance MVP)."""
import time
from collections import Counter, defaultdict, deque


class Metrics:
    def __init__(self):
        self.started = time.time()
        self.requests = 0
        self.status_counts = Counter()
        self.latencies = deque(maxlen=1000)
        self.tier_counts = Counter()
        self.model_load_ms = None
        self.model_load_failures = 0
        self.version_counts = Counter()

    def record_request(self, status, latency_ms, model_version=None):
        self.requests += 1
        self.status_counts[status] += 1
        self.latencies.append(latency_ms)
        self.version_counts[model_version or "unavailable"] += 1

    def record_prediction(self, tier):
        self.tier_counts[tier] += 1

    def snapshot(self, model_version):
        lat = sorted(self.latencies)

        def pick(q):
            return round(lat[min(int(q * len(lat)), len(lat) - 1)], 1) if lat else None

        errors = sum(n for s, n in self.status_counts.items() if s >= 400)
        return {
            "modelVersion": model_version,
            "modelLoadMs": self.model_load_ms,
            "modelLoadFailures": self.model_load_failures,
            "requestsPerSecond": round(self.requests / max(time.time() - self.started, 1), 4),
            "modelVersionRequests": dict(self.version_counts),
            "uptimeSec": round(time.time() - self.started),
            "requests": self.requests,
            "errors": errors,
            "errorRate": round(errors / self.requests, 4) if self.requests else 0,
            "statusCounts": {str(s): n for s, n in sorted(self.status_counts.items())},
            "latencyMs": {"p50": pick(0.5), "p95": pick(0.95)},
            "predictions": sum(self.tier_counts.values()),
            "riskTierCounts": {t: self.tier_counts[t] for t in ("HIGH", "MEDIUM", "LOW")},
        }


class RateLimiter:
    """Allow at most `limit` requests per client in a sliding `window` of seconds."""

    def __init__(self, limit, window=60):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)

    def allow(self, client):
        now = time.monotonic()
        hits = self.hits[client]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True
