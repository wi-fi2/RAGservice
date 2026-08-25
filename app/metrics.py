"""Metrics collection for observability."""

import time
from typing import Dict, List
from collections import defaultdict, deque
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()

request_counter = Counter(
    'rag_requests_total',
    'Total number of requests',
    ['operation', 'status'],
    registry=registry
)

request_duration = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['operation'],
    registry=registry
)

active_operations = Gauge(
    'rag_active_operations',
    'Number of active operations',
    ['operation'],
    registry=registry
)

token_usage = Counter(
    'rag_tokens_total',
    'Total tokens used',
    registry=registry
)

vector_count = Gauge(
    'rag_indexed_vectors',
    'Number of indexed vectors',
    registry=registry
)


class MetricsCollector:
    """Collects and calculates metrics for the RAG service."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_tokens = 0
        
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record an operation completion."""
        status = 'success' if success else 'error'
        request_counter.labels(operation=operation, status=status).inc()
        request_duration.labels(operation=operation).observe(duration)
        
        if success:
            self.operation_counts[operation] += 1
            self.latencies[operation].append(duration * 1000)  # Convert to ms
        else:
            self.error_counts[operation] += 1
    
    def record_latency(self, component: str, latency_ms: float):
        """Record component latency."""
        self.latencies[component].append(latency_ms)
    
    def record_tokens(self, count: int):
        """Record token usage."""
        self.total_tokens += count
        token_usage.inc(count)
    
    def update_vector_count(self, count: int):
        """Update indexed vector count."""
        vector_count.set(count)
    
    def get_percentiles(self, operation: str) -> Dict[str, float]:
        """Calculate percentiles for an operation."""
        if operation not in self.latencies or not self.latencies[operation]:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        latencies = list(self.latencies[operation])
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        stats = {
            "operations": {},
            "errors": dict(self.error_counts),
            "total_tokens": self.total_tokens
        }
        
        for op in set(list(self.operation_counts.keys()) + list(self.latencies.keys())):
            stats["operations"][op] = {
                "count": self.operation_counts.get(op, 0),
                "latency_ms": self.get_percentiles(op)
            }
        
        return stats


# Global metrics instance
metrics = MetricsCollector()


def metrics_handler():
    """Generate Prometheus metrics."""
    return generate_latest(registry).decode('utf-8')


def log_metrics_summary():
    """Log a summary of current metrics."""
    stats = metrics.get_stats()
    logger.info("=== Metrics Summary ===")
    for op, data in stats["operations"].items():
        logger.info(f"{op}: count={data['count']}, p50={data['latency_ms']['p50']:.1f}ms, p95={data['latency_ms']['p95']:.1f}ms")
    logger.info(f"Total tokens: {stats['total_tokens']}")
    logger.info(f"Errors: {stats['errors']}")










