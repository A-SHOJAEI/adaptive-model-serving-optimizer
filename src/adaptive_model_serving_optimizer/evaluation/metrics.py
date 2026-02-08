"""Evaluation metrics and monitoring for adaptive model serving optimizer.

This module provides comprehensive monitoring capabilities including performance
metrics, accuracy tracking, and real-time alerting for the MLOps system.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    top5_accuracy: float = 0.0
    confidence_score: float = 0.0
    num_samples: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """Alert for monitoring thresholds."""

    metric_name: str
    threshold: float
    current_value: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates metrics from model serving."""

    def __init__(self, config: Config):
        """Initialize metrics collector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.window_size = config.monitoring.throughput_window

        # Performance metrics storage
        self.performance_history: deque = deque(maxlen=10000)
        self.accuracy_history: deque = deque(maxlen=10000)

        # Real-time metrics
        self.current_metrics = PerformanceMetrics()
        self.current_accuracy = AccuracyMetrics()

        # Aggregated metrics by strategy
        self.strategy_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.strategy_accuracy: Dict[str, List[AccuracyMetrics]] = defaultdict(list)

        # Alert system
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_thresholds = config.monitoring.alert_thresholds

    def record_performance(
        self,
        strategy_name: str,
        latency_ms: float,
        throughput_fps: float,
        memory_usage_mb: float = 0.0,
        cpu_usage_percent: float = 0.0,
        gpu_usage_percent: float = 0.0,
        gpu_memory_usage_mb: float = 0.0,
        error_rate: float = 0.0
    ) -> None:
        """Record performance metrics.

        Args:
            strategy_name: Name of serving strategy
            latency_ms: Inference latency in milliseconds
            throughput_fps: Throughput in frames per second
            memory_usage_mb: Memory usage in MB
            cpu_usage_percent: CPU usage percentage
            gpu_usage_percent: GPU usage percentage
            gpu_memory_usage_mb: GPU memory usage in MB
            error_rate: Error rate (0-1)
        """
        metrics = PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            gpu_usage_percent=gpu_usage_percent,
            gpu_memory_usage_mb=gpu_memory_usage_mb,
            error_rate=error_rate
        )

        # Store in history
        self.performance_history.append(metrics)
        self.strategy_metrics[strategy_name].append(metrics)

        # Update current metrics
        self.current_metrics = metrics

        # Check alerts
        self._check_performance_alerts(metrics)

        logger.debug(
            f"Recorded performance: {strategy_name} - "
            f"latency={latency_ms:.2f}ms, throughput={throughput_fps:.2f}fps"
        )

    def record_accuracy(
        self,
        strategy_name: str,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        confidence_scores: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> AccuracyMetrics:
        """Record accuracy metrics.

        Args:
            strategy_name: Name of serving strategy
            predictions: Model predictions
            targets: Ground truth labels
            confidence_scores: Optional confidence scores

        Returns:
            Computed accuracy metrics
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if confidence_scores is not None and isinstance(confidence_scores, torch.Tensor):
            confidence_scores = confidence_scores.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Calculate top-5 accuracy for multi-class problems
        top5_acc = 0.0
        if len(np.unique(targets)) > 5:  # Multi-class with >5 classes
            top5_acc = self._calculate_top5_accuracy(predictions, targets)

        # Calculate average confidence
        avg_confidence = 0.0
        if confidence_scores is not None:
            avg_confidence = float(np.mean(confidence_scores))

        metrics = AccuracyMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            top5_accuracy=top5_acc,
            confidence_score=avg_confidence,
            num_samples=len(targets)
        )

        # Store in history
        self.accuracy_history.append(metrics)
        self.strategy_accuracy[strategy_name].append(metrics)

        # Update current accuracy
        self.current_accuracy = metrics

        # Check accuracy alerts
        self._check_accuracy_alerts(metrics)

        logger.debug(
            f"Recorded accuracy: {strategy_name} - "
            f"acc={accuracy:.4f}, f1={f1:.4f}, samples={len(targets)}"
        )

        return metrics

    def _calculate_top5_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate top-5 accuracy.

        Args:
            predictions: Predicted class probabilities or logits
            targets: Ground truth labels

        Returns:
            Top-5 accuracy
        """
        try:
            if len(predictions.shape) == 1:
                # Single class predictions, return regular accuracy
                return accuracy_score(targets, predictions)

            # Get top-5 predictions
            top5_preds = np.argsort(predictions, axis=1)[:, -5:]
            correct = 0

            for i, target in enumerate(targets):
                if target in top5_preds[i]:
                    correct += 1

            return correct / len(targets)

        except Exception as e:
            logger.warning(f"Failed to calculate top-5 accuracy: {e}", exc_info=True)
            logger.info("Returning NaN to indicate calculation failure, not zero accuracy")
            return float('nan')

    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance threshold violations and generate alerts.

        Evaluates current performance metrics against configured thresholds
        and creates alert objects for violations. Alerts are categorized as
        'warning' or 'critical' based on severity (critical if >1.5x threshold).

        Monitored performance metrics:
        - p99_latency_ms: 99th percentile latency in milliseconds
        - error_rate: Request error rate (0.0-1.0)
        - memory_usage_percent: System memory usage as percentage
        - gpu_memory_usage_percent: GPU memory usage as percentage

        Args:
            metrics: PerformanceMetrics object containing latency_ms, error_rate,
                memory_usage_mb, gpu_memory_usage_mb, and other performance data

        Side Effects:
            Calls _add_alert() for each threshold violation found, which stores
            alerts in the collector for later retrieval and notification.
        """
        alerts_to_check = [
            ('p99_latency_ms', metrics.latency_ms, 'Latency too high'),
            ('error_rate', metrics.error_rate, 'Error rate too high'),
            ('memory_usage_percent', metrics.memory_usage_mb / 1024, 'Memory usage too high'),
            ('gpu_memory_usage_percent', metrics.gpu_memory_usage_mb / 1024, 'GPU memory usage too high')
        ]

        for alert_name, value, message in alerts_to_check:
            if alert_name in self.alert_thresholds:
                threshold = self.alert_thresholds[alert_name]

                if value > threshold:
                    alert = Alert(
                        metric_name=alert_name,
                        threshold=threshold,
                        current_value=value,
                        severity='warning' if value < threshold * 1.5 else 'critical',
                        message=f"{message}: {value:.2f} > {threshold:.2f}"
                    )
                    self._add_alert(alert)

    def _check_accuracy_alerts(self, metrics: AccuracyMetrics) -> None:
        """Check for accuracy threshold violations and detect model degradation.

        Compares current accuracy metrics against baseline values to detect
        significant accuracy drops that may indicate model drift, data quality
        issues, or other problems requiring intervention.

        Args:
            metrics: AccuracyMetrics object containing accuracy scores including
                top1_accuracy, top5_accuracy, and other classification metrics
                that should be compared against baseline expectations

        Side Effects:
            Generates and stores alerts via _add_alert() when accuracy drops
            exceed configured thresholds, helping detect model degradation early.
        """
        if 'accuracy_drop' in self.alert_thresholds:
            threshold = self.alert_thresholds['accuracy_drop']

            # Check if accuracy dropped significantly from baseline
            baseline_accuracy = self._get_baseline_accuracy()
            if baseline_accuracy > 0:
                accuracy_drop = baseline_accuracy - metrics.accuracy

                if accuracy_drop > threshold:
                    alert = Alert(
                        metric_name='accuracy_drop',
                        threshold=threshold,
                        current_value=accuracy_drop,
                        severity='critical',
                        message=f"Accuracy dropped by {accuracy_drop:.4f} from baseline {baseline_accuracy:.4f}"
                    )
                    self._add_alert(alert)

    def _get_baseline_accuracy(self) -> float:
        """Get baseline accuracy from early measurements.

        Returns:
            Baseline accuracy value
        """
        if len(self.accuracy_history) < 100:
            return 0.0

        # Use first 100 samples as baseline
        baseline_samples = list(self.accuracy_history)[:100]
        return np.mean([sample.accuracy for sample in baseline_samples])

    def _add_alert(self, alert: Alert) -> None:
        """Add alert to active alerts.

        Args:
            alert: Alert to add
        """
        # Check if similar alert already exists
        for existing_alert in self.active_alerts:
            if (existing_alert.metric_name == alert.metric_name and
                abs(existing_alert.current_value - alert.current_value) < 0.1):
                return  # Don't add duplicate alert

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Log alert
        logger.warning(
            f"ALERT [{alert.severity}]: {alert.message} "
            f"(threshold: {alert.threshold}, current: {alert.current_value})"
        )

    def get_strategy_comparison(self, window_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Compare strategies over a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Strategy comparison metrics
        """
        cutoff_time = time.time() - (window_minutes * 60)
        comparison = {}

        for strategy_name in self.strategy_metrics.keys():
            # Filter recent metrics
            recent_perf = [
                m for m in self.strategy_metrics[strategy_name]
                if m.timestamp > cutoff_time
            ]
            recent_acc = [
                m for m in self.strategy_accuracy[strategy_name]
                if m.timestamp > cutoff_time
            ]

            if not recent_perf:
                continue

            # Calculate aggregated metrics
            strategy_stats = {
                'avg_latency_ms': np.mean([m.latency_ms for m in recent_perf]),
                'p95_latency_ms': np.percentile([m.latency_ms for m in recent_perf], 95),
                'p99_latency_ms': np.percentile([m.latency_ms for m in recent_perf], 99),
                'avg_throughput_fps': np.mean([m.throughput_fps for m in recent_perf]),
                'avg_error_rate': np.mean([m.error_rate for m in recent_perf]),
                'sample_count': len(recent_perf)
            }

            if recent_acc:
                strategy_stats.update({
                    'avg_accuracy': np.mean([m.accuracy for m in recent_acc]),
                    'avg_f1_score': np.mean([m.f1_score for m in recent_acc]),
                    'accuracy_samples': sum([m.num_samples for m in recent_acc])
                })

            comparison[strategy_name] = strategy_stats

        return comparison

    def get_percentile_metrics(
        self,
        strategy_name: Optional[str] = None,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get percentile metrics for latency analysis.

        Args:
            strategy_name: Optional strategy to filter by
            window_minutes: Time window in minutes

        Returns:
            Percentile metrics
        """
        cutoff_time = time.time() - (window_minutes * 60)

        # Select metrics to analyze
        if strategy_name and strategy_name in self.strategy_metrics:
            metrics_to_analyze = [
                m for m in self.strategy_metrics[strategy_name]
                if m.timestamp > cutoff_time
            ]
        else:
            metrics_to_analyze = [
                m for m in self.performance_history
                if m.timestamp > cutoff_time
            ]

        if not metrics_to_analyze:
            return {}

        latencies = [m.latency_ms for m in metrics_to_analyze]
        throughputs = [m.throughput_fps for m in metrics_to_analyze]

        percentiles = {}
        for p in self.config.monitoring.latency_percentiles:
            percentiles[f'p{int(p)}_latency_ms'] = np.percentile(latencies, p)

        percentiles.update({
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'mean_throughput_fps': np.mean(throughputs),
            'sample_count': len(latencies)
        })

        return percentiles

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts.

        Returns:
            List of active alerts
        """
        # Remove old alerts (older than 1 hour)
        cutoff_time = time.time() - 3600
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff_time
        ]

        return self.active_alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()
        logger.info("Cleared all active alerts")

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard.

        Returns:
            Dictionary with dashboard data
        """
        recent_comparison = self.get_strategy_comparison(window_minutes=30)
        percentile_metrics = self.get_percentile_metrics(window_minutes=30)
        active_alerts = self.get_active_alerts()

        dashboard_data = {
            'current_performance': {
                'latency_ms': self.current_metrics.latency_ms,
                'throughput_fps': self.current_metrics.throughput_fps,
                'error_rate': self.current_metrics.error_rate,
                'memory_usage_mb': self.current_metrics.memory_usage_mb,
                'gpu_memory_usage_mb': self.current_metrics.gpu_memory_usage_mb
            },
            'current_accuracy': {
                'accuracy': self.current_accuracy.accuracy,
                'f1_score': self.current_accuracy.f1_score,
                'confidence_score': self.current_accuracy.confidence_score
            },
            'strategy_comparison': recent_comparison,
            'percentile_metrics': percentile_metrics,
            'active_alerts': [
                {
                    'metric_name': alert.metric_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in active_alerts
            ],
            'total_samples': len(self.performance_history),
            'monitoring_uptime_hours': (time.time() - self.performance_history[0].timestamp) / 3600
            if self.performance_history else 0
        }

        return dashboard_data

    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export metrics to file.

        Args:
            filepath: Path to save metrics
            format: Export format ('json', 'csv')
        """
        try:
            if format == 'json':
                import json

                export_data = {
                    'performance_history': [
                        {
                            'latency_ms': m.latency_ms,
                            'throughput_fps': m.throughput_fps,
                            'memory_usage_mb': m.memory_usage_mb,
                            'error_rate': m.error_rate,
                            'timestamp': m.timestamp
                        }
                        for m in self.performance_history
                    ],
                    'accuracy_history': [
                        {
                            'accuracy': m.accuracy,
                            'f1_score': m.f1_score,
                            'num_samples': m.num_samples,
                            'timestamp': m.timestamp
                        }
                        for m in self.accuracy_history
                    ],
                    'alert_history': [
                        {
                            'metric_name': alert.metric_name,
                            'severity': alert.severity,
                            'message': alert.message,
                            'timestamp': alert.timestamp
                        }
                        for alert in self.alert_history
                    ]
                }

                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)

            elif format == 'csv':
                import pandas as pd

                # Convert performance history to DataFrame
                perf_df = pd.DataFrame([
                    {
                        'timestamp': m.timestamp,
                        'latency_ms': m.latency_ms,
                        'throughput_fps': m.throughput_fps,
                        'memory_usage_mb': m.memory_usage_mb,
                        'error_rate': m.error_rate
                    }
                    for m in self.performance_history
                ])

                perf_df.to_csv(filepath, index=False)

            logger.info(f"Metrics exported to {filepath} (format: {format})")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


class ModelDriftDetector:
    """Detector for model drift and performance degradation."""

    def __init__(self, config: Config, baseline_window: int = 1000):
        """Initialize drift detector.

        Args:
            config: Configuration object
            baseline_window: Number of samples for baseline calculation
        """
        self.config = config
        self.baseline_window = baseline_window

        # Baseline statistics
        self.baseline_accuracy: Optional[float] = None
        self.baseline_confidence: Optional[float] = None
        self.baseline_latency: Optional[float] = None

        # Recent measurements
        self.recent_accuracies: deque = deque(maxlen=baseline_window)
        self.recent_confidences: deque = deque(maxlen=baseline_window)
        self.recent_latencies: deque = deque(maxlen=baseline_window)

        # Drift detection thresholds
        self.accuracy_threshold = config.monitoring.accuracy_threshold
        self.confidence_threshold = 0.1  # 10% drop in confidence
        self.latency_threshold = 0.2  # 20% increase in latency

    def update_baseline(
        self,
        accuracy: float,
        confidence: float,
        latency: float
    ) -> None:
        """Update baseline statistics.

        Args:
            accuracy: Model accuracy
            confidence: Average confidence score
            latency: Average latency
        """
        self.recent_accuracies.append(accuracy)
        self.recent_confidences.append(confidence)
        self.recent_latencies.append(latency)

        # Calculate new baselines when we have enough samples
        if len(self.recent_accuracies) >= self.baseline_window:
            self.baseline_accuracy = np.mean(self.recent_accuracies)
            self.baseline_confidence = np.mean(self.recent_confidences)
            self.baseline_latency = np.mean(self.recent_latencies)

    def detect_drift(
        self,
        current_accuracy: float,
        current_confidence: float,
        current_latency: float
    ) -> Dict[str, bool]:
        """Detect drift in model performance across multiple metrics.

        Compares current performance metrics against established baselines to
        identify significant degradation in accuracy, confidence, or latency
        that may indicate model drift requiring retraining or intervention.

        Args:
            current_accuracy: Current model accuracy (0.0-1.0)
            current_confidence: Current model confidence score (0.0-1.0)
            current_latency: Current inference latency in milliseconds

        Returns:
            Dictionary with drift detection results containing keys:
            - 'accuracy_drift': True if accuracy dropped below threshold
            - 'confidence_drift': True if confidence dropped below threshold
            - 'latency_drift': True if latency increased beyond threshold
            - 'any_drift': True if any individual drift was detected

            Each boolean indicates whether drift was detected for that specific
            metric based on comparison with baseline values and configured
            thresholds.

        Example:
            >>> detector = ModelDriftDetector(baseline_accuracy=0.95, accuracy_threshold=0.02)
            >>> drift_results = detector.detect_drift(0.92, 0.85, 150.0)
            >>> if drift_results['any_drift']:
            ...     print("Model performance has degraded, consider retraining")
        """
        drift_detected = {
            'accuracy_drift': False,
            'confidence_drift': False,
            'latency_drift': False,
            'any_drift': False
        }

        if self.baseline_accuracy is not None:
            accuracy_drop = self.baseline_accuracy - current_accuracy
            if accuracy_drop > self.accuracy_threshold:
                drift_detected['accuracy_drift'] = True
                logger.warning(
                    f"Accuracy drift detected: {accuracy_drop:.4f} drop "
                    f"(baseline: {self.baseline_accuracy:.4f}, current: {current_accuracy:.4f})"
                )

        if self.baseline_confidence is not None:
            confidence_drop = self.baseline_confidence - current_confidence
            if confidence_drop > self.confidence_threshold:
                drift_detected['confidence_drift'] = True
                logger.warning(
                    f"Confidence drift detected: {confidence_drop:.4f} drop "
                    f"(baseline: {self.baseline_confidence:.4f}, current: {current_confidence:.4f})"
                )

        if self.baseline_latency is not None:
            latency_increase = (current_latency - self.baseline_latency) / self.baseline_latency
            if latency_increase > self.latency_threshold:
                drift_detected['latency_drift'] = True
                logger.warning(
                    f"Latency drift detected: {latency_increase:.2%} increase "
                    f"(baseline: {self.baseline_latency:.2f}ms, current: {current_latency:.2f}ms)"
                )

        drift_detected['any_drift'] = any([
            drift_detected['accuracy_drift'],
            drift_detected['confidence_drift'],
            drift_detected['latency_drift']
        ])

        return drift_detected


def calculate_top_k_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 5
) -> float:
    """Calculate top-k accuracy for model predictions.

    Public utility function for calculating top-k accuracy, useful for
    multi-class classification evaluation and testing.

    Args:
        predictions: Predicted class probabilities or logits, shape (n_samples, n_classes)
        targets: Ground truth labels, shape (n_samples,)
        k: Number of top predictions to consider (default: 5)

    Returns:
        Top-k accuracy as a float between 0 and 1, or NaN if calculation fails

    Example:
        >>> predictions = np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.1]])
        >>> targets = np.array([2, 1])
        >>> accuracy = calculate_top_k_accuracy(predictions, targets, k=2)
        >>> print(f"Top-2 accuracy: {accuracy:.2f}")
    """
    try:
        if len(predictions.shape) == 1:
            # Single class predictions, return regular accuracy
            return accuracy_score(targets, predictions)

        # Get top-k predictions
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
        correct = 0

        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1

        return correct / len(targets)

    except Exception as e:
        logger.warning(f"Failed to calculate top-{k} accuracy: {e}", exc_info=True)
        logger.info("Returning NaN to indicate calculation failure")
        return float('nan')