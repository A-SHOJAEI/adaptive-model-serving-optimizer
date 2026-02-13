"""Training orchestrator for adaptive model serving optimizer.

This module implements the core training logic including multi-armed bandit
algorithms for serving strategy selection and A/B testing orchestration.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import beta

from ..utils.config import Config
from ..models.model import BaseModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class ArmStatistics:
    """Statistics for a bandit arm (serving strategy)."""

    name: str
    pulls: int = 0
    rewards: List[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    avg_reward: float = 0.0
    confidence_bound: float = 0.0
    last_pull_time: float = 0.0

    # Performance metrics
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughputs: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update_reward(self, reward: float, latency: float, throughput: float, error_rate: float) -> None:
        """Update arm statistics with new reward.

        Args:
            reward: Reward value
            latency: Inference latency in ms
            throughput: Throughput in samples/sec
            error_rate: Error rate (0-1)
        """
        self.pulls += 1
        self.rewards.append(reward)

        if reward > 0:
            self.successes += 1
        else:
            self.failures += 1

        # Update running average
        self.avg_reward = sum(self.rewards) / len(self.rewards)

        # Update performance metrics
        self.latencies.append(latency)
        self.throughputs.append(throughput)
        self.error_rates.append(error_rate)
        self.last_pull_time = time.time()

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return 0.0
        return float(np.percentile(list(self.latencies), 95))

    def get_avg_throughput(self) -> float:
        """Get average throughput."""
        if not self.throughputs:
            return 0.0
        return float(np.mean(self.throughputs))

    def get_avg_error_rate(self) -> float:
        """Get average error rate."""
        if not self.error_rates:
            return 0.0
        return float(np.mean(self.error_rates))


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms."""

    def __init__(self, config: Config):
        """Initialize bandit algorithm.

        Args:
            config: Configuration object
        """
        self.config = config
        self.arms: Dict[str, ArmStatistics] = {}
        self.total_pulls = 0

    @abstractmethod
    def select_arm(self) -> str:
        """Select arm based on algorithm strategy.

        Returns:
            Selected arm name
        """
        pass

    @abstractmethod
    def update_arm(self, arm_name: str, reward: float, **kwargs) -> None:
        """Update arm with reward.

        Args:
            arm_name: Name of arm to update
            reward: Reward value
            **kwargs: Additional metrics
        """
        pass

    def add_arm(self, arm_name: str) -> None:
        """Add new arm to bandit.

        Args:
            arm_name: Name of arm to add
        """
        if arm_name not in self.arms:
            self.arms[arm_name] = ArmStatistics(name=arm_name)
            logger.info(f"Added arm: {arm_name}")

    def get_arm_stats(self) -> Dict[str, ArmStatistics]:
        """Get statistics for all arms.

        Returns:
            Dictionary of arm statistics
        """
        return self.arms.copy()

    def reset(self) -> None:
        """Reset all arm statistics."""
        for arm in self.arms.values():
            arm.pulls = 0
            arm.rewards.clear()
            arm.successes = 0
            arm.failures = 0
            arm.avg_reward = 0.0
            arm.confidence_bound = 0.0
            arm.latencies.clear()
            arm.throughputs.clear()
            arm.error_rates.clear()

        self.total_pulls = 0
        logger.info("Reset bandit statistics")


class UCBBandit(BanditAlgorithm):
    """Upper Confidence Bound bandit algorithm."""

    def __init__(self, config: Config):
        """Initialize UCB bandit.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.confidence_level = config.bandits.confidence_interval

    def select_arm(self) -> str:
        """Select arm using UCB strategy.

        Returns:
            Selected arm name
        """
        if not self.arms:
            raise ValueError("No arms available")

        # Ensure each arm is pulled at least once
        for arm_name, arm in self.arms.items():
            if arm.pulls == 0:
                logger.info(f"UCB: Selecting unexplored arm '{arm_name}' (total_pulls: {self.total_pulls})")
                return arm_name

        # Calculate UCB for each arm
        best_arm = None
        best_ucb = float('-inf')

        for arm_name, arm in self.arms.items():
            if arm.pulls == 0:
                ucb_value = float('inf')
            else:
                confidence_bonus = np.sqrt(
                    (2 * np.log(self.total_pulls)) / arm.pulls
                )
                ucb_value = arm.avg_reward + confidence_bonus

            arm.confidence_bound = ucb_value

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_arm = arm_name

        # Enhanced UCB decision logging
        arm_summary = {name: {'reward': arm.avg_reward, 'ucb': arm.confidence_bound, 'pulls': arm.pulls}
                      for name, arm in self.arms.items()}
        logger.info(f"UCB selected '{best_arm}' (UCB: {best_ucb:.3f}, reward: {self.arms[best_arm].avg_reward:.3f}) "
                   f"from {len(self.arms)} arms after {self.total_pulls} total pulls")
        logger.debug(f"All arm stats: {arm_summary}")

        return best_arm

    def update_arm(self, arm_name: str, reward: float, **kwargs) -> None:
        """Update arm with UCB.

        Args:
            arm_name: Name of arm to update
            reward: Reward value
            **kwargs: Additional metrics (latency, throughput, error_rate)
        """
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        latency = kwargs.get('latency', None)
        throughput = kwargs.get('throughput', None)
        error_rate = kwargs.get('error_rate', None)

        # Warn if critical metrics are missing
        if latency is None:
            logger.warning(f"Missing latency metric for arm {arm_name}, using 0.0")
            latency = 0.0
        if throughput is None:
            logger.warning(f"Missing throughput metric for arm {arm_name}, using 0.0")
            throughput = 0.0
        if error_rate is None:
            logger.warning(f"Missing error_rate metric for arm {arm_name}, using 0.0")
            error_rate = 0.0

        arm.update_reward(reward, latency, throughput, error_rate)
        self.total_pulls += 1


class ThompsonSamplingBandit(BanditAlgorithm):
    """Thompson Sampling bandit algorithm."""

    def __init__(self, config: Config):
        """Initialize Thompson Sampling bandit.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.alpha = config.bandits.alpha
        self.beta_param = config.bandits.beta

    def select_arm(self) -> str:
        """Select arm using Thompson Sampling.

        Returns:
            Selected arm name
        """
        if not self.arms:
            raise ValueError("No arms available")

        best_arm = None
        best_sample = float('-inf')

        for arm_name, arm in self.arms.items():
            # Beta parameters for Thompson Sampling
            alpha_param = self.alpha + arm.successes
            beta_param = self.beta_param + arm.failures

            # Sample from beta distribution
            sample = np.random.beta(alpha_param, beta_param)

            if sample > best_sample:
                best_sample = sample
                best_arm = arm_name

        # Enhanced Thompson Sampling decision logging
        beta_params = {name: {'alpha': self.alpha + arm.successes, 'beta': self.beta_param + arm.failures}
                      for name, arm in self.arms.items()}
        logger.info(f"Thompson Sampling selected '{best_arm}' (sample: {best_sample:.3f}, "
                   f"α={self.alpha + self.arms[best_arm].successes:.1f}, "
                   f"β={self.beta_param + self.arms[best_arm].failures:.1f})")
        logger.debug(f"Beta parameters for all arms: {beta_params}")

        return best_arm

    def update_arm(self, arm_name: str, reward: float, **kwargs) -> None:
        """Update arm with Thompson Sampling.

        Args:
            arm_name: Name of arm to update
            reward: Reward value
            **kwargs: Additional metrics (latency, throughput, error_rate)
        """
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        latency = kwargs.get('latency', None)
        throughput = kwargs.get('throughput', None)
        error_rate = kwargs.get('error_rate', None)

        # Warn if critical metrics are missing
        if latency is None:
            logger.warning(f"Missing latency metric for arm {arm_name}, using 0.0")
            latency = 0.0
        if throughput is None:
            logger.warning(f"Missing throughput metric for arm {arm_name}, using 0.0")
            throughput = 0.0
        if error_rate is None:
            logger.warning(f"Missing error_rate metric for arm {arm_name}, using 0.0")
            error_rate = 0.0

        arm.update_reward(reward, latency, throughput, error_rate)
        self.total_pulls += 1


class EpsilonGreedyBandit(BanditAlgorithm):
    """Epsilon-greedy bandit algorithm."""

    def __init__(self, config: Config):
        """Initialize epsilon-greedy bandit.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.epsilon = config.bandits.epsilon
        self.epsilon_decay = config.bandits.exploration_decay
        self.min_epsilon = config.bandits.min_exploration

    def select_arm(self) -> str:
        """Select arm using epsilon-greedy strategy.

        Returns:
            Selected arm name
        """
        if not self.arms:
            raise ValueError("No arms available")

        # Decay epsilon over time
        current_epsilon = max(
            self.min_epsilon,
            self.epsilon * (self.epsilon_decay ** self.total_pulls)
        )

        # Explore with probability epsilon
        if random.random() < current_epsilon:
            selected_arm = random.choice(list(self.arms.keys()))
            logger.info(f"Epsilon-Greedy: EXPLORE '{selected_arm}' (ε={current_epsilon:.3f}, "
                       f"decay from {self.epsilon:.3f} after {self.total_pulls} pulls)")
            return selected_arm

        # Exploit best arm
        best_arm = None
        best_reward = float('-inf')

        for arm_name, arm in self.arms.items():
            if arm.avg_reward > best_reward:
                best_reward = arm.avg_reward
                best_arm = arm_name

        final_arm = best_arm or list(self.arms.keys())[0]
        reward_summary = {name: arm.avg_reward for name, arm in self.arms.items()}
        logger.info(f"Epsilon-Greedy: EXPLOIT '{final_arm}' (reward: {best_reward:.3f}, "
                   f"ε={current_epsilon:.3f}) from {reward_summary}")
        return final_arm

    def update_arm(self, arm_name: str, reward: float, **kwargs) -> None:
        """Update arm with epsilon-greedy.

        Args:
            arm_name: Name of arm to update
            reward: Reward value
            **kwargs: Additional metrics (latency, throughput, error_rate)
        """
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        latency = kwargs.get('latency', None)
        throughput = kwargs.get('throughput', None)
        error_rate = kwargs.get('error_rate', None)

        # Warn if critical metrics are missing
        if latency is None:
            logger.warning(f"Missing latency metric for arm {arm_name}, using 0.0")
            latency = 0.0
        if throughput is None:
            logger.warning(f"Missing throughput metric for arm {arm_name}, using 0.0")
            throughput = 0.0
        if error_rate is None:
            logger.warning(f"Missing error_rate metric for arm {arm_name}, using 0.0")
            error_rate = 0.0

        arm.update_reward(reward, latency, throughput, error_rate)
        self.total_pulls += 1


class ServingStrategyOptimizer:
    """Orchestrator for adaptive serving strategy selection."""

    def __init__(self, config: Config):
        """Initialize serving strategy optimizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.bandit = self._create_bandit(config.bandits.algorithm)
        self.serving_adapters: Dict[str, BaseModelAdapter] = {}

        # Performance tracking
        self.experiment_history: List[Dict[str, Any]] = []
        self.update_frequency = config.bandits.update_frequency
        self.updates_since_last = 0

        # Convergence tracking
        self.convergence_window = config.bandits.window_size
        self.convergence_threshold = config.reward.convergence_threshold

    def _create_bandit(self, algorithm: str) -> BanditAlgorithm:
        """Create bandit algorithm instance.

        Args:
            algorithm: Algorithm name

        Returns:
            Bandit algorithm instance
        """
        algorithm = algorithm.lower()

        if algorithm == 'ucb':
            return UCBBandit(self.config)
        elif algorithm == 'thompson':
            return ThompsonSamplingBandit(self.config)
        elif algorithm == 'epsilon_greedy':
            return EpsilonGreedyBandit(self.config)
        else:
            raise ValueError(f"Unsupported bandit algorithm: {algorithm}")

    def register_serving_adapter(self, name: str, adapter: BaseModelAdapter) -> None:
        """Register a serving adapter as a bandit arm.

        Args:
            name: Name of serving strategy
            adapter: Model adapter instance
        """
        self.serving_adapters[name] = adapter
        self.bandit.add_arm(name)
        logger.info(f"Registered serving adapter: {name}")

    def select_serving_strategy(self) -> Tuple[str, BaseModelAdapter]:
        """Select optimal serving strategy.

        Returns:
            Tuple of (strategy_name, adapter)
        """
        strategy_name = self.bandit.select_arm()
        adapter = self.serving_adapters[strategy_name]
        return strategy_name, adapter

    def update_strategy_performance(
        self,
        strategy_name: str,
        latency: float,
        throughput: float,
        accuracy: float,
        error_rate: float
    ) -> None:
        """Update strategy performance metrics.

        Args:
            strategy_name: Name of strategy
            latency: Inference latency in milliseconds
            throughput: Throughput in samples per second
            accuracy: Model accuracy (0-1)
            error_rate: Error rate (0-1)
        """
        # Calculate composite reward
        reward = self._calculate_reward(latency, throughput, accuracy, error_rate)

        # Update bandit
        self.bandit.update_arm(
            strategy_name,
            reward,
            latency=latency,
            throughput=throughput,
            error_rate=error_rate
        )

        # Record experiment data
        self.experiment_history.append({
            'timestamp': time.time(),
            'strategy': strategy_name,
            'reward': reward,
            'latency': latency,
            'throughput': throughput,
            'accuracy': accuracy,
            'error_rate': error_rate
        })

        self.updates_since_last += 1

        # Periodic logging
        if self.updates_since_last >= self.update_frequency:
            self._log_performance_summary()
            self.updates_since_last = 0

    def _calculate_reward(
        self,
        latency: float,
        throughput: float,
        accuracy: float,
        error_rate: float
    ) -> float:
        """Calculate composite reward for serving strategy.

        Args:
            latency: Inference latency in milliseconds
            throughput: Throughput in samples per second
            accuracy: Model accuracy (0-1)
            error_rate: Error rate (0-1)

        Returns:
            Composite reward value
        """
        # Validate input metrics
        if not (0 <= accuracy <= 1):
            logger.warning(f"Accuracy out of expected range [0,1]: {accuracy}")
        if not (0 <= error_rate <= 1):
            logger.warning(f"Error rate out of expected range [0,1]: {error_rate}")
        if latency < 0:
            logger.warning(f"Negative latency detected: {latency}")
        if throughput < 0:
            logger.warning(f"Negative throughput detected: {throughput}")

        # Calculate component scores using configurable normalization
        norm = self.config.reward.normalization
        latency_score = max(0, 1.0 - latency / norm['max_latency_ms'])
        throughput_score = min(1.0, throughput / norm['ideal_throughput'])
        accuracy_score = max(0, (accuracy - norm['min_accuracy']) / (1.0 - norm['min_accuracy']))
        error_score = max(0, 1.0 - (error_rate / norm['max_error_rate']))

        # Use configurable weights
        weights = self.config.reward.weights

        # Log component breakdown for debugging
        logger.debug(f"Reward components - Latency: {latency_score:.3f}, "
                    f"Throughput: {throughput_score:.3f}, Accuracy: {accuracy_score:.3f}, "
                    f"Error: {error_score:.3f}")

        reward = (
            weights['latency'] * latency_score +
            weights['throughput'] * throughput_score +
            weights['accuracy'] * accuracy_score +
            weights['error'] * error_score
        )

        return reward

    def _log_performance_summary(self) -> None:
        """Log comprehensive performance summary for all serving strategies.

        This method provides detailed performance analysis including:
        1. Current timestamp and experiment count
        2. Per-strategy statistics (pulls, average reward, confidence bounds)
        3. Performance metrics (P95 latency, average throughput, error rates)
        4. Current best strategy and its performance characteristics
        5. Exploration vs exploitation balance analysis

        The summary is logged at INFO level and provides insights into:
        - Which strategies are performing well/poorly
        - Whether the bandit algorithm is converging
        - Current performance characteristics of each strategy
        - Statistical confidence in the performance measurements

        Used for:
        - Periodic monitoring of bandit algorithm performance
        - Debugging strategy selection issues
        - Performance analysis and reporting
        - Operational visibility into model serving decisions
        """
        current_time = time.time()
        logger.info("=== Serving Strategy Performance Summary ===")
        logger.info(f"Timestamp: {current_time:.2f}")

        for strategy_name, arm in self.bandit.get_arm_stats().items():
            if arm.pulls > 0:
                logger.info(
                    f"{strategy_name}: "
                    f"pulls={arm.pulls}, "
                    f"avg_reward={arm.avg_reward:.4f}, "
                    f"p95_latency={arm.get_p95_latency():.2f}ms, "
                    f"avg_throughput={arm.get_avg_throughput():.2f}/s, "
                    f"error_rate={arm.get_avg_error_rate():.4f}"
                )

    def check_convergence(self) -> bool:
        """Check if optimization has converged.

        Returns:
            True if converged, False otherwise
        """
        if len(self.experiment_history) < self.convergence_window:
            return False

        # Get recent history
        recent_history = self.experiment_history[-self.convergence_window:]

        # Group by strategy and calculate stability
        strategy_rewards = defaultdict(list)
        for entry in recent_history:
            strategy_rewards[entry['strategy']].append(entry['reward'])

        # Check if reward variance is low for all strategies
        for strategy, rewards in strategy_rewards.items():
            if len(rewards) < 10:  # Need minimum samples
                continue

            reward_std = np.std(rewards)
            reward_mean = np.mean(rewards)

            if reward_mean > 0 and (reward_std / reward_mean) > self.convergence_threshold:
                return False

        logger.info("Optimization has converged")
        return True

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report.

        Returns:
            Dictionary containing optimization results
        """
        arm_stats = self.bandit.get_arm_stats()

        # Find best performing strategy
        best_strategy = None
        best_reward = float('-inf')

        for strategy_name, arm in arm_stats.items():
            if arm.avg_reward > best_reward:
                best_reward = arm.avg_reward
                best_strategy = strategy_name

        # Calculate metrics improvements
        improvements = {}
        if len(self.experiment_history) > 0:
            baseline_metrics = self._calculate_baseline_metrics()
            current_metrics = self._calculate_current_metrics()

            for metric in ['latency', 'throughput', 'accuracy']:
                if baseline_metrics[metric] > 0:
                    improvement = (
                        (current_metrics[metric] - baseline_metrics[metric]) /
                        baseline_metrics[metric] * 100
                    )
                    improvements[f'{metric}_improvement_percent'] = improvement

        report = {
            'best_strategy': best_strategy,
            'best_reward': best_reward,
            'total_experiments': len(self.experiment_history),
            'convergence_achieved': self.check_convergence(),
            'strategy_statistics': {
                name: {
                    'pulls': arm.pulls,
                    'avg_reward': arm.avg_reward,
                    'p95_latency_ms': arm.get_p95_latency(),
                    'avg_throughput': arm.get_avg_throughput(),
                    'error_rate': arm.get_avg_error_rate()
                }
                for name, arm in arm_stats.items()
            },
            'performance_improvements': improvements
        }

        return report

    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline metrics from early experiments for comparison.

        Computes baseline performance metrics from the first N experiments
        (where N is configurable via reward.baseline_window_size) to establish
        a reference point for measuring performance improvements.

        The baseline represents initial system performance before optimization
        and is used to:
        1. Calculate percentage improvements over time
        2. Detect performance regressions
        3. Normalize reward calculations
        4. Generate performance improvement reports

        Returns:
            Dictionary containing baseline metrics:
            - 'latency': Mean latency from early experiments (ms)
            - 'throughput': Mean throughput from early experiments (samples/sec)
            - 'accuracy': Mean accuracy from early experiments (0-1)

        Note:
            - Returns zeros if no experiments are available
            - Window size is configurable via config.reward.baseline_window_size
            - Baseline is calculated once and cached for consistency
            - Used primarily for reporting and analysis, not for reward calculation
        """
        window_size = self.config.reward.baseline_window_size
        early_experiments = self.experiment_history[:window_size]  # First N experiments

        if not early_experiments:
            return {'latency': 0.0, 'throughput': 0.0, 'accuracy': 0.0}

        return {
            'latency': np.mean([exp['latency'] for exp in early_experiments]),
            'throughput': np.mean([exp['throughput'] for exp in early_experiments]),
            'accuracy': np.mean([exp['accuracy'] for exp in early_experiments])
        }

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current metrics from recent experiments for comparison.

        Computes current performance metrics from the most recent N experiments
        (where N is configurable via reward.baseline_window_size) to assess
        current system performance.

        Current metrics represent recent system performance and are used to:
        1. Compare against baseline to calculate improvements
        2. Detect recent performance trends
        3. Generate real-time performance reports
        4. Trigger alerts for performance degradation

        Returns:
            Dictionary containing current metrics:
            - 'latency': Mean latency from recent experiments (ms)
            - 'throughput': Mean throughput from recent experiments (samples/sec)
            - 'accuracy': Mean accuracy from recent experiments (0-1)

        Note:
            - Returns zeros if no experiments are available
            - Window size matches baseline calculation for fair comparison
            - Recalculated on each call to reflect most recent performance
            - Excludes experiments currently in progress
        """
        window_size = self.config.reward.baseline_window_size
        recent_experiments = self.experiment_history[-window_size:]  # Last N experiments

        if not recent_experiments:
            return {'latency': 0.0, 'throughput': 0.0, 'accuracy': 0.0}

        return {
            'latency': np.mean([exp['latency'] for exp in recent_experiments]),
            'throughput': np.mean([exp['throughput'] for exp in recent_experiments]),
            'accuracy': np.mean([exp['accuracy'] for exp in recent_experiments])
        }

    def save_experiment_data(self, filepath: str) -> None:
        """Save experiment data to file.

        Args:
            filepath: Path to save experiment data
        """
        try:
            import json

            data = {
                'config': {
                    'algorithm': self.config.bandits.algorithm,
                    'epsilon': self.config.bandits.epsilon,
                    'alpha': self.config.bandits.alpha,
                    'beta': self.config.bandits.beta
                },
                'experiment_history': self.experiment_history,
                'arm_statistics': {
                    name: {
                        'pulls': arm.pulls,
                        'successes': arm.successes,
                        'failures': arm.failures,
                        'avg_reward': arm.avg_reward
                    }
                    for name, arm in self.bandit.get_arm_stats().items()
                }
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Experiment data saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save experiment data: {e}")

    def load_experiment_data(self, filepath: str) -> None:
        """Load experiment data from file.

        Args:
            filepath: Path to load experiment data from
        """
        try:
            import json

            with open(filepath, 'r') as f:
                data = json.load(f)

            self.experiment_history = data.get('experiment_history', [])

            # Restore arm statistics
            arm_stats = data.get('arm_statistics', {})
            for name, stats in arm_stats.items():
                if name in self.bandit.arms:
                    arm = self.bandit.arms[name]
                    arm.pulls = stats['pulls']
                    arm.successes = stats['successes']
                    arm.failures = stats['failures']
                    arm.avg_reward = stats['avg_reward']

            logger.info(f"Experiment data loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load experiment data: {e}")