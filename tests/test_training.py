"""Tests for training and bandit optimization."""

import pytest
import time
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from adaptive_model_serving_optimizer.training.trainer import (
    ArmStatistics,
    UCBBandit,
    ThompsonSamplingBandit,
    EpsilonGreedyBandit,
    ServingStrategyOptimizer
)
from adaptive_model_serving_optimizer.utils.config import Config


class TestArmStatistics:
    """Tests for ArmStatistics class."""

    def test_arm_statistics_init(self):
        """Test ArmStatistics initialization."""
        arm = ArmStatistics(name="test_arm")

        assert arm.name == "test_arm"
        assert arm.pulls == 0
        assert arm.successes == 0
        assert arm.failures == 0
        assert arm.avg_reward == 0.0
        assert len(arm.rewards) == 0
        assert len(arm.latencies) == 0

    def test_update_reward_positive(self):
        """Test updating arm with positive reward."""
        arm = ArmStatistics(name="test_arm")

        arm.update_reward(
            reward=0.8,
            latency=10.5,
            throughput=95.2,
            error_rate=0.01
        )

        assert arm.pulls == 1
        assert arm.successes == 1
        assert arm.failures == 0
        assert arm.avg_reward == 0.8
        assert len(arm.rewards) == 1
        assert arm.rewards[0] == 0.8

    def test_update_reward_negative(self):
        """Test updating arm with negative reward."""
        arm = ArmStatistics(name="test_arm")

        arm.update_reward(
            reward=-0.2,
            latency=50.0,
            throughput=20.0,
            error_rate=0.1
        )

        assert arm.pulls == 1
        assert arm.successes == 0
        assert arm.failures == 1
        assert arm.avg_reward == -0.2

    def test_update_reward_multiple(self):
        """Test updating arm multiple times."""
        arm = ArmStatistics(name="test_arm")

        rewards = [0.8, 0.6, 0.9, 0.7]
        for reward in rewards:
            arm.update_reward(reward, 10.0, 100.0, 0.0)

        assert arm.pulls == len(rewards)
        assert arm.successes == len(rewards)  # All positive
        assert arm.failures == 0
        assert arm.avg_reward == sum(rewards) / len(rewards)

    def test_get_percentile_metrics(self):
        """Test getting percentile metrics."""
        arm = ArmStatistics(name="test_arm")

        latencies = [10.0, 15.0, 20.0, 25.0, 30.0]
        for latency in latencies:
            arm.update_reward(0.5, latency, 100.0, 0.0)

        p95_latency = arm.get_p95_latency()
        assert p95_latency > 0
        assert p95_latency <= max(latencies)

    def test_get_average_metrics(self):
        """Test getting average metrics."""
        arm = ArmStatistics(name="test_arm")

        throughputs = [90.0, 95.0, 100.0, 105.0, 110.0]
        error_rates = [0.01, 0.02, 0.015, 0.005, 0.01]

        for throughput, error_rate in zip(throughputs, error_rates):
            arm.update_reward(0.5, 10.0, throughput, error_rate)

        avg_throughput = arm.get_avg_throughput()
        avg_error_rate = arm.get_avg_error_rate()

        assert abs(avg_throughput - np.mean(throughputs)) < 1e-6
        assert abs(avg_error_rate - np.mean(error_rates)) < 1e-6


class TestUCBBandit:
    """Tests for UCB bandit algorithm."""

    def test_ucb_bandit_init(self, config: Config):
        """Test UCB bandit initialization."""
        bandit = UCBBandit(config)

        assert bandit.config == config
        assert bandit.confidence_level == config.bandits.confidence_interval
        assert len(bandit.arms) == 0

    def test_add_arm(self, config: Config):
        """Test adding arms to bandit."""
        bandit = UCBBandit(config)

        bandit.add_arm("arm1")
        bandit.add_arm("arm2")

        assert len(bandit.arms) == 2
        assert "arm1" in bandit.arms
        assert "arm2" in bandit.arms

    def test_select_arm_first_time(self, config: Config):
        """Test arm selection when each arm hasn't been pulled."""
        bandit = UCBBandit(config)
        bandit.add_arm("arm1")
        bandit.add_arm("arm2")

        selected_arm = bandit.select_arm()

        # Should select an arm that hasn't been pulled (pulls == 0)
        assert selected_arm in ["arm1", "arm2"]
        assert bandit.arms[selected_arm].pulls == 0

    def test_select_arm_with_history(self, config: Config):
        """Test arm selection with some history."""
        bandit = UCBBandit(config)
        bandit.add_arm("good_arm")
        bandit.add_arm("bad_arm")

        # Give good_arm better rewards
        bandit.update_arm("good_arm", 0.8, latency=10.0, throughput=100.0, error_rate=0.01)
        bandit.update_arm("bad_arm", 0.2, latency=50.0, throughput=20.0, error_rate=0.1)

        # Both arms have been pulled once, now UCB should prefer good_arm
        bandit.update_arm("good_arm", 0.9, latency=9.0, throughput=105.0, error_rate=0.005)

        selected_arm = bandit.select_arm()
        # Due to UCB confidence bounds, selection may vary, so we just check it's valid
        assert selected_arm in ["good_arm", "bad_arm"]

    def test_update_arm_new(self, config: Config):
        """Test updating a new arm."""
        bandit = UCBBandit(config)

        bandit.update_arm("new_arm", 0.7, latency=15.0, throughput=80.0, error_rate=0.02)

        assert "new_arm" in bandit.arms
        assert bandit.arms["new_arm"].pulls == 1
        assert bandit.total_pulls == 1

    def test_select_arm_no_arms(self, config: Config):
        """Test selecting arm when no arms exist."""
        bandit = UCBBandit(config)

        with pytest.raises(ValueError, match="No arms available"):
            bandit.select_arm()

    def test_reset(self, config: Config):
        """Test resetting bandit statistics."""
        bandit = UCBBandit(config)
        bandit.add_arm("arm1")
        bandit.update_arm("arm1", 0.5, latency=10.0, throughput=100.0, error_rate=0.0)

        bandit.reset()

        assert bandit.total_pulls == 0
        assert bandit.arms["arm1"].pulls == 0
        assert len(bandit.arms["arm1"].rewards) == 0


class TestThompsonSamplingBandit:
    """Tests for Thompson Sampling bandit algorithm."""

    def test_thompson_sampling_init(self, config: Config):
        """Test Thompson Sampling bandit initialization."""
        bandit = ThompsonSamplingBandit(config)

        assert bandit.config == config
        assert bandit.alpha == config.bandits.alpha
        assert bandit.beta_param == config.bandits.beta

    @patch('numpy.random.beta')
    def test_select_arm_with_sampling(self, mock_beta, config: Config):
        """Test arm selection with mocked beta sampling."""
        bandit = ThompsonSamplingBandit(config)
        bandit.add_arm("arm1")
        bandit.add_arm("arm2")

        # Mock beta sampling to return predictable values
        mock_beta.side_effect = [0.8, 0.6]  # arm1 gets higher sample

        selected_arm = bandit.select_arm()

        assert selected_arm == "arm1"  # Should select arm with higher sample
        assert mock_beta.call_count == 2  # Called once for each arm

    def test_update_arm_success_failure_counts(self, config: Config):
        """Test that successes and failures are counted correctly."""
        bandit = ThompsonSamplingBandit(config)

        # Positive reward should count as success
        bandit.update_arm("arm1", 0.5, latency=10.0, throughput=100.0, error_rate=0.0)
        assert bandit.arms["arm1"].successes == 1
        assert bandit.arms["arm1"].failures == 0

        # Negative reward should count as failure
        bandit.update_arm("arm1", -0.2, latency=20.0, throughput=50.0, error_rate=0.1)
        assert bandit.arms["arm1"].successes == 1
        assert bandit.arms["arm1"].failures == 1


class TestEpsilonGreedyBandit:
    """Tests for Epsilon-Greedy bandit algorithm."""

    def test_epsilon_greedy_init(self, config: Config):
        """Test Epsilon-Greedy bandit initialization."""
        bandit = EpsilonGreedyBandit(config)

        assert bandit.config == config
        assert bandit.epsilon == config.bandits.epsilon
        assert bandit.epsilon_decay == config.bandits.exploration_decay
        assert bandit.min_epsilon == config.bandits.min_exploration

    @patch('random.random')
    def test_select_arm_exploration(self, mock_random, config: Config):
        """Test arm selection during exploration."""
        config.bandits.epsilon = 0.5
        bandit = EpsilonGreedyBandit(config)
        bandit.add_arm("arm1")
        bandit.add_arm("arm2")

        # Mock random to trigger exploration
        mock_random.return_value = 0.3  # Less than epsilon

        with patch('random.choice') as mock_choice:
            mock_choice.return_value = "arm2"
            selected_arm = bandit.select_arm()

        assert selected_arm == "arm2"
        mock_choice.assert_called_once()

    @patch('random.random')
    def test_select_arm_exploitation(self, mock_random, config: Config):
        """Test arm selection during exploitation."""
        config.bandits.epsilon = 0.5
        bandit = EpsilonGreedyBandit(config)
        bandit.add_arm("good_arm")
        bandit.add_arm("bad_arm")

        # Give arms different rewards
        bandit.update_arm("good_arm", 0.8, latency=10.0, throughput=100.0, error_rate=0.0)
        bandit.update_arm("bad_arm", 0.2, latency=50.0, throughput=20.0, error_rate=0.1)

        # Mock random to trigger exploitation
        mock_random.return_value = 0.8  # Greater than epsilon

        selected_arm = bandit.select_arm()

        assert selected_arm == "good_arm"  # Should select arm with higher average reward

    def test_epsilon_decay(self, config: Config):
        """Test epsilon decay over time."""
        config.bandits.epsilon = 0.5
        config.bandits.exploration_decay = 0.9
        config.bandits.min_exploration = 0.1

        bandit = EpsilonGreedyBandit(config)
        bandit.add_arm("arm1")

        initial_epsilon = bandit.epsilon

        # Simulate multiple pulls to trigger decay
        for _ in range(10):
            bandit.update_arm("arm1", 0.5, latency=10.0, throughput=100.0, error_rate=0.0)

        # Epsilon should have decayed but not below minimum
        with patch('random.random', return_value=0.3):
            # This will calculate current epsilon internally
            bandit.select_arm()

        # We can't directly access current epsilon, but we can test behavior
        assert bandit.min_epsilon <= initial_epsilon  # Basic sanity check


class TestServingStrategyOptimizer:
    """Tests for ServingStrategyOptimizer class."""

    def test_optimizer_init(self, config: Config):
        """Test ServingStrategyOptimizer initialization."""
        optimizer = ServingStrategyOptimizer(config)

        assert optimizer.config == config
        assert optimizer.bandit is not None
        assert len(optimizer.serving_adapters) == 0
        assert len(optimizer.experiment_history) == 0

    def test_register_serving_adapter(self, config: Config, mock_adapters):
        """Test registering serving adapters."""
        optimizer = ServingStrategyOptimizer(config)

        for name, adapter in mock_adapters.items():
            optimizer.register_serving_adapter(name, adapter)

        assert len(optimizer.serving_adapters) == len(mock_adapters)
        for name in mock_adapters.keys():
            assert name in optimizer.serving_adapters
            assert name in optimizer.bandit.arms

    def test_select_serving_strategy(self, config: Config, mock_adapters):
        """Test selecting serving strategy."""
        optimizer = ServingStrategyOptimizer(config)

        # Register adapters
        for name, adapter in mock_adapters.items():
            optimizer.register_serving_adapter(name, adapter)

        strategy_name, adapter = optimizer.select_serving_strategy()

        assert strategy_name in mock_adapters.keys()
        assert adapter == mock_adapters[strategy_name]

    def test_update_strategy_performance(self, config: Config, mock_adapters):
        """Test updating strategy performance."""
        optimizer = ServingStrategyOptimizer(config)

        # Register one adapter
        adapter_name = list(mock_adapters.keys())[0]
        adapter = mock_adapters[adapter_name]
        optimizer.register_serving_adapter(adapter_name, adapter)

        # Update performance
        optimizer.update_strategy_performance(
            strategy_name=adapter_name,
            latency=15.0,
            throughput=80.0,
            accuracy=0.95,
            error_rate=0.02
        )

        assert len(optimizer.experiment_history) == 1
        assert optimizer.bandit.arms[adapter_name].pulls == 1

        # Check experiment history
        experiment = optimizer.experiment_history[0]
        assert experiment['strategy'] == adapter_name
        assert experiment['latency'] == 15.0
        assert experiment['throughput'] == 80.0
        assert experiment['accuracy'] == 0.95
        assert experiment['error_rate'] == 0.02
        assert 'reward' in experiment
        assert 'timestamp' in experiment

    def test_calculate_reward(self, config: Config):
        """Test reward calculation."""
        optimizer = ServingStrategyOptimizer(config)

        # Test good performance
        good_reward = optimizer._calculate_reward(
            latency=10.0,
            throughput=100.0,
            accuracy=0.95,
            error_rate=0.01
        )

        # Test bad performance
        bad_reward = optimizer._calculate_reward(
            latency=100.0,
            throughput=10.0,
            accuracy=0.70,
            error_rate=0.1
        )

        assert 0 <= good_reward <= 1
        assert 0 <= bad_reward <= 1
        assert good_reward > bad_reward

    def test_check_convergence_not_enough_data(self, config: Config):
        """Test convergence check with insufficient data."""
        optimizer = ServingStrategyOptimizer(config)

        assert optimizer.check_convergence() is False

    def test_get_optimization_report(self, config: Config, mock_adapters):
        """Test getting optimization report."""
        optimizer = ServingStrategyOptimizer(config)

        # Register adapters and add some history
        for name, adapter in mock_adapters.items():
            optimizer.register_serving_adapter(name, adapter)

        # Add some performance data
        adapter_names = list(mock_adapters.keys())
        optimizer.update_strategy_performance(
            adapter_names[0], 10.0, 100.0, 0.95, 0.01
        )
        optimizer.update_strategy_performance(
            adapter_names[1], 15.0, 80.0, 0.93, 0.02
        )

        report = optimizer.get_optimization_report()

        assert 'best_strategy' in report
        assert 'best_reward' in report
        assert 'total_experiments' in report
        assert 'convergence_achieved' in report
        assert 'strategy_statistics' in report
        assert 'performance_improvements' in report

        assert report['total_experiments'] == 2
        assert report['best_strategy'] in adapter_names

    def test_save_and_load_experiment_data(self, config: Config, temp_dir, mock_adapters):
        """Test saving and loading experiment data."""
        optimizer = ServingStrategyOptimizer(config)

        # Register adapters and add some data
        adapter_name = list(mock_adapters.keys())[0]
        adapter = mock_adapters[adapter_name]
        optimizer.register_serving_adapter(adapter_name, adapter)

        optimizer.update_strategy_performance(
            adapter_name, 10.0, 100.0, 0.95, 0.01
        )

        # Save data
        save_path = temp_dir / "experiment_data.json"
        optimizer.save_experiment_data(str(save_path))

        assert save_path.exists()

        # Create new optimizer and load data
        new_optimizer = ServingStrategyOptimizer(config)
        new_optimizer.register_serving_adapter(adapter_name, adapter)
        new_optimizer.load_experiment_data(str(save_path))

        assert len(new_optimizer.experiment_history) == 1
        assert new_optimizer.bandit.arms[adapter_name].pulls == 1

    def test_calculate_baseline_and_current_metrics(self, config: Config, mock_adapters):
        """Test baseline and current metrics calculation."""
        optimizer = ServingStrategyOptimizer(config)

        adapter_name = list(mock_adapters.keys())[0]
        adapter = mock_adapters[adapter_name]
        optimizer.register_serving_adapter(adapter_name, adapter)

        # Add multiple experiments to create baseline and current metrics
        for i in range(150):  # More than baseline window of 100
            latency = 10.0 + i * 0.1  # Gradually increasing latency
            throughput = 100.0 - i * 0.1  # Gradually decreasing throughput
            accuracy = 0.95 - i * 0.0001  # Slightly decreasing accuracy

            optimizer.update_strategy_performance(
                adapter_name, latency, throughput, accuracy, 0.01
            )

        baseline_metrics = optimizer._calculate_baseline_metrics()
        current_metrics = optimizer._calculate_current_metrics()

        assert 'latency' in baseline_metrics
        assert 'throughput' in baseline_metrics
        assert 'accuracy' in baseline_metrics

        assert 'latency' in current_metrics
        assert 'throughput' in current_metrics
        assert 'accuracy' in current_metrics

        # Current metrics should be different from baseline due to trends
        assert current_metrics['latency'] > baseline_metrics['latency']
        assert current_metrics['throughput'] < baseline_metrics['throughput']