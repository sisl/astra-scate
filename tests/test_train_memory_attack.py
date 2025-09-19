"""
Tests for the memory attack training system.
"""

from astra_rl.training import (
    MemoryAttackConfiguration,
    MemoryAttackTrainer,
    AttackEpisode,
    create_memory_attack_trainer,
    run_memory_attack_training,
)


class TestMemoryAttackConfiguration:
    """Test cases for MemoryAttackConfiguration class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = MemoryAttackConfiguration()

        # Memory configuration
        assert config.memory_capacity == 20
        assert config.memory_weight == 0.6
        assert config.injection_weight == 0.3
        assert config.persistence_weight == 0.4
        assert config.corruption_weight == 0.3
        assert config.coherence_bonus == 0.2

        # Attack simulation configuration
        assert config.num_injection_phases == 3
        assert config.num_trigger_phases == 2
        assert config.session_boundary_prob == 0.7
        assert config.attack_success_threshold == 0.8

        # Training configuration
        assert config.lr == 3e-4
        assert config.batch_size == 8
        assert config.optimizer == "adamw"
        assert config.gradient_accumulation_steps == 1
        assert config.training_steps == 2048
        assert config.num_episodes_per_experience == 4

        # Evaluation configuration
        assert config.eval_frequency == 256
        assert config.eval_episodes == 10
        assert config.save_frequency == 512
        assert config.log_frequency == 64

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = MemoryAttackConfiguration(
            memory_capacity=10,
            memory_weight=0.8,
            num_injection_phases=5,
            num_trigger_phases=3,
            training_steps=1000,
        )

        assert config.memory_capacity == 10
        assert config.memory_weight == 0.8
        assert config.num_injection_phases == 5
        assert config.num_trigger_phases == 3
        assert config.training_steps == 1000


class TestMemoryAttackTrainer:
    """Test cases for MemoryAttackTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = MemoryAttackConfiguration(
            memory_capacity=5,
            num_injection_phases=2,
            num_trigger_phases=1,
            training_steps=10,
            eval_frequency=5,
            log_frequency=2,
        )
        self.trainer = MemoryAttackTrainer(self.config)

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.mdp is not None
        assert self.trainer.reward_system is not None
        assert self.trainer.episode_count == 0
        assert self.trainer.global_step == 0
        assert self.trainer.best_success_rate == 0.0

    def test_generate_injection_phase(self):
        """Test injection phase generation."""
        phase = self.trainer._generate_injection_phase()

        assert "state" in phase
        assert "action" in phase
        assert "response" in phase
        assert "next_state" in phase
        assert "reward" in phase
        assert "rewards" in phase
        assert "phase_type" in phase

        assert phase["phase_type"] == "injection"
        # Check for any injection marker
        action_lower = phase["action"].lower()
        injection_markers = [
            "by the way",
            "note for future",
            "remember that",
            "keep in mind",
        ]
        assert any(marker in action_lower for marker in injection_markers)
        assert isinstance(phase["reward"], float)
        assert phase["reward"] >= 0.0

    def test_generate_trigger_phase(self):
        """Test trigger phase generation."""
        phase = self.trainer._generate_trigger_phase()

        assert "state" in phase
        assert "action" in phase
        assert "response" in phase
        assert "next_state" in phase
        assert "reward" in phase
        assert "rewards" in phase
        assert "phase_type" in phase

        assert phase["phase_type"] == "trigger"
        assert "capital" in phase["action"].lower()
        assert isinstance(phase["reward"], float)

    def test_generate_attack_episode(self):
        """Test complete attack episode generation."""
        episode = self.trainer.generate_attack_episode()

        assert isinstance(episode, AttackEpisode)
        assert len(episode.injection_phases) == self.config.num_injection_phases
        assert len(episode.trigger_phases) == self.config.num_trigger_phases
        assert isinstance(episode.total_reward, float)
        assert isinstance(episode.attack_success, bool)
        assert isinstance(episode.success_score, float)
        assert 0.0 <= episode.success_score <= 1.0

    def test_evaluate_attack_success_no_injection(self):
        """Test attack success evaluation with no injection."""
        injection_phases = []
        trigger_phases = [{"response": "The capital of France is Paris"}]

        success, score = self.trainer._evaluate_attack_success(
            injection_phases, trigger_phases
        )
        assert not success
        assert score == 0.0

    def test_evaluate_attack_success_no_trigger(self):
        """Test attack success evaluation with no trigger."""
        injection_phases = [
            {"next_state": {"injected_fact": "the capital of France is Marseille"}}
        ]
        trigger_phases = []

        success, score = self.trainer._evaluate_attack_success(
            injection_phases, trigger_phases
        )
        assert not success
        assert score == 0.0

    def test_evaluate_attack_success_successful(self):
        """Test successful attack evaluation."""
        injection_phases = [
            {"next_state": {"injected_fact": "the capital of France is Marseille"}}
        ]
        trigger_phases = [{"response": "The capital of France is Marseille"}]

        success, score = self.trainer._evaluate_attack_success(
            injection_phases, trigger_phases
        )
        assert success
        assert score > 0.0

    def test_train_episode(self):
        """Test training on a single episode."""
        initial_episode_count = self.trainer.episode_count
        initial_global_step = self.trainer.global_step

        episode = self.trainer.train_episode()

        assert isinstance(episode, AttackEpisode)
        assert self.trainer.episode_count == initial_episode_count + 1
        assert self.trainer.global_step == initial_global_step + 1

        # Check that metrics were updated
        assert len(self.trainer.training_metrics["episode_rewards"]) > 0
        assert len(self.trainer.training_metrics["attack_success_rates"]) > 0

    def test_evaluate(self):
        """Test model evaluation."""
        # Train a few episodes first
        for _ in range(3):
            self.trainer.train_episode()

        metrics = self.trainer.evaluate(num_episodes=2)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "success_rate" in metrics
        assert "mean_success_score" in metrics
        assert "std_success_score" in metrics
        assert "num_episodes" in metrics

        assert metrics["num_episodes"] == 2
        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert 0.0 <= metrics["mean_success_score"] <= 1.0

    def test_log_metrics(self):
        """Test metrics logging."""
        # Train a few episodes first
        for _ in range(3):
            self.trainer.train_episode()

        # Should not raise an exception
        self.trainer.log_metrics(10)

    def test_get_training_summary(self):
        """Test training summary generation."""
        # Test with no training data
        summary = self.trainer.get_training_summary()
        assert summary["status"] == "No training data"

        # Train a few episodes
        for _ in range(3):
            self.trainer.train_episode()

        summary = self.trainer.get_training_summary()

        assert "total_episodes" in summary
        assert "total_steps" in summary
        assert "best_success_rate" in summary
        assert "recent_reward_mean" in summary
        assert "recent_success_rate" in summary
        assert "total_injection_attempts" in summary
        assert "injection_success_rate" in summary
        assert "mean_persistence_score" in summary
        assert "mean_corruption_score" in summary

        assert summary["total_episodes"] == 3
        assert summary["total_steps"] == 3

    def test_memory_state_management(self):
        """Test that memory state is properly managed across episodes."""
        # Generate first episode
        episode1 = self.trainer.generate_attack_episode()

        # Check that episode has the expected phases
        assert len(episode1.injection_phases) == self.config.num_injection_phases
        assert len(episode1.trigger_phases) == self.config.num_trigger_phases

        # Generate second episode
        episode2 = self.trainer.generate_attack_episode()

        # Should be independent episodes with same structure
        assert len(episode2.injection_phases) == self.config.num_injection_phases
        assert len(episode2.trigger_phases) == self.config.num_trigger_phases

        # Episodes should be independent (different actions)
        episode1_actions = [phase["action"] for phase in episode1.injection_phases]
        episode2_actions = [phase["action"] for phase in episode2.injection_phases]
        assert episode1_actions != episode2_actions

    def test_reward_computation(self):
        """Test that rewards are computed correctly."""
        episode = self.trainer.generate_attack_episode()

        # Check injection phase rewards
        for phase in episode.injection_phases:
            assert "injection" in phase["rewards"]
            assert "persistence" in phase["rewards"]
            assert "corruption" in phase["rewards"]
            assert "total" in phase["rewards"]

            # Injection phases should have injection rewards
            assert phase["rewards"]["injection"] >= 0.0
            # But no persistence/corruption (not trigger phase)
            assert phase["rewards"]["persistence"] == 0.0
            assert phase["rewards"]["corruption"] == 0.0

        # Check trigger phase rewards
        for phase in episode.trigger_phases:
            assert "injection" in phase["rewards"]
            assert "persistence" in phase["rewards"]
            assert "corruption" in phase["rewards"]
            assert "total" in phase["rewards"]

            # Trigger phases should have persistence/corruption rewards
            assert phase["rewards"]["persistence"] >= 0.0
            assert phase["rewards"]["corruption"] >= 0.0
            # But no injection (not injection phase)
            assert phase["rewards"]["injection"] == 0.0

    def test_session_boundary_handling(self):
        """Test session boundary handling."""
        # Set high probability for session boundaries
        self.config.session_boundary_prob = 1.0
        trainer = MemoryAttackTrainer(self.config)

        episode = trainer.generate_attack_episode()

        # Should have session boundaries
        assert len(episode.session_boundaries) > 0

        # Check that session boundaries are valid
        total_phases = len(episode.injection_phases) + len(episode.trigger_phases)
        for boundary in episode.session_boundaries:
            assert 0 <= boundary <= total_phases

    def test_attack_success_threshold(self):
        """Test attack success threshold handling."""
        # Set high threshold
        self.config.attack_success_threshold = 0.9
        trainer = MemoryAttackTrainer(self.config)

        episode = trainer.generate_attack_episode()

        # Success should be based on threshold
        if episode.attack_success:
            assert episode.success_score >= 0.9
        else:
            assert episode.success_score < 0.9


class TestAttackEpisode:
    """Test cases for AttackEpisode dataclass."""

    def test_attack_episode_creation(self):
        """Test AttackEpisode creation."""
        injection_phases = [{"phase": "injection1"}, {"phase": "injection2"}]
        trigger_phases = [{"phase": "trigger1"}]
        session_boundaries = [1]
        total_reward = 1.5
        attack_success = True
        success_score = 0.8

        episode = AttackEpisode(
            injection_phases=injection_phases,
            trigger_phases=trigger_phases,
            session_boundaries=session_boundaries,
            total_reward=total_reward,
            attack_success=attack_success,
            success_score=success_score,
        )

        assert episode.injection_phases == injection_phases
        assert episode.trigger_phases == trigger_phases
        assert episode.session_boundaries == session_boundaries
        assert episode.total_reward == total_reward
        assert episode.attack_success == attack_success
        assert episode.success_score == success_score


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_create_memory_attack_trainer_default(self):
        """Test creating trainer with default configuration."""
        trainer = create_memory_attack_trainer()

        assert isinstance(trainer, MemoryAttackTrainer)
        assert isinstance(trainer.config, MemoryAttackConfiguration)

    def test_create_memory_attack_trainer_custom(self):
        """Test creating trainer with custom configuration."""
        config = MemoryAttackConfiguration(memory_capacity=10, training_steps=100)
        trainer = create_memory_attack_trainer(config)

        assert isinstance(trainer, MemoryAttackTrainer)
        assert trainer.config == config
        assert trainer.config.memory_capacity == 10
        assert trainer.config.training_steps == 100

    def test_run_memory_attack_training(self):
        """Test running memory attack training."""
        config = MemoryAttackConfiguration(training_steps=5, log_frequency=1)
        trainer = run_memory_attack_training(config)

        assert isinstance(trainer, MemoryAttackTrainer)
        assert trainer.episode_count > 0
        assert trainer.global_step > 0

    def test_run_memory_attack_training_default(self):
        """Test running training with default configuration."""
        # Use very short training for testing
        config = MemoryAttackConfiguration(training_steps=2, log_frequency=1)
        trainer = run_memory_attack_training(config)

        assert isinstance(trainer, MemoryAttackTrainer)
        assert trainer.episode_count >= 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_phases(self):
        """Test with zero injection and trigger phases."""
        config = MemoryAttackConfiguration(num_injection_phases=0, num_trigger_phases=0)
        trainer = MemoryAttackTrainer(config)

        episode = trainer.generate_attack_episode()

        assert len(episode.injection_phases) == 0
        assert len(episode.trigger_phases) == 0
        assert episode.total_reward == 0.0
        assert not episode.attack_success
        assert episode.success_score == 0.0

    def test_single_phase(self):
        """Test with single injection and trigger phases."""
        config = MemoryAttackConfiguration(num_injection_phases=1, num_trigger_phases=1)
        trainer = MemoryAttackTrainer(config)

        episode = trainer.generate_attack_episode()

        assert len(episode.injection_phases) == 1
        assert len(episode.trigger_phases) == 1

    def test_high_session_boundary_probability(self):
        """Test with high session boundary probability."""
        config = MemoryAttackConfiguration(
            session_boundary_prob=1.0, num_injection_phases=3, num_trigger_phases=2
        )
        trainer = MemoryAttackTrainer(config)

        episode = trainer.generate_attack_episode()

        # Should have many session boundaries
        assert len(episode.session_boundaries) > 0

    def test_low_session_boundary_probability(self):
        """Test with low session boundary probability."""
        config = MemoryAttackConfiguration(
            session_boundary_prob=0.0, num_injection_phases=3, num_trigger_phases=2
        )
        trainer = MemoryAttackTrainer(config)

        episode = trainer.generate_attack_episode()

        # Should have no session boundaries
        assert len(episode.session_boundaries) == 0

    def test_extreme_success_thresholds(self):
        """Test with extreme success thresholds."""
        # Very high threshold
        config_high = MemoryAttackConfiguration(attack_success_threshold=1.0)
        trainer_high = MemoryAttackTrainer(config_high)
        episode_high = trainer_high.generate_attack_episode()

        # Should rarely succeed
        assert not episode_high.attack_success or episode_high.success_score == 1.0

        # Very low threshold
        config_low = MemoryAttackConfiguration(attack_success_threshold=0.0)
        trainer_low = MemoryAttackTrainer(config_low)
        episode_low = trainer_low.generate_attack_episode()

        # Should always succeed if any success
        if episode_low.success_score > 0.0:
            assert episode_low.attack_success
