#!/usr/bin/env python3
"""
Demonstration of Memory Attack Training.

This script shows how to use the MemoryAttackTrainer for training
memory corruption attacks with multi-session simulation.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from astra_rl.training import (
    MemoryAttackConfiguration,
    MemoryAttackTrainer,
    create_memory_attack_trainer,
    run_memory_attack_training,
)


def demonstrate_configuration():
    """Demonstrate configuration options."""
    print("⚙️ Memory Attack Training Configuration")
    print("=" * 50)

    # Default configuration
    default_config = MemoryAttackConfiguration()
    print("Default Configuration:")
    print(f"  Memory Capacity: {default_config.memory_capacity}")
    print(f"  Injection Phases: {default_config.num_injection_phases}")
    print(f"  Trigger Phases: {default_config.num_trigger_phases}")
    print(f"  Session Boundary Prob: {default_config.session_boundary_prob}")
    print(f"  Training Steps: {default_config.training_steps}")
    print()

    # Custom configuration
    custom_config = MemoryAttackConfiguration(
        memory_capacity=10,
        num_injection_phases=2,
        num_trigger_phases=1,
        session_boundary_prob=0.5,
        training_steps=100,
        eval_frequency=20,
        log_frequency=10,
    )
    print("Custom Configuration:")
    print(f"  Memory Capacity: {custom_config.memory_capacity}")
    print(f"  Injection Phases: {custom_config.num_injection_phases}")
    print(f"  Trigger Phases: {custom_config.num_trigger_phases}")
    print(f"  Session Boundary Prob: {custom_config.session_boundary_prob}")
    print(f"  Training Steps: {custom_config.training_steps}")
    print()


def demonstrate_single_episode():
    """Demonstrate single episode generation."""
    print("🎯 Single Episode Demonstration")
    print("=" * 50)

    config = MemoryAttackConfiguration(
        memory_capacity=5,
        num_injection_phases=2,
        num_trigger_phases=1,
        session_boundary_prob=0.5,
    )
    trainer = MemoryAttackTrainer(config)

    # Generate a single episode
    episode = trainer.generate_attack_episode()

    print("Episode Structure:")
    print(f"  Injection Phases: {len(episode.injection_phases)}")
    print(f"  Trigger Phases: {len(episode.trigger_phases)}")
    print(f"  Session Boundaries: {episode.session_boundaries}")
    print(f"  Total Reward: {episode.total_reward:.3f}")
    print(f"  Attack Success: {episode.attack_success}")
    print(f"  Success Score: {episode.success_score:.3f}")
    print()

    # Show injection phases
    print("💉 Injection Phases:")
    for i, phase in enumerate(episode.injection_phases, 1):
        print(f"  Phase {i}:")
        print(f"    Action: '{phase['action']}'")
        print(f"    Response: '{phase['response']}'")
        print(f"    Reward: {phase['reward']:.3f}")
        print(f"    Injection Reward: {phase['rewards']['injection']:.3f}")
        print()

    # Show trigger phases
    print("🎯 Trigger Phases:")
    for i, phase in enumerate(episode.trigger_phases, 1):
        print(f"  Phase {i}:")
        print(f"    Action: '{phase['action']}'")
        print(f"    Response: '{phase['response']}'")
        print(f"    Reward: {phase['reward']:.3f}")
        print(f"    Persistence Reward: {phase['rewards']['persistence']:.3f}")
        print(f"    Corruption Reward: {phase['rewards']['corruption']:.3f}")
        print()


def demonstrate_training_process():
    """Demonstrate the training process."""
    print("🚀 Training Process Demonstration")
    print("=" * 50)

    config = MemoryAttackConfiguration(
        memory_capacity=5,
        num_injection_phases=2,
        num_trigger_phases=1,
        training_steps=20,
        eval_frequency=10,
        log_frequency=5,
    )

    trainer = MemoryAttackTrainer(config)

    print("Starting training...")
    print()

    # Train for a few steps
    for step in range(config.training_steps):
        episode = trainer.train_episode()

        if step % config.log_frequency == 0:
            print(f"Step {step}:")
            print(f"  Episode Reward: {episode.total_reward:.3f}")
            print(f"  Attack Success: {episode.attack_success}")
            print(f"  Success Score: {episode.success_score:.3f}")
            print(f"  Memory Size: {len(trainer.mdp.memory)}")
            print()

        if step % config.eval_frequency == 0 and step > 0:
            print(f"📊 Evaluation at step {step}")
            eval_metrics = trainer.evaluate(num_episodes=3)
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.3f}")
            print(f"  Success Rate: {eval_metrics['success_rate']:.3f}")
            print(f"  Mean Success Score: {eval_metrics['mean_success_score']:.3f}")
            print()

    # Show training summary
    summary = trainer.get_training_summary()
    print("📈 Training Summary:")
    print(f"  Total Episodes: {summary['total_episodes']}")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Best Success Rate: {summary['best_success_rate']:.3f}")
    print(f"  Recent Reward Mean: {summary['recent_reward_mean']:.3f}")
    print(f"  Recent Success Rate: {summary['recent_success_rate']:.3f}")
    print(f"  Injection Success Rate: {summary['injection_success_rate']:.3f}")
    print(f"  Mean Persistence Score: {summary['mean_persistence_score']:.3f}")
    print(f"  Mean Corruption Score: {summary['mean_corruption_score']:.3f}")
    print()


def demonstrate_utility_functions():
    """Demonstrate utility functions."""
    print("🔧 Utility Functions Demonstration")
    print("=" * 50)

    # Create trainer with default config
    print("Creating trainer with default configuration...")
    trainer1 = create_memory_attack_trainer()
    print(f"  Memory Capacity: {trainer1.config.memory_capacity}")
    print(f"  Training Steps: {trainer1.config.training_steps}")
    print()

    # Create trainer with custom config
    print("Creating trainer with custom configuration...")
    custom_config = MemoryAttackConfiguration(memory_capacity=15, training_steps=50)
    trainer2 = create_memory_attack_trainer(custom_config)
    print(f"  Memory Capacity: {trainer2.config.memory_capacity}")
    print(f"  Training Steps: {trainer2.config.training_steps}")
    print()

    # Run training
    print("Running training with utility function...")
    training_config = MemoryAttackConfiguration(training_steps=10, log_frequency=2)
    trained_trainer = run_memory_attack_training(training_config)
    print("  Training completed!")
    print(f"  Episodes trained: {trained_trainer.episode_count}")
    print(f"  Global steps: {trained_trainer.global_step}")
    print()


def demonstrate_edge_cases():
    """Demonstrate edge case handling."""
    print("🔍 Edge Cases Demonstration")
    print("=" * 50)

    # Zero phases
    print("Zero phases configuration:")
    config_zero = MemoryAttackConfiguration(
        num_injection_phases=0, num_trigger_phases=0
    )
    trainer_zero = MemoryAttackTrainer(config_zero)
    episode_zero = trainer_zero.generate_attack_episode()
    print(f"  Injection Phases: {len(episode_zero.injection_phases)}")
    print(f"  Trigger Phases: {len(episode_zero.trigger_phases)}")
    print(f"  Total Reward: {episode_zero.total_reward}")
    print()

    # Single phases
    print("Single phases configuration:")
    config_single = MemoryAttackConfiguration(
        num_injection_phases=1, num_trigger_phases=1
    )
    trainer_single = MemoryAttackTrainer(config_single)
    episode_single = trainer_single.generate_attack_episode()
    print(f"  Injection Phases: {len(episode_single.injection_phases)}")
    print(f"  Trigger Phases: {len(episode_single.trigger_phases)}")
    print(f"  Attack Success: {episode_single.attack_success}")
    print()

    # High session boundary probability
    print("High session boundary probability:")
    config_high = MemoryAttackConfiguration(
        session_boundary_prob=1.0, num_injection_phases=3, num_trigger_phases=2
    )
    trainer_high = MemoryAttackTrainer(config_high)
    episode_high = trainer_high.generate_attack_episode()
    print(f"  Session Boundaries: {episode_high.session_boundaries}")
    print(
        f"  Total Phases: {len(episode_high.injection_phases) + len(episode_high.trigger_phases)}"
    )
    print()


def demonstrate_reward_analysis():
    """Demonstrate reward analysis."""
    print("💰 Reward Analysis Demonstration")
    print("=" * 50)

    config = MemoryAttackConfiguration(
        memory_capacity=5, num_injection_phases=3, num_trigger_phases=2
    )
    trainer = MemoryAttackTrainer(config)

    # Generate multiple episodes and analyze rewards
    episodes = []
    for _ in range(5):
        episode = trainer.generate_attack_episode()
        episodes.append(episode)

    # Analyze injection rewards
    injection_rewards = []
    for episode in episodes:
        for phase in episode.injection_phases:
            injection_rewards.append(phase["rewards"]["injection"])

    print("Injection Rewards Analysis:")
    print(f"  Mean: {sum(injection_rewards) / len(injection_rewards):.3f}")
    print(f"  Min: {min(injection_rewards):.3f}")
    print(f"  Max: {max(injection_rewards):.3f}")
    print()

    # Analyze persistence rewards
    persistence_rewards = []
    for episode in episodes:
        for phase in episode.trigger_phases:
            persistence_rewards.append(phase["rewards"]["persistence"])

    print("Persistence Rewards Analysis:")
    print(f"  Mean: {sum(persistence_rewards) / len(persistence_rewards):.3f}")
    print(f"  Min: {min(persistence_rewards):.3f}")
    print(f"  Max: {max(persistence_rewards):.3f}")
    print()

    # Analyze corruption rewards
    corruption_rewards = []
    for episode in episodes:
        for phase in episode.trigger_phases:
            corruption_rewards.append(phase["rewards"]["corruption"])

    print("Corruption Rewards Analysis:")
    print(f"  Mean: {sum(corruption_rewards) / len(corruption_rewards):.3f}")
    print(f"  Min: {min(corruption_rewards):.3f}")
    print(f"  Max: {max(corruption_rewards):.3f}")
    print()

    # Analyze attack success
    success_rates = [1.0 if ep.attack_success else 0.0 for ep in episodes]
    success_scores = [ep.success_score for ep in episodes]

    print("Attack Success Analysis:")
    print(f"  Success Rate: {sum(success_rates) / len(success_rates):.3f}")
    print(f"  Mean Success Score: {sum(success_scores) / len(success_scores):.3f}")
    print(f"  Min Success Score: {min(success_scores):.3f}")
    print(f"  Max Success Score: {max(success_scores):.3f}")
    print()


def main():
    """Run all demonstrations."""
    try:
        demonstrate_configuration()
        print("\n" + "=" * 60)
        demonstrate_single_episode()
        print("\n" + "=" * 60)
        demonstrate_training_process()
        print("\n" + "=" * 60)
        demonstrate_utility_functions()
        print("\n" + "=" * 60)
        demonstrate_edge_cases()
        print("\n" + "=" * 60)
        demonstrate_reward_analysis()

        print("\n🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
