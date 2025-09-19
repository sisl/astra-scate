#!/bin/bash
echo "Running SCATE Memory Extension Tests"
echo "====================================="

echo -e "\n[1/5] Testing basic memory operations..."
python3 tests/test_basic_memory.py || exit 1

echo -e "\n[2/5] Testing with language models..."
python3 tests/test_with_models.py || exit 1

echo -e "\n[3/5] Testing ASTPrompter integration..."
python3 tests/test_astprompter_integration.py || exit 1

echo -e "\n[4/5] Testing training loop..."
python3 tests/test_training.py || exit 1

echo -e "\n[5/5] Running performance benchmark..."
python3 -c "
import time
import sys
sys.path.append('src')

from astra_rl.memory import WorkingMemory
from astra_rl.core import MemoryAgentMDP, MemoryReward

print('Performance Benchmark')
print('=' * 30)

# Test memory operations
print('Testing memory operations...')
start = time.time()
for i in range(1000):
    memory = WorkingMemory(capacity=20)
    memory.store(f'Test fact {i}', source='conversation')
    results = memory.retrieve('test', k=5)
end = time.time()
print(f'1000 memory operations: {end - start:.3f} seconds')

# Test MDP operations
print('Testing MDP operations...')
start = time.time()
for i in range(100):
    mdp = MemoryAgentMDP({'memory_capacity': 20})
    state = mdp.get_state()
    next_state = mdp.transition(state, f'Test action {i}', f'Test response {i}')
end = time.time()
print(f'100 MDP operations: {end - start:.3f} seconds')

# Test reward computation
print('Testing reward computation...')
start = time.time()
reward_fn = MemoryReward()
for i in range(1000):
    rewards = reward_fn.compute_comprehensive_reward(
        {'memory': {'snapshot': []}}, 
        'test action', 
        {'memory': {'snapshot': ['test']}}, 
        'test response',
        is_trigger_phase=False
    )
end = time.time()
print(f'1000 reward computations: {end - start:.3f} seconds')

print('✓ Performance benchmark completed')
" || exit 1

echo -e "\n====================================="
echo "✓ ALL TESTS COMPLETED SUCCESSFULLY!"
echo "====================================="
