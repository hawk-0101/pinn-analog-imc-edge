"""
Greedy physics-sensitivity-guided bit-width allocation.

Algorithm:
  1. Sort layers by sensitivity score (descending).
  2. Assign highest feasible bit-width within energy budget.
  3. Energy model: cost(b) = b/8  (linear proxy).
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def allocate_bits(
    sensitivity: Dict[str, float],
    bit_options: List[int] = [4, 8],
    energy_budget: Optional[float] = None,
    base_energy: float = 1.0,
) -> Tuple[Dict[str, int], float]:
    """
    Returns (allocation dict layer→bits, actual energy used).
    If energy_budget is None, uses budget that would give all layers 8-bit.
    """
    n = len(sensitivity)
    if energy_budget is None:
        energy_budget = n * base_energy  # all at max bits

    sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    allocation: Dict[str, int] = {}
    remaining = energy_budget

    for layer_name, _ in sorted_layers:
        assigned = bit_options[0]
        for b in sorted(bit_options, reverse=True):
            cost = (b / 8.0) * base_energy
            if cost <= remaining + 1e-9:
                assigned = b
                break
        allocation[layer_name] = assigned
        remaining -= (assigned / 8.0) * base_energy

    actual_energy = energy_budget - remaining
    return allocation, actual_energy
