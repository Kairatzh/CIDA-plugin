"""
Latent MCTS (Monte Carlo Tree Search) — Unbounded Trajectory Exploration in Hidden Space.

Instead of building a text-level Tree of Thoughts (generating thousands of tokens),
LatentMCTS explores hypothesis trajectories directly in the vector space (hidden states h).

UCB Formula:
    UCB_i = Q_i + c * sqrt(ln(N) / n_i)
"""

import math
import torch
from typing import List, Dict, Any, Optional, Callable


class MCTSNode:
    """
    Node in the Latent MCTS tree representing a hidden state hypothesis trajectory.
    """
    def __init__(self, h: torch.Tensor, parent: Optional['MCTSNode'] = None, metadata: Optional[Dict[str, Any]] = None):
        self.h = h.detach().clone()  # (1, d_hidden) or (1, num_agents, d_hidden)
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.score: float = 0.0
        self.is_terminal: bool = False
        self.metadata: Dict[str, Any] = metadata or {}

    @property
    def value(self) -> float:
        """Average value Q = value_sum / visit_count."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_child(self, child_node: 'MCTSNode'):
        child_node.parent = self
        self.children.append(child_node)

    def ucb(self, exploration_constant: float = 1.414) -> float:
        """
        Upper Confidence Bound (UCB1) score.
        """
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(math.log(max(1, parent_visits)) / self.visit_count)
        return exploitation + exploration


class LatentMCTS:
    """
    Monte Carlo Tree Search in Latent Space.

    Explores trajectory branches from a base hidden state `h_base`, evaluates
    hypotheses via partial CoT sandbox or reward evaluators, and computes
    directional steering vectors from rejected branches.

    Parameters
    ----------
    exploration_constant : float
        c_puct / UCB exploration parameter (default: 1.414).
    steering_learning_rate : float
        Multiplier for steering offset computation (default: 0.15).
    """

    def __init__(self, exploration_constant: float = 1.414, steering_learning_rate: float = 0.15):
        self.c_puct = exploration_constant
        self.lr_steering = steering_learning_rate
        self.root: Optional[MCTSNode] = None

    def initialize_tree(self, h_base: torch.Tensor) -> MCTSNode:
        """Reset tree with root at base hidden state."""
        self.root = MCTSNode(h=h_base)
        self.root.visit_count = 1
        return self.root

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        if not node.children:
            return node
        best_score = -float('inf')
        best_child = node.children[0]
        for child in node.children:
            val = child.ucb(self.c_puct)
            if val > best_score:
                best_score = val
                best_child = child
        return best_child

    def explore_branches(
        self,
        node: MCTSNode,
        num_branches: int = 3,
        noise_scale: float = 0.05,
        steering_offset: Optional[torch.Tensor] = None
    ) -> List[MCTSNode]:
        """
        Generate candidate child nodes by perturbing the hidden state,
        optionally applying a directional steering offset away from failed hypotheses.
        """
        new_children = []
        for b_idx in range(num_branches):
            h_branch = node.h.clone()

            # Apply steering offset if available
            if steering_offset is not None:
                offset_cast = steering_offset.to(device=h_branch.device, dtype=h_branch.dtype)
                if offset_cast.shape == h_branch.shape:
                    h_branch = h_branch + offset_cast
                else:
                    # Broadened broadcast if shapes differ slightly
                    h_branch = h_branch + offset_cast.reshape_as(h_branch)

            # Add exploration noise for trajectory divergence (skip on first branch for clean baseline)
            if b_idx > 0:
                noise = torch.randn_like(h_branch) * noise_scale
                h_branch = h_branch + noise

            child = MCTSNode(h=h_branch, parent=node, metadata={"branch_idx": b_idx})
            node.add_child(child)
            new_children.append(child)

        return new_children

    def backpropagate(self, node: MCTSNode, score: float):
        """Update visit counts and value sums back to the root."""
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += score
            curr.score = score
            curr = curr.parent

    def compute_steering_vector(self, h_base: torch.Tensor, failed_nodes: List[MCTSNode]) -> Optional[torch.Tensor]:
        """
        Compute Directional Vector Steering Offset (Micro-Learning).

        Vector offset = lr * mean(h_base - h_failed)
        Moves search trajectory away from rejected hypothesis clusters.
        """
        if not failed_nodes:
            return None

        diffs = []
        for fn in failed_nodes:
            diff = (h_base.float() - fn.h.float())
            diffs.append(diff)

        avg_diff = torch.stack(diffs).mean(dim=0)
        steering_offset = self.lr_steering * avg_diff
        return steering_offset.to(dtype=h_base.dtype, device=h_base.device)

    def select_optimal_node(self) -> MCTSNode:
        """Select the node with the highest average value / score across the tree."""
        if self.root is None:
            raise ValueError("MCTS Tree not initialized.")

        best_node = self.root
        best_val = -float('inf')

        nodes_to_visit = [self.root]
        while nodes_to_visit:
            curr = nodes_to_visit.pop(0)
            if curr.visit_count > 0 and curr.value > best_val:
                best_val = curr.value
                best_node = curr
            nodes_to_visit.extend(curr.children)

        return best_node
