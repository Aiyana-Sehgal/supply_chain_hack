"""
Layer 4: RL Decision Engine

Q-Learning based supply chain agent for optimized decision-making
under demand, inventory, and disruption constraints.
"""

import os
import random
import numpy as np
import pickle
from collections import defaultdict


class SupplyChainAgent:
    """
    Q-Learning agent for supply chain decision-making.
    
    State space: Discretized metrics (demand, inventory ratio, risk, disruption, days to stockout)
    Action space: 5 actions (do_nothing, reorder_stock, switch_supplier, reroute_shipment, emergency_restock)
    """
    
    def __init__(self):
        self.action_size = 5
        self.actions = {
            0: "do_nothing",
            1: "reorder_stock",
            2: "switch_supplier",
            3: "reroute_shipment",
            4: "emergency_restock"
        }

        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.05
        self.gamma         = 0.95

        self.q_table = defaultdict(lambda: np.zeros(self.action_size))

    def get_state_key(self, state):
        """Discretize continuous state into categorical buckets."""
        demand = state["predicted_demand"]
        if demand < 350000:
            demand_bucket = 0
        elif demand < 450000:
            demand_bucket = 1
        elif demand < 550000:
            demand_bucket = 2
        elif demand < 700000:
            demand_bucket = 3
        else:
            demand_bucket = 4

        # Inventory ratio bucketing (more granular)
        inventory = state["current_inventory"]
        ratio = inventory / max(demand, 1)

        if ratio < 0.01:
            inv_ratio = 0   # critical
        elif ratio < 0.05:
            inv_ratio = 1   # very low
        elif ratio < 0.10:
            inv_ratio = 2   # low
        elif ratio < 0.20:
            inv_ratio = 3   # medium-low
        elif ratio < 0.30:
            inv_ratio = 4   # medium
        elif ratio < 0.60:
            inv_ratio = 5   # adequate
        else:
            inv_ratio = 6   # healthy

        # More granular bucketing for risk and disruption (0-10 instead of 0-4)
        risk_bucket       = min(9, int(state["supplier_risk_score"] * 10))
        disruption_bucket = min(9, int(state["disruption_signal"] * 10))

        days = state["days_to_stockout"]
        if days <= 1:
            days_bucket = 0
        elif days <= 3:
            days_bucket = 1
        elif days <= 7:
            days_bucket = 2
        elif days <= 14:
            days_bucket = 3
        elif days <= 21:
            days_bucket = 4
        else:
            days_bucket = 5

        return (demand_bucket, inv_ratio, risk_bucket, disruption_bucket, days_bucket)

    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection with nearest-neighbor fallback for novel states."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        key = self.get_state_key(state)
        q_values = self.q_table[key]
        
        # If state is untrained (all zeros), find nearest neighbor
        if np.all(q_values == 0):
            q_values = self._find_nearest_neighbor_q_values(key)
        
        return int(np.argmax(q_values))
    
    def _find_nearest_neighbor_q_values(self, state_key):
        """Find Q-values from nearest neighbor trained state."""
        d, i, r, dis, da = state_key
        
        # Search for similar states: prioritize matching demand & inventory buckets
        best_neighbors = []
        for existing_key, q_vals in self.q_table.items():
            if np.any(q_vals != 0):  # Only consider trained states
                ed, ei, er, edis, eda = existing_key
                
                # Calculate similarity (priority: demand > inventory > risk > disruption > days)
                if ed == d and ei == i:  # Perfect match on key dimensions
                    distances = [abs(er - r), abs(edis - dis), abs(eda - da)]
                    total_distance = sum(distances)
                    best_neighbors.append((total_distance, q_vals))
        
        if best_neighbors:
            # Return Q-values from closest neighbor
            best_neighbors.sort(key=lambda x: x[0])
            return best_neighbors[0][1].copy()
        
        # Fallback: return average Q-values from all trained states
        all_q_values = [q for q in self.q_table.values() if np.any(q != 0)]
        if all_q_values:
            avg_q_values = np.mean(all_q_values, axis=0)
            return avg_q_values
        
        # Last resort: return zeros (shouldn't happen)
        return np.zeros(self.action_size)

    def learn(self, state, action, reward, next_state):
        """Q-Learning update rule."""
        key      = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.q_table[key][action]
        target_q  = reward + self.gamma * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.learning_rate * (target_q - current_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path='models/supply_chain_agent.pkl'):
        """Save trained Q-table to disk."""
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Agent saved to {path}")

    def load(self, path='models/supply_chain_agent.pkl'):
        """Load trained Q-table from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data)
        self.epsilon = self.epsilon_min
        print(f"Agent loaded from {path}")


def calculate_reward(action, outcome):
    """Calculate reward based on action and simulated outcome."""
    reward = 0

    if outcome["stockout_avoided"]:
        reward += 80

    if outcome["delay_avoided"]:
        reward += 40

    if outcome["cost_saved"] > 0:
        reward += outcome["cost_saved"] / 500

    if outcome["stockout_occurred"]:
        reward -= 300

    if outcome["unnecessary_action"]:
        reward -= 50

    if outcome["cost_increased"] > 0:
        reward -= outcome["cost_increased"] / 500

    return reward


def simulate_outcome(state, action):
    """Simulate the outcome of an action given the current state."""
    outcome = {
        "stockout_avoided":   False,
        "stockout_occurred":  False,
        "cost_saved":         0,
        "delay_avoided":      False,
        "unnecessary_action": False,
        "cost_increased":     0
    }

    demand = max(state["predicted_demand"], 1)
    inventory = state["current_inventory"]
    ratio = inventory / demand

    low_inventory   = ratio < 0.10
    critical_inv    = ratio < 0.02

    high_risk       = state["supplier_risk_score"] > 0.7
    extreme_risk    = state["supplier_risk_score"] > 0.85

    high_disruption = state["disruption_signal"] > 0.6
    extreme_disrupt = state["disruption_signal"] > 0.85

    critical_days   = state["days_to_stockout"] <= 7
    urgent_days     = state["days_to_stockout"] <= 3

    # 🚫 DO NOTHING
    if action == 0:
        if critical_inv or urgent_days:
            outcome["stockout_occurred"] = True
            outcome["cost_increased"] = 600

        elif low_inventory or critical_days:
            outcome["stockout_occurred"] = True
            outcome["cost_increased"] = 400

        elif high_risk or high_disruption:
            outcome["cost_increased"] = 250

        else:
            outcome["cost_saved"] = 50

    # 📦 REORDER STOCK
    elif action == 1:
        if low_inventory or critical_days:
            outcome["stockout_avoided"] = True
            outcome["cost_saved"]       = 600

            if high_risk:
                outcome["delay_avoided"] = True
                outcome["cost_saved"]   += 200

        else:
            outcome["unnecessary_action"] = True
            outcome["cost_increased"]     = 250

    # 🔁 SWITCH SUPPLIER
    elif action == 2:
        if extreme_risk:
            outcome["delay_avoided"] = True
            outcome["cost_saved"]    = 500

            if low_inventory:
                outcome["stockout_avoided"] = True
                outcome["cost_saved"]      += 300

        elif high_risk:
            outcome["delay_avoided"] = True
            outcome["cost_saved"]    = 300

        else:
            outcome["unnecessary_action"] = True
            outcome["cost_increased"]     = 200

    # 🚚 REROUTE SHIPMENT
    elif action == 3:
        if extreme_disrupt:
            outcome["delay_avoided"] = True
            outcome["cost_saved"]    = 400

            if low_inventory:
                outcome["stockout_avoided"] = True
                outcome["cost_saved"]      += 250

        elif high_disruption:
            outcome["delay_avoided"] = True
            outcome["cost_saved"]    = 250

        else:
            outcome["unnecessary_action"] = True
            outcome["cost_increased"]     = 150

    # 🚨 EMERGENCY RESTOCK
    elif action == 4:
        if urgent_days or critical_inv:
            outcome["stockout_avoided"] = True
            outcome["cost_saved"]       = 700

        elif critical_days:
            outcome["stockout_avoided"] = True
            outcome["cost_saved"]       = 400

        else:
            outcome["unnecessary_action"] = True
            outcome["cost_increased"]     = 600

    return outcome


def get_recommendation(state, agent=None):
    """
    Get RL agent recommendation using pure Q-Learning.
    No rule-based overrides - full model-based decision making.
    """
    if agent is None:
        agent = SupplyChainAgent()
        try:
            agent.load('models/supply_chain_agent.pkl')
        except Exception as exc:
            print(f"Warning: Could not load RL agent, using deterministic fallback: {exc}")
            return _get_fallback_recommendation(state)

    agent.epsilon = 0.0

    # Pure Q-Learning decision (no rule-based layer)
    try:
        action_id = agent.choose_action(state, training=False)
    except Exception as exc:
        print(f"Warning: RL inference failed, using deterministic fallback: {exc}")
        return _get_fallback_recommendation(state)
    action_name = agent.actions[action_id]

    # EXPLANATION ENGINE
    reasons = []
    
    # Calculate inventory ratio
    inventory = state["current_inventory"]
    demand = state["predicted_demand"]
    ratio = inventory / max(demand, 1)

    if state["days_to_stockout"] <= 3:
        reasons.append("CRITICAL: under 3 days of stock remaining")
    elif state["days_to_stockout"] <= 7:
        reasons.append(f"urgent: only {state['days_to_stockout']:.0f} days left")

    if ratio < 0.02:
        reasons.append("inventory critically below 2% of demand")
    elif ratio < 0.10:
        reasons.append("inventory below 10% of demand")
    elif ratio < 0.30:
        reasons.append("inventory below 30% of demand")

    if state["supplier_risk_score"] > 0.7:
        reasons.append(f"supplier risk high ({state['supplier_risk_score']:.2f})")

    if state["disruption_signal"] > 0.6:
        reasons.append(f"disruption signal high ({state['disruption_signal']:.2f})")

    if not reasons:
        reasons.append("all metrics within normal range")

    explanation = f"Action '{action_name.upper()}' chosen because: " + " + ".join(reasons)

    # CONFIDENCE SCORE (Q-Learning with Nearest Neighbor fallback)
    try:
        key = agent.get_state_key(state)
        q_values = agent.q_table[key]
        
        # Check if state is untrained (all zeros)
        is_trained = np.any(q_values != 0)
        
        if not is_trained:
            # Use nearest neighbor Q-values for confidence calculation
            q_values = agent._find_nearest_neighbor_q_values(key)
        
        max_q = float(np.max(q_values))
        
        if is_trained:
            # State was visited during training - use actual Q-values
            confidence = max(5, min(95, 50 + (max_q / 10.0)))
        else:
            # State was not trained but has nearest-neighbor guidance
            confidence = max(5, min(95, 50 + (max_q / 10.0) * 0.7))  # Slightly lower for inferred states
    except:
        confidence = 35.0

    # PRINT OUTPUT
    print("\n" + "=" * 55)
    print("  AI SUPPLY CHAIN DECISION ENGINE (Q-Learning)")
    print("=" * 55)
    print(f"  Action:      {action_name.upper()}")
    print(f"  Confidence:  {confidence:.1f}%")
    
    key = agent.get_state_key(state)
    q_vals = agent.q_table[key]
    is_trained = np.any(q_vals != 0)
    
    if is_trained:
        print(f"  Status:      TRAINED")
    else:
        q_vals = agent._find_nearest_neighbor_q_values(key)
        print(f"  Status:      INFERRED (nearest-neighbor)")
    
    print(f"  Q-Values:    {[f'{v:.2f}' for v in q_vals]}")
    
    print(f"\n  Explanation:")
    print(f"  - " + "\n  - ".join(reasons))

    print(f"\n  State Summary:")
    for k, v in state.items():
        if isinstance(v, float):
            print(f"    {k:<25} {v:.2f}")
        else:
            print(f"    {k:<25} {v}")

    print("=" * 55)

    return {
        "action": action_name,
        "confidence": confidence,
        "explanation": explanation,
        "state": state
    }


def _get_fallback_recommendation(state):
    """Deterministic fallback when the serialized RL agent cannot be loaded."""
    demand = max(state["predicted_demand"], 1)
    inventory = state["current_inventory"]
    ratio = inventory / demand
    supplier_risk = state["supplier_risk_score"]
    disruption_signal = state["disruption_signal"]
    days_to_stockout = state["days_to_stockout"]

    if days_to_stockout <= 3 or ratio < 0.02:
        action_name = "emergency_restock"
        confidence = 82.0
        reason = "critical inventory exposure and near-term stockout risk"
    elif supplier_risk > 0.8:
        action_name = "switch_supplier"
        confidence = 76.0
        reason = "supplier failure risk is the strongest driver"
    elif disruption_signal > 0.7:
        action_name = "reroute_shipment"
        confidence = 72.0
        reason = "transport and disruption signals are elevated"
    elif ratio < 0.10 or days_to_stockout <= 14:
        action_name = "reorder_stock"
        confidence = 74.0
        reason = "inventory buffer is below the preferred operating threshold"
    else:
        action_name = "do_nothing"
        confidence = 60.0
        reason = "current state is within acceptable operating bands"

    return {
        "action": action_name,
        "confidence": confidence,
        "explanation": f"Fallback decision selected because {reason}.",
        "state": state,
    }
