"""
Rule-based XAI helpers for supply chain recommendations and risk scores.
"""

from typing import Any, Dict


class XAIExplainer:
    """Rule-based explainer aligned with the project's Layer 5b design."""

    def explain_risk_score(self, state: Dict[str, Any]) -> Dict[str, str]:
        supplier_risk = float(state.get("supplier_risk_score", 0.0))
        disruption_signal = float(state.get("disruption_signal", 0.0))
        days_to_stockout = float(state.get("days_to_stockout", 0.0))
        predicted_demand = float(state.get("predicted_demand", 0.0))
        current_inventory = float(state.get("current_inventory", 0.0))
        composite_risk = float(state.get("composite_risk_score", 0.0))

        inventory_ratio = current_inventory / max(predicted_demand, 1.0)

        explanations = {
            "supplier_risk": self._explain_supplier_risk(supplier_risk),
            "disruption_risk": self._explain_disruption_risk(disruption_signal),
            "inventory_risk": self._explain_inventory_risk(days_to_stockout, inventory_ratio),
            "composite_risk": self._explain_composite_risk(
                composite_risk,
                supplier_risk,
                disruption_signal,
                days_to_stockout,
            ),
        }
        return explanations

    def explain_recommendation(self, state: Dict[str, Any], action: str, confidence: float) -> Dict[str, Any]:
        disruption_signal = float(state.get("disruption_signal", 0.0))
        days_to_stockout = float(state.get("days_to_stockout", 0.0))
        predicted_demand = float(state.get("predicted_demand", 0.0))
        current_inventory = float(state.get("current_inventory", 0.0))
        inventory_ratio = current_inventory / max(predicted_demand, 1.0)

        stockout_risk = self._stockout_risk_level(days_to_stockout)
        delay_risk = self._delay_risk_level(disruption_signal)

        action_sentences = {
            "emergency_restock": (
                "Critical inventory shortage detected. Immediate restocking required to prevent stockout."
            ),
            "reroute_shipment": (
                "High delay risk detected. Rerouting is recommended to maintain delivery timelines."
            ),
            "reorder_stock": (
                "Moderate stockout risk detected. Reordering stock will rebuild the inventory buffer."
            ),
            "switch_supplier": (
                "Supplier-side disruption is elevated. Switching suppliers will reduce execution risk."
            ),
            "do_nothing": (
                "System conditions are stable. No immediate operational action is required."
            ),
        }

        condition_clauses = []
        if stockout_risk == "high":
            condition_clauses.append(
                f"inventory coverage is short at {days_to_stockout:.1f} days"
            )
        elif stockout_risk == "medium":
            condition_clauses.append(
                f"inventory coverage is moderate at {days_to_stockout:.1f} days"
            )
        else:
            condition_clauses.append("inventory coverage remains healthy")

        if delay_risk == "high":
            condition_clauses.append("disruption signals are elevated")
        elif delay_risk == "medium":
            condition_clauses.append("there is moderate disruption exposure")
        else:
            condition_clauses.append("delay risk is low")

        rationale = (
            f"{action_sentences.get(action, 'Recommended action selected based on current supply chain risk profile.')} "
            f"This is driven by {condition_clauses[0]} and {condition_clauses[1]}."
        )

        return {
            "action": action,
            "confidence": confidence,
            "rationale": rationale,
            "primary_drivers": condition_clauses,
        }

    def explain_scenario_impact(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        impact = scenario_results.get("impact_analysis", {})
        risk_breakdown = scenario_results.get("risk_breakdown", {})
        recommendations = scenario_results.get("recommendations", {})

        cost_impact = float(impact.get("cost_impact", 0.0))
        demand_change = float(impact.get("demand_change", 0.0))
        stockout_probability = float(impact.get("stockout_probability", 0.0))
        delay_probability = float(impact.get("delay_probability", 0.0))
        days_to_stockout = float(risk_breakdown.get("days_to_stockout", 0.0))
        disruption_signal = float(risk_breakdown.get("d_s", risk_breakdown.get("disruption_risk", 0.0)))

        stockout_level = self._stockout_risk_level(days_to_stockout)
        delay_level = self._delay_risk_level(disruption_signal)

        cost_description = (
            "Cost remains stable due to reduced demand and low delay risk."
            if demand_change <= 0 and delay_level == "low"
            else "Cost increases due to higher demand and elevated delay risk."
        )

        summary = (
            f"Scenario impact shows {stockout_level} stockout risk and {delay_level} delay risk. "
            f"Estimated cost impact is {cost_impact:,.0f}. {cost_description} "
            f"Recommended response is {recommendations.get('action', 'do_nothing').replace('_', ' ')}."
        )

        key_factors = [
            f"stockout risk: {stockout_level}",
            f"delay risk: {delay_level}",
        ]
        if demand_change > 0:
            key_factors.append("demand is elevated")
        if cost_impact > 0:
            key_factors.append("cost pressure is rising")

        return {
            "scenario_summary": summary,
            "key_factors": key_factors,
        }

    def _stockout_risk_level(self, days_to_stockout: float) -> str:
        if days_to_stockout > 30:
            return "low"
        if days_to_stockout >= 7:
            return "medium"
        return "high"

    def _delay_risk_level(self, disruption_signal: float) -> str:
        if disruption_signal >= 0.8:
            return "high"
        if disruption_signal >= 0.4:
            return "medium"
        return "low"

    def _explain_supplier_risk(self, supplier_risk: float) -> str:
        if supplier_risk >= 0.8:
            return "High supplier risk due to weak supply-side resilience and elevated failure likelihood."
        if supplier_risk >= 0.5:
            return "Moderate supplier risk driven by unstable fulfillment conditions."
        return "Supplier risk is currently low and does not appear to be the main pressure point."

    def _explain_disruption_risk(self, disruption_signal: float) -> str:
        if disruption_signal >= 0.8:
            return "High delay risk due to strong disruption signals from supplier or logistics issues."
        if disruption_signal >= 0.4:
            return "Moderate delay risk due to one or more disruption factors."
        return "Low delay risk due to stable supplier and logistics conditions."

    def _explain_inventory_risk(self, days_to_stockout: float, inventory_ratio: float) -> str:
        if days_to_stockout < 5:
            return "High stockout risk because inventory coverage is short and demand is consuming stock quickly."
        if days_to_stockout <= 15:
            return "Moderate stockout risk from a smaller inventory buffer relative to demand."
        return "Low stockout risk because inventory coverage is sufficient for current demand."

    def _explain_composite_risk(
        self,
        composite_risk: float,
        supplier_risk: float,
        disruption_signal: float,
        days_to_stockout: float,
    ) -> str:
        if composite_risk >= 0.75:
            return (
                "Overall risk is high due to a combination of inventory pressure, disruption exposure, "
                "and supplier instability."
            )
        if composite_risk >= 0.45:
            return (
                "Overall risk is moderate as the system is balancing inventory coverage with rising disruption signals."
            )
        return "Overall risk is low because inventory, supplier, and disruption indicators are all within normal ranges."
