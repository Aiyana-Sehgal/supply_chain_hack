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
        supplier_risk = float(state.get("supplier_risk_score", 0.0))
        disruption_signal = float(state.get("disruption_signal", 0.0))
        days_to_stockout = float(state.get("days_to_stockout", 0.0))
        predicted_demand = float(state.get("predicted_demand", 0.0))
        current_inventory = float(state.get("current_inventory", 0.0))
        inventory_ratio = current_inventory / max(predicted_demand, 1.0)

        drivers = []
        if days_to_stockout <= 3:
            drivers.append("days to stockout are critically low")
        elif days_to_stockout <= 7:
            drivers.append("inventory coverage is tight")

        if inventory_ratio < 0.1:
            drivers.append("inventory is far below projected demand")
        elif inventory_ratio < 0.3:
            drivers.append("inventory is below the preferred buffer")

        if supplier_risk >= 0.7:
            drivers.append("supplier risk is elevated")
        if disruption_signal >= 0.6:
            drivers.append("NewsData.io disruption signal is high")

        if not drivers:
            drivers.append("current conditions are stable")

        action_guidance = {
            "do_nothing": "hold current operations and continue monitoring incoming signals",
            "reorder_stock": "increase replenishment to protect service levels",
            "switch_supplier": "shift sourcing to reduce supplier-side execution risk",
            "reroute_shipment": "change logistics routing to avoid disruption hotspots",
            "emergency_restock": "use expedited replenishment to prevent an imminent stockout",
        }

        rationale = (
            f"Recommended action is {action.replace('_', ' ')} because "
            + ", ".join(drivers)
            + f". The decision aims to {action_guidance.get(action, 'reduce operational risk')}."
        )

        return {
            "action": action,
            "confidence": confidence,
            "rationale": rationale,
            "primary_drivers": drivers,
        }

    def explain_scenario_impact(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        impact = scenario_results.get("impact_analysis", {})
        risk_breakdown = scenario_results.get("risk_breakdown", {})
        recommendations = scenario_results.get("recommendations", {})

        cost_impact = float(impact.get("cost_impact", 0.0))
        stockout_probability = float(impact.get("stockout_probability", 0.0))
        delay_probability = float(impact.get("delay_probability", 0.0))

        factors = []
        if stockout_probability >= 0.6:
            factors.append("stockout probability is materially elevated")
        if delay_probability >= 0.6:
            factors.append("delay probability is materially elevated")
        if risk_breakdown.get("supplier_risk", 0.0) >= 0.7:
            factors.append("supplier exposure is high")
        if risk_breakdown.get("disruption_risk", 0.0) >= 0.6:
            factors.append("hyperlocal disruption signals remain elevated")
        if not factors:
            factors.append("risk remains within the normal operating band")

        summary = (
            f"Scenario impact shows an estimated cost change of {cost_impact:,.0f}, "
            f"stockout probability of {stockout_probability:.0%}, and delay probability of {delay_probability:.0%}. "
            f"Primary drivers: {', '.join(factors)}. "
            f"Recommended response is {recommendations.get('action', 'do_nothing').replace('_', ' ')}."
        )

        return {
            "scenario_summary": summary,
            "key_factors": factors,
        }

    def _explain_supplier_risk(self, supplier_risk: float) -> str:
        if supplier_risk >= 0.8:
            return "High supplier risk due to weak supply-side resilience and elevated failure likelihood."
        if supplier_risk >= 0.5:
            return "Moderate supplier risk driven by unstable fulfillment conditions."
        return "Supplier risk is currently low and does not appear to be the main pressure point."

    def _explain_disruption_risk(self, disruption_signal: float) -> str:
        if disruption_signal >= 0.8:
            return "High risk due to strong NewsData.io disruption signals such as transport or regional disturbance alerts."
        if disruption_signal >= 0.5:
            return "Moderate disruption risk from India-focused news signals that may affect movement or demand."
        return "NewsData.io disruption signal is low, suggesting limited external disruption pressure."

    def _explain_inventory_risk(self, days_to_stockout: float, inventory_ratio: float) -> str:
        if days_to_stockout <= 3 or inventory_ratio < 0.05:
            return "Inventory risk is critical because stock could run out very soon under current demand."
        if days_to_stockout <= 7 or inventory_ratio < 0.15:
            return "Inventory risk is elevated because the current buffer may not absorb a near-term demand spike."
        return "Inventory position is healthy relative to expected demand."

    def _explain_composite_risk(
        self,
        composite_risk: float,
        supplier_risk: float,
        disruption_signal: float,
        days_to_stockout: float,
    ) -> str:
        if composite_risk >= 0.75:
            return (
                "Overall risk is high due to a combination of supplier instability, disruption exposure, "
                "and limited inventory coverage."
            )
        if composite_risk >= 0.45:
            return (
                "Overall risk is moderate, shaped by a mix of operational and external disruption indicators."
            )
        if supplier_risk < 0.4 and disruption_signal < 0.4 and days_to_stockout > 14:
            return "Overall risk is low because supplier, disruption, and inventory indicators are all stable."
        return "Overall risk is manageable but should continue to be monitored."
