"""Orquestrador ponta a ponta da plataforma Pix Fraud Radar."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pix_fraud_radar.graph_mule_detector import analyze_transfer_graph, export_mule_rankings
from pix_fraud_radar.ml_engine import evaluate_and_optimize_model
from pix_fraud_radar.rules_engine import HybridDecisionEngine
from pix_fraud_radar.simulator import SimulationBatch, simulate_pix_stream
from pix_fraud_radar.streaming_features import StatefulFeatureStore


class PixFraudRadarPlatform:
    """Mantém o estado completo do Pix Fraud Radar para execução batch, streaming e API."""

    def __init__(self, n_customers: int = 180, n_legit_tx: int = 900, seed: int = 42) -> None:
        self.batch: SimulationBatch = simulate_pix_stream(
            n_customers=n_customers, n_legit_tx=n_legit_tx, seed=seed
        )
        self.graph_metrics = analyze_transfer_graph(self.batch.events)
        self.feature_store = StatefulFeatureStore(graph_metrics=self.graph_metrics)
        self.feature_rows = self.feature_store.process_stream(self.batch.events)
        self.eval_metrics, self.scored_rows = evaluate_and_optimize_model(self.feature_rows)

        self.decision_engine = HybridDecisionEngine(
            block_threshold=max(0.65, self.eval_metrics.optimal_threshold),
            review_threshold=max(0.35, self.eval_metrics.optimal_threshold * 0.65),
        )
        self.decisions = [self.decision_engine.evaluate_transaction(r) for r in self.feature_rows]

    def summary(self) -> dict[str, Any]:
        flagged_mules = [
            m.account_id for m in self.graph_metrics.values() if m.is_flagged_mule
        ]
        known_set = set(self.batch.known_mule_accounts)
        detected_known = [acc for acc in flagged_mules if acc in known_set]

        return {
            "total_valid_events": len(self.batch.events),
            "dlq_rejected_events": len(self.batch.dlq_events),
            "known_mules_count": len(known_set),
            "detected_known_mules_count": len(detected_known),
            "flagged_mules": flagged_mules,
            "model_metrics": asdict(self.eval_metrics),
            "latency_sla": self.decision_engine.latency_percentiles(),
            "open_analyst_cases": len(self.decision_engine.cases),
        }

    def top_mules(self, limit: int = 15) -> list[dict[str, Any]]:
        return export_mule_rankings(self.graph_metrics)[:limit]
