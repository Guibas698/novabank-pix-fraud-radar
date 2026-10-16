"""Motor Híbrido de Regras (YAML Hot-Reload) + Modelo de ML e Fila de Investigação do Analista.

Combina regras determinísticas de segurança bancária com o score probabilístico do modelo Champion:
- `RULE_01_ATO_CRITICAL`: Dispositivo novo + reset de senha há <= 30 min + dreno >= 80% -> `BLOCK`
- `RULE_02_MULE_RING_HIGH`: Conta destino com `receiver_mule_risk_score >= 0.65` -> `BLOCK`
- `RULE_03_SMURFING_BURST`: `tx_count_5m >= 3` com valores entre R$ 900 e R$ 1.000 -> `BLOCK`
- `RULE_04_NIGHT_NEW_KEY_HIGH_RATIO`: Madrugada + chave <= 3 dias + valor >= 5x média -> `STEP_UP_REVIEW`

Gerencia também a **Fila de Casos do Analista (`AnalystCaseQueue`)**, permitindo rotulagem
(`CONFIRMED_FRAUD` vs `FALSE_POSITIVE`) para retroalimentar o retreino semanal no MLflow.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from pix_fraud_radar.ml_engine import score_and_explain


@dataclass
class AnalystCase:
    case_id: str
    tx_id: str
    created_at: str
    sender_account_id: str
    receiver_account_id: str
    amount: float
    fraud_probability: float
    decision: str
    triggered_rules: list[str]
    top_reason_codes: list[str]
    status: str = "PENDING_REVIEW"  # PENDING_REVIEW | CONFIRMED_FRAUD | FALSE_POSITIVE
    analyst_notes: str = ""
    labeled_at: str | None = None


class HybridDecisionEngine:
    """Combina regras determinísticas e modelo de ML mantendo telemetria de latência (p50/p95/p99)."""

    def __init__(self, block_threshold: float = 0.72, review_threshold: float = 0.42) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.latencies_ms: list[float] = []
        self.cases: dict[str, AnalystCase] = {}
        self._case_seq = 0

    def evaluate_transaction(self, feature_row: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        prob, contribs, reasons = score_and_explain(feature_row)
        triggered_rules: list[str] = []

        if (
            int(feature_row.get("is_new_device_int", 0)) == 1
            and int(feature_row.get("recent_password_reset_int", 0)) == 1
            and float(feature_row.get("balance_drain_ratio", 0.0)) >= 0.75
        ):
            triggered_rules.append("RULE_01_ATO_CRITICAL")

        if float(feature_row.get("receiver_mule_risk_score", 0.0)) >= 0.60:
            triggered_rules.append("RULE_02_MULE_RING_HIGH")

        if int(feature_row.get("tx_count_5m", 0)) >= 3 and 850.0 <= float(feature_row.get("amount", 0.0)) <= 1050.0:
            triggered_rules.append("RULE_03_SMURFING_BURST")

        if (
            int(feature_row.get("is_night_hour", 0)) == 1
            and int(feature_row.get("receiver_key_age_days", 999)) <= 3
            and float(feature_row.get("amount_to_user_avg_ratio", 1.0)) >= 5.0
        ):
            triggered_rules.append("RULE_04_NIGHT_NEW_KEY_HIGH_RATIO")

        if any(r in triggered_rules for r in ("RULE_01_ATO_CRITICAL", "RULE_02_MULE_RING_HIGH", "RULE_03_SMURFING_BURST")) or prob >= self.block_threshold:
            decision = "BLOCK"
        elif triggered_rules or prob >= self.review_threshold:
            decision = "STEP_UP_REVIEW"
        else:
            decision = "APPROVE"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        self.latencies_ms.append(elapsed_ms)

        if decision in ("BLOCK", "STEP_UP_REVIEW"):
            self._case_seq += 1
            cid = f"CASE_{self._case_seq:05d}"
            self.cases[cid] = AnalystCase(
                case_id=cid,
                tx_id=feature_row["tx_id"],
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                sender_account_id=feature_row["sender_account_id"],
                receiver_account_id=feature_row["receiver_account_id"],
                amount=float(feature_row["amount"]),
                fraud_probability=prob,
                decision=decision,
                triggered_rules=triggered_rules,
                top_reason_codes=reasons,
            )

        return {
            "tx_id": feature_row["tx_id"],
            "decision": decision,
            "fraud_probability": prob,
            "triggered_rules": triggered_rules,
            "shap_contributions": contribs,
            "top_reason_codes": reasons,
            "latency_ms": elapsed_ms,
        }

    def label_case(self, case_id: str, label: str, notes: str = "") -> dict[str, Any]:
        """Rotula um caso na fila do analista para alimentar o retreino (Active Learning)."""
        if case_id not in self.cases:
            raise KeyError(f"Caso {case_id} não encontrado")
        label_up = label.upper().strip()
        if label_up not in ("CONFIRMED_FRAUD", "FALSE_POSITIVE"):
            raise ValueError("Label deve ser CONFIRMED_FRAUD ou FALSE_POSITIVE")
        case = self.cases[case_id]
        case.status = label_up
        case.analyst_notes = notes
        case.labeled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return asdict(case)

    def latency_percentiles(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        s = sorted(self.latencies_ms)
        n = len(s)
        return {
            "p50_ms": round(s[int(0.50 * (n - 1))], 4),
            "p95_ms": round(s[int(0.95 * (n - 1))], 4),
            "p99_ms": round(s[int(0.99 * (n - 1))], 4),
        }
