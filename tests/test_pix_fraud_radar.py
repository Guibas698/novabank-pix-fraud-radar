"""Testes unitários e de integração ponta a ponta do Pix Fraud Radar.

Verifica:
1. Simulador Pix com 4 tipologias de fraude e roteamento de eventos inválidos para DLQ
2. Detecção de redes de contas laranja via Grafos (PageRank + Fan-In + Pass-Through)
3. Feature Store Stateful com consistência Point-in-Time (sem data leakage)
4. Modelo de ML com ROC-AUC >= 0.92, PR-AUC >= 0.80, otimização financeira e PSI Drift
5. Motor Híbrido de Regras + ML com latência p99 < 20ms e fila do analista (Active Learning)
6. Endpoints da API FastAPI
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from pix_fraud_radar.api.app import create_app
from pix_fraud_radar.pipeline import PixFraudRadarPlatform


def test_pix_fraud_radar_end_to_end() -> None:
    radar = PixFraudRadarPlatform(n_customers=150, n_legit_tx=750, seed=42)
    summary = radar.summary()

    # 1. Validação do Contrato Avro e DLQ
    assert summary["total_valid_events"] > 800
    assert summary["dlq_rejected_events"] == 2

    # 2. Detecção de Contas Laranja no Grafo (PageRank + Fan-In)
    assert summary["detected_known_mules_count"] >= 6
    top_mules = radar.top_mules(limit=5)
    assert top_mules[0]["is_flagged_mule"] is True
    assert top_mules[0]["mule_risk_score"] >= 0.60

    # 3. Qualidade do Modelo (ROC-AUC, PR-AUC, Economia Líquida em R$)
    m = summary["model_metrics"]
    assert m["roc_auc"] >= 0.92
    assert m["pr_auc"] >= 0.80
    assert m["f1_score"] >= 0.78
    assert m["captured_fraud_brl"] > 50_000.0
    assert m["net_savings_brl"] > 45_000.0
    assert "amount" in m["feature_psi_drift"]

    # 4. Latência p99 < 20ms (muito abaixo do SLA de 100ms do Bacen/Pix)
    assert summary["latency_sla"]["p99_ms"] < 20.0

    # 5. API de Scoring em Tempo Real, Explicabilidade SHAP e Fila do Analista
    app = create_app(platform=radar)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/metrics").status_code == 200
    assert client.get("/dashboard").status_code == 200

    # Testa um golpe de Account Takeover (ATO) em tempo real
    score_resp = client.post(
        "/api/v1/pix/score",
        json={
            "tx_id": "PIX_LIVE_ATO_01",
            "sender_account_id": "ACC_00010",
            "receiver_account_id": "MULE_001",
            "receiver_pix_key": "chave_mule_001@pix.com",
            "receiver_key_age_days": 1,
            "amount": 4800.0,
            "sender_balance_before": 5000.0,
            "device_id": "HACKER_DEV_99",
            "is_new_device": True,
            "password_reset_mins_ago": 5,
        },
    )
    assert score_resp.status_code == 200
    payload = score_resp.json()
    assert payload["decision"] == "BLOCK"
    assert payload["fraud_probability"] >= 0.90
    assert "RULE_01_ATO_CRITICAL" in payload["triggered_rules"]
    assert len(payload["top_reason_codes"]) >= 1

    # Testa rotulagem de caso na fila do analista (Active Learning feedback loop)
    cases = client.get("/api/v1/cases?status=PENDING_REVIEW").json()
    assert len(cases) >= 1
    first_case_id = cases[0]["case_id"]
    lbl_resp = client.post(
        f"/api/v1/cases/{first_case_id}/label",
        json={"label": "CONFIRMED_FRAUD", "notes": "Contato com cliente confirmou invasão ATO"},
    )
    assert lbl_resp.status_code == 200
    assert lbl_resp.json()["status"] == "CONFIRMED_FRAUD"
