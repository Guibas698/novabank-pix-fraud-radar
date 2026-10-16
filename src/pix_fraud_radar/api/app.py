"""FastAPI Real-Time Pix Fraud Scoring Service, Graph Mule Monitor & Analyst Case Queue UI."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pix_fraud_radar.pipeline import PixFraudRadarPlatform


class PixRealtimeScoreRequest(BaseModel):
    tx_id: str = Field(default="PIX_RT_99901")
    sender_account_id: str = Field(default="ACC_00001")
    receiver_account_id: str = Field(default="MULE_001")
    receiver_pix_key: str = Field(default="chave_mule_001@pix.com")
    receiver_key_age_days: int = Field(default=1, ge=0)
    amount: float = Field(default=2450.0, gt=0)
    sender_balance_before: float = Field(default=2600.0, gt=0)
    device_id: str = Field(default="NEW_DEV_X99")
    is_new_device: bool = Field(default=True)
    password_reset_mins_ago: int | None = Field(default=8)


class CaseLabelRequest(BaseModel):
    label: str = Field(..., description="CONFIRMED_FRAUD ou FALSE_POSITIVE")
    notes: str = Field(default="", description="Observação da análise antifraude")


def create_app(platform: PixFraudRadarPlatform | None = None) -> FastAPI:
    radar = platform or PixFraudRadarPlatform(n_customers=150, n_legit_tx=750, seed=42)

    app = FastAPI(
        title="NovaBank Pix Fraud Radar — Real-Time Scoring & MLOps API",
        description=(
            "Motor de prevenção a fraudes Pix em tempo real (< 10ms) com Feature Store Stateful, "
            "detecção de contas laranja via Grafos (PageRank), explicabilidade SHAP e monitoramento de Drift (PSI)."
        ),
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": "novabank-pix-fraud-radar"}

    @app.get("/api/v1/metrics")
    def get_metrics() -> dict[str, Any]:
        return radar.summary()

    @app.post("/api/v1/pix/score")
    def score_pix_realtime(req: PixRealtimeScoreRequest) -> dict[str, Any]:
        event = {
            "tx_id": req.tx_id,
            "event_ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            "sender_account_id": req.sender_account_id,
            "receiver_account_id": req.receiver_account_id,
            "receiver_pix_key": req.receiver_pix_key,
            "receiver_key_age_days": req.receiver_key_age_days,
            "amount": req.amount,
            "sender_balance_before": req.sender_balance_before,
            "device_id": req.device_id,
            "is_new_device": req.is_new_device,
            "password_reset_mins_ago": req.password_reset_mins_ago,
        }
        feat_row = radar.feature_store.compute_and_update(event)
        decision = radar.decision_engine.evaluate_transaction(feat_row)
        return {**decision, "computed_features": feat_row}

    @app.get("/api/v1/graph/mules")
    def get_flagged_mules(limit: int = Query(15, ge=1, le=100)) -> list[dict[str, Any]]:
        return radar.top_mules(limit=limit)

    @app.get("/api/v1/cases")
    def list_cases(
        status: str | None = Query(None, description="PENDING_REVIEW | CONFIRMED_FRAUD | FALSE_POSITIVE"),
        limit: int = Query(30, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        cases = [asdict(c) for c in radar.decision_engine.cases.values()]
        if status:
            cases = [c for c in cases if c["status"] == status.upper()]
        return cases[:limit]

    @app.post("/api/v1/cases/{case_id}/label")
    def label_analyst_case(case_id: str, req: CaseLabelRequest) -> dict[str, Any]:
        try:
            return radar.decision_engine.label_case(case_id, req.label, req.notes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        s = radar.summary()
        m = s["model_metrics"]
        lat = s["latency_sla"]
        return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<title>NovaBank Pix Fraud Radar — Real-Time ML & Graph Analytics</title>
<style>
  body {{ background:#090d16; color:#e2e8f0; font-family:Inter,system-ui,sans-serif; margin:0; padding:32px; }}
  h1 {{ margin:0 0 8px; font-size:28px; color:#f43f5e; }}
  .sub {{ color:#94a3b8; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin-bottom:28px; }}
  .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px; }}
  .val {{ font-size:26px; font-weight:700; color:#f8fafc; }}
  .lbl {{ font-size:13px; color:#94a3b8; margin-top:4px; }}
</style>
</head>
<body>
  <h1>🚨 NovaBank Pix Fraud Radar</h1>
  <div class="sub">Scala/Python Stateful Streaming • Graph Mule Detection (PageRank) • SHAP Explainability • PSI Drift</div>
  <div class="grid">
    <div class="card"><div class="val">{m['roc_auc']}</div><div class="lbl">ROC-AUC (Champion Model)</div></div>
    <div class="card"><div class="val">{m['pr_auc']}</div><div class="lbl">PR-AUC (Precision-Recall)</div></div>
    <div class="card"><div class="val">R$ {m['captured_fraud_brl']:,.2f}</div><div class="lbl">Fraude Pix Evitada (R$)</div></div>
    <div class="card"><div class="val">R$ {m['net_savings_brl']:,.2f}</div><div class="lbl">Economia Líquida (após custo FP)</div></div>
    <div class="card"><div class="val">{s['detected_known_mules_count']}/{s['known_mules_count']}</div><div class="lbl">Contas Laranja Detectadas no Grafo</div></div>
    <div class="card"><div class="val">{lat['p99_ms']:.2f} ms</div><div class="lbl">Latência de Scoring p99 (SLA &lt; 100ms)</div></div>
    <div class="card"><div class="val">{m['drift_status']}</div><div class="lbl">Status de Drift (PSI)</div></div>
    <div class="card"><div class="val">{s['open_analyst_cases']}</div><div class="lbl">Alertas na Fila Antifraude</div></div>
  </div>
  <p>Teste o scoring síncrono com explicabilidade SHAP em <a style="color:#f43f5e" href="/docs">/docs</a>.</p>
</body>
</html>"""

    return app


app = create_app()
