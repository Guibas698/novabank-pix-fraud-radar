"""Engenharia de Features Stateful em Streaming e Feature Store (Online + Offline Point-in-Time).

Calcula janelas deslizantes em tempo de evento (`event_ts`) por pagador e recebedor,
garantindo **Point-in-Time Correctness** (cada transação enxerga apenas os eventos
ocorridos estritamente antes dela, eliminando *data leakage* entre treino e inferência):

Features calculadas em tempo real:
- `tx_count_5m`: quantidade de Pix enviados pelo pagador nos últimos 5 minutos
- `tx_amount_sum_1h`: soma em R$ enviada pelo pagador na última 1 hora
- `distinct_receivers_1h`: recebedores distintos acionados na última 1 hora
- `amount_to_user_avg_ratio`: razão entre o valor do Pix atual e a média móvel histórica do cliente
- `balance_drain_ratio`: fração do saldo prévio consumida pela transação (`amount / sender_balance_before`)
- `is_night_hour`: 1 se o evento ocorre entre 22h e 05h59, 0 caso contrário
- `is_new_device_int`: 1 se o dispositivo nunca foi usado antes pelo cliente
- `recent_password_reset_int`: 1 se houve troca de senha há menos de 30 minutos
- `receiver_key_age_days`: idade da chave Pix de destino em dias
- `receiver_unique_senders_24h`: pagadores distintos que enviaram para essa conta/chave nas últimas 24h
- `receiver_mule_risk_score`: score de grafo (PageRank + Fan-In) da conta recebedora
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pix_fraud_radar.graph_mule_detector import AccountGraphMetrics


@dataclass
class SenderWindowState:
    """Estado mantido em memória (Redis / Spark State Store) para cada conta pagadora."""

    recent_events: deque[tuple[datetime, float, str]] = field(default_factory=deque)
    historical_count: int = 0
    historical_sum: float = 0.0


@dataclass
class ReceiverWindowState:
    """Estado mantido para cada conta recebedora nas últimas 24h."""

    recent_senders: deque[tuple[datetime, str, float]] = field(default_factory=deque)


class StatefulFeatureStore:
    """Feature Store Híbrida: mantém estado Online (baixa latência) e gera tabela Offline Point-in-Time."""

    def __init__(self, graph_metrics: dict[str, AccountGraphMetrics] | None = None) -> None:
        self.sender_state: dict[str, SenderWindowState] = {}
        self.receiver_state: dict[str, ReceiverWindowState] = {}
        self.graph_metrics: dict[str, AccountGraphMetrics] = graph_metrics or {}
        self.offline_feature_table: list[dict[str, Any]] = []

    def compute_and_update(self, event: dict[str, Any]) -> dict[str, Any]:
        """Calcula o vetor de features point-in-time para `event` e atualiza o estado em janela."""
        ts = datetime.fromisoformat(event["event_ts"])
        sender_id = event["sender_account_id"]
        receiver_id = event["receiver_account_id"]
        amount = float(event["amount"])
        bal_before = max(1.0, float(event.get("sender_balance_before") or amount))

        s_state = self.sender_state.setdefault(sender_id, SenderWindowState())
        r_state = self.receiver_state.setdefault(receiver_id, ReceiverWindowState())

        # Descarta eventos fora da janela de 1h do pagador e 24h do recebedor
        cutoff_1h = ts - timedelta(hours=1)
        cutoff_5m = ts - timedelta(minutes=5)
        cutoff_24h = ts - timedelta(hours=24)

        while s_state.recent_events and s_state.recent_events[0][0] < cutoff_1h:
            s_state.recent_events.popleft()

        while r_state.recent_senders and r_state.recent_senders[0][0] < cutoff_24h:
            r_state.recent_senders.popleft()

        # Cálculo estritamente antes do evento atual (Point-in-Time Correctness)
        tx_count_5m = sum(1 for ev_ts, _, _ in s_state.recent_events if ev_ts >= cutoff_5m)
        tx_amount_sum_1h = round(sum(ev_amt for _, ev_amt, _ in s_state.recent_events), 2)
        distinct_receivers_1h = len({rec for _, _, rec in s_state.recent_events})

        user_avg = (
            s_state.historical_sum / s_state.historical_count
            if s_state.historical_count >= 1
            else 180.0
        )
        amount_to_avg_ratio = round(amount / max(10.0, user_avg), 4)
        balance_drain_ratio = round(min(1.5, amount / bal_before), 4)

        is_night = 1 if (ts.hour >= 22 or ts.hour <= 5) else 0
        is_new_dev = 1 if event.get("is_new_device") else 0
        pwd_mins = event.get("password_reset_mins_ago")
        recent_pwd_reset = 1 if (pwd_mins is not None and int(pwd_mins) <= 30) else 0
        key_age_days = int(event.get("receiver_key_age_days", 365))

        rec_unique_senders_24h = len({snd for _, snd, _ in r_state.recent_senders})
        g_metric = self.graph_metrics.get(receiver_id)
        mule_risk = g_metric.mule_risk_score if g_metric else 0.0

        feature_row = {
            "tx_id": event["tx_id"],
            "event_ts": event["event_ts"],
            "sender_account_id": sender_id,
            "receiver_account_id": receiver_id,
            "amount": amount,
            "tx_count_5m": tx_count_5m,
            "tx_amount_sum_1h": tx_amount_sum_1h,
            "distinct_receivers_1h": distinct_receivers_1h,
            "amount_to_user_avg_ratio": amount_to_avg_ratio,
            "balance_drain_ratio": balance_drain_ratio,
            "is_night_hour": is_night,
            "is_new_device_int": is_new_dev,
            "recent_password_reset_int": recent_pwd_reset,
            "receiver_key_age_days": key_age_days,
            "receiver_unique_senders_24h": rec_unique_senders_24h,
            "receiver_mule_risk_score": mule_risk,
            "is_fraud": bool(event.get("is_fraud", False)),
            "fraud_type": event.get("fraud_type", "LEGITIMO"),
        }

        # Atualiza estado após computar as features
        s_state.recent_events.append((ts, amount, receiver_id))
        if not event.get("is_fraud", False):
            s_state.historical_count += 1
            s_state.historical_sum += amount
        r_state.recent_senders.append((ts, sender_id, amount))
        self.offline_feature_table.append(feature_row)
        return feature_row

    def process_stream(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Processa uma sequência de eventos em ordem temporal."""
        return [self.compute_and_update(ev) for ev in events]
