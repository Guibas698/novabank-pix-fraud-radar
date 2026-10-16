"""Detecção de Redes de Contas Laranja (Money Mule Rings) via Análise de Grafos.

Constrói o grafo direcionado de transferências Pix `G = (V, E)` e calcula para cada conta:
1. **PageRank Iterativo** (`pagerank_score`): mede a centralidade de fluxo financeiro recebido.
2. **Fan-In (In-Degree de Pagadores Únicos)**: quantas contas distintas enviaram Pix para o recebedor.
3. **Pass-Through Ratio (Taxa de Escoamento)**: proporção do volume recebido que é rapidamente
   repassada para outra conta centralizadora (comportamento clássico de 'conta de passagem' / laranja).
4. **Mule Risk Score (`0.0` a `1.0`)**: score composto que alimenta a Feature Store Online para
   bloqueio em tempo real de transferências destinadas a contas laranja.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AccountGraphMetrics:
    account_id: str
    unique_senders_in: int
    unique_receivers_out: int
    total_amount_in: float
    total_amount_out: float
    pass_through_ratio: float
    pagerank: float
    mule_risk_score: float
    is_flagged_mule: bool


def analyze_transfer_graph(
    events: list[dict[str, Any]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, AccountGraphMetrics]:
    """Executa PageRank e análise topológica Fan-In/Pass-Through sobre o grafo de transações Pix."""
    nodes: set[str] = set()
    out_edges: dict[str, set[str]] = defaultdict(set)
    in_edges: dict[str, set[str]] = defaultdict(set)
    amount_in: dict[str, float] = defaultdict(float)
    amount_out: dict[str, float] = defaultdict(float)

    for ev in events:
        src = ev["sender_account_id"]
        dst = ev["receiver_account_id"]
        amt = float(ev["amount"])
        nodes.add(src)
        nodes.add(dst)
        out_edges[src].add(dst)
        in_edges[dst].add(src)
        amount_out[src] += amt
        amount_in[dst] += amt

    if not nodes:
        return {}

    n_nodes = len(nodes)
    pr: dict[str, float] = {u: 1.0 / n_nodes for u in nodes}

    for _ in range(iterations):
        next_pr: dict[str, float] = {}
        sink_sum = sum(pr[u] for u in nodes if not out_edges[u])
        base_val = (1.0 - damping) / n_nodes + damping * (sink_sum / n_nodes)
        for v in nodes:
            incoming = sum(pr[u] / len(out_edges[u]) for u in in_edges[v] if out_edges[u])
            next_pr[v] = base_val + damping * incoming
        pr = next_pr

    max_pr = max(pr.values()) if pr else 1.0
    results: dict[str, AccountGraphMetrics] = {}

    for acc in sorted(nodes):
        u_in = len(in_edges[acc])
        u_out = len(out_edges[acc])
        tot_in = round(amount_in[acc], 2)
        tot_out = round(amount_out[acc], 2)
        pass_ratio = round(min(1.5, tot_out / tot_in), 4) if tot_in > 0 else 0.0
        norm_pr = pr[acc] / max_pr if max_pr > 0 else 0.0

        # Score de risco de conta laranja:
        # Combina alto Fan-In (>= 5 pagadores distintos), alto PageRank ou alta taxa de repasse (pass-through)
        fan_in_signal = min(1.0, u_in / 7.0)
        pass_signal = 1.0 if (u_in >= 5 and 0.65 <= pass_ratio <= 1.25) else (0.5 if u_in >= 6 else 0.0)
        hub_signal = min(1.0, norm_pr * 1.2) if tot_in >= 4000.0 else 0.0

        risk = round(min(1.0, 0.45 * fan_in_signal + 0.30 * pass_signal + 0.25 * hub_signal), 4)
        flagged = risk >= 0.58 or (u_in >= 6 and tot_in >= 5000.0)

        results[acc] = AccountGraphMetrics(
            account_id=acc,
            unique_senders_in=u_in,
            unique_receivers_out=u_out,
            total_amount_in=tot_in,
            total_amount_out=tot_out,
            pass_through_ratio=pass_ratio,
            pagerank=round(pr[acc], 6),
            mule_risk_score=risk,
            is_flagged_mule=flagged,
        )

    return results


def export_mule_rankings(metrics: dict[str, AccountGraphMetrics]) -> list[dict[str, Any]]:
    """Retorna a lista de contas ordenadas pelo score de risco de conta laranja."""
    ranked = sorted(
        metrics.values(),
        key=lambda m: (m.is_flagged_mule, m.mule_risk_score, m.total_amount_in),
        reverse=True,
    )
    return [asdict(m) for m in ranked]
