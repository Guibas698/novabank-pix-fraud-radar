"""Motor de Machine Learning, Explicabilidade (estilo SHAP), Limiar Ótimo Financeiro e Drift (PSI).

Funcionalidades:
1. **Scoring com Atribuição Aditiva de Features (SHAP-like log-odds contributions)**:
   - Calcula a probabilidade calibrada de fraude e decompõe exatamente quanto cada
     sinal contribuiu para a decisão final (`reason_codes`).
2. **Otimização de Limiar por Custo Financeiro (Business Metric Optimization)**:
   - Encontra o threshold ótimo que maximiza a economia líquida em reais:
     `Economia Líquida = R$ Fraudes Bloqueadas - (Falsos Positivos × R$ 8,50 custo de revisão/atrito)`.
3. **Métricas de Avaliação em Dados Desbalanceados**:
   - **ROC-AUC**, **PR-AUC (Precision-Recall AUC)**, **Precision**, **Recall** e **F1-Score**.
4. **Monitoramento de Drift com PSI (Population Stability Index)**:
   - Compara a distribuição das features de referência (treino) com a janela atual de produção:
     `PSI < 0.10` (Estável), `0.10 <= PSI < 0.25` (Atenção), `PSI >= 0.25` (Drift Crítico -> Dispara Retreino).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, x))))


@dataclass
class FraudPrediction:
    tx_id: str
    fraud_probability: float
    decision: str  # APPROVE | STEP_UP_REVIEW | BLOCK
    triggered_rules: list[str]
    shap_contributions: dict[str, float]
    top_reason_codes: list[str]


@dataclass
class ModelEvaluationMetrics:
    model_version: str
    optimal_threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    total_fraud_brl: float
    captured_fraud_brl: float
    false_positives_count: int
    net_savings_brl: float
    feature_psi_drift: dict[str, float] = field(default_factory=dict)
    drift_status: str = "STABLE"


def score_and_explain(feature_row: dict[str, Any]) -> tuple[float, dict[str, float], list[str]]:
    """Calcula a probabilidade de fraude e as contribuições individuais (SHAP-style) de cada feature."""
    base_logit = -3.85
    contribs: dict[str, float] = {}
    reasons: list[tuple[float, str]] = []

    # 1. Velocidade em 5 minutos (Smurfing / Estruturação)
    tx_5m = int(feature_row.get("tx_count_5m", 0))
    if tx_5m >= 3:
        c = 1.95 + 0.45 * (tx_5m - 3)
        contribs["tx_count_5m"] = round(c, 3)
        reasons.append((c, f"Alta velocidade: {tx_5m + 1} Pix em menos de 5 minutos (Smurfing)"))
    elif tx_5m >= 1:
        contribs["tx_count_5m"] = 0.35

    # 2. Razão entre valor e média histórica do pagador
    ratio_avg = float(feature_row.get("amount_to_user_avg_ratio", 1.0))
    if ratio_avg >= 5.5:
        c = min(2.4, 0.30 * ratio_avg)
        contribs["amount_to_user_avg_ratio"] = round(c, 3)
        reasons.append((c, f"Valor {ratio_avg:.1f}x acima da média histórica do cliente"))
    elif ratio_avg >= 3.0:
        contribs["amount_to_user_avg_ratio"] = 0.55

    # 3. Drenagem de saldo (Account Takeover)
    drain = float(feature_row.get("balance_drain_ratio", 0.0))
    if drain >= 0.80:
        c = 1.75
        contribs["balance_drain_ratio"] = c
        reasons.append((c, f"Consome {drain * 100:.0f}% do saldo disponível na conta"))

    # 4. Dispositivo novo + Reset recente de senha
    new_dev = int(feature_row.get("is_new_device_int", 0))
    pwd_reset = int(feature_row.get("recent_password_reset_int", 0))
    if new_dev and pwd_reset:
        c = 2.65
        contribs["device_and_password_reset"] = c
        reasons.append((c, "Dispositivo não reconhecido logo após redefinição de senha (ATO)"))
    elif new_dev:
        c = 1.05
        contribs["is_new_device_int"] = c
        reasons.append((c, "Transação originada de dispositivo novo"))

    # 5. Idade da chave Pix de destino e horário noturno (Golpe do Falso Parente)
    key_age = int(feature_row.get("receiver_key_age_days", 365))
    is_night = int(feature_row.get("is_night_hour", 0))
    if key_age <= 3:
        c = 1.65
        contribs["receiver_key_age_days"] = c
        reasons.append((c, f"Chave Pix recebedora recém-criada ({key_age} dias)"))
    elif key_age <= 7:
        contribs["receiver_key_age_days"] = 0.65

    if is_night:
        c = 0.95 if (key_age <= 5 or ratio_avg >= 4.0) else 0.30
        contribs["is_night_hour"] = c
        if c >= 0.80:
            reasons.append((c, "Transação de alto risco em horário noturno/madrugada"))

    # 6. Score de Grafo da conta recebedora (Rede de Conta Laranja)
    mule_score = float(feature_row.get("receiver_mule_risk_score", 0.0))
    rec_senders = int(feature_row.get("receiver_unique_senders_24h", 0))
    if mule_score >= 0.55 or rec_senders >= 5:
        c = round(max(1.85, 2.8 * mule_score), 3)
        contribs["receiver_mule_risk_score"] = c
        reasons.append((c, f"Conta destino em rede suspeita de laranja (Graph Risk={mule_score:.2f})"))

    logit = base_logit + sum(contribs.values())
    prob = round(_sigmoid(logit), 4)

    reasons.sort(key=lambda x: x[0], reverse=True)
    top_reasons = [desc for _, desc in reasons[:3]]
    if not top_reasons:
        top_reasons.append("Padrão comportamental compatível com o histórico do cliente")

    return prob, contribs, top_reasons


def compute_roc_and_pr_auc(y_true: list[int], y_score: list[float]) -> tuple[float, float]:
    """Calcula ROC-AUC e PR-AUC (Precision-Recall AUC) exatos."""
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # ROC-AUC via Mann-Whitney U
    asc_pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum = sum(idx for idx, (_, y) in enumerate(asc_pairs, start=1) if y == 1)
    roc_auc = (rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)

    # PR-AUC via integração trapezoidal na curva Precision-Recall ordenada decrescentemente
    desc_pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    prev_prec = 1.0
    pr_auc = 0.0
    for _, y in desc_pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / (tp + fp)
        pr_auc += (rec - prev_recall) * ((prec + prev_prec) / 2.0)
        prev_recall = rec
        prev_prec = prec

    return round(roc_auc, 4), round(pr_auc, 4)


def calculate_psi(expected: list[float], actual: list[float], bins: int = 5) -> float:
    """Calcula o Population Stability Index (PSI) entre duas amostras contínua/discreta."""
    if not expected or not actual:
        return 0.0
    sorted_exp = sorted(expected)
    n_exp = len(sorted_exp)
    n_act = len(actual)

    cuts = [sorted_exp[min(n_exp - 1, int(i * n_exp / bins))] for i in range(1, bins)]
    bounds = [-math.inf, *cuts, math.inf]

    psi_total = 0.0
    for i in range(bins):
        low, high = bounds[i], bounds[i + 1]
        exp_cnt = sum(1 for v in expected if low < v <= high) if i > 0 else sum(1 for v in expected if v <= high)
        act_cnt = sum(1 for v in actual if low < v <= high) if i > 0 else sum(1 for v in actual if v <= high)
        p_exp = max(0.001, exp_cnt / n_exp)
        p_act = max(0.001, act_cnt / n_act)
        psi_total += (p_act - p_exp) * math.log(p_act / p_exp)

    return round(abs(psi_total), 4)


def evaluate_and_optimize_model(
    feature_rows: list[dict[str, Any]],
    fp_review_cost_brl: float = 8.50,
    model_version: str = "champion-lgbm-v2.1",
) -> tuple[ModelEvaluationMetrics, list[dict[str, Any]]]:
    """Avalia o modelo sobre a tabela de features, encontra o limiar ótimo financeiro e calcula PSI."""
    y_true: list[int] = []
    y_score: list[float] = []
    amounts: list[float] = []
    scored_rows: list[dict[str, Any]] = []

    for row in feature_rows:
        prob, contribs, top_reasons = score_and_explain(row)
        y_true.append(1 if row["is_fraud"] else 0)
        y_score.append(prob)
        amounts.append(float(row["amount"]))
        scored_rows.append(
            {
                **row,
                "fraud_probability": prob,
                "shap_contributions": contribs,
                "top_reason_codes": top_reasons,
            }
        )

    roc_auc, pr_auc = compute_roc_and_pr_auc(y_true, y_score)

    # Busca de limiar ótimo financeiro (varre de 0.15 a 0.85)
    best_th = 0.50
    best_net_savings = -math.inf
    for step in range(15, 86, 5):
        th = step / 100.0
        captured_brl = sum(amt for y, s, amt in zip(y_true, y_score, amounts) if y == 1 and s >= th)
        fp_cnt = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s >= th)
        net = captured_brl - fp_cnt * fp_review_cost_brl
        if net > best_net_savings:
            best_net_savings = net
            best_th = th

    tp = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s >= best_th)
    fp = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s >= best_th)
    fn = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s < best_th)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    total_fraud_brl = round(sum(amt for y, amt in zip(y_true, amounts) if y == 1), 2)
    captured_brl = round(sum(amt for y, s, amt in zip(y_true, y_score, amounts) if y == 1 and s >= best_th), 2)

    # Monitoramento de Drift (compara primeira metade temporal vs segunda metade)
    mid = max(1, len(scored_rows) // 2)
    drift_features = ("amount", "amount_to_user_avg_ratio", "balance_drain_ratio", "fraud_probability")
    psi_map: dict[str, float] = {}
    for col in drift_features:
        ref_vals = [float(r[col]) for r in scored_rows[:mid]]
        cur_vals = [float(r[col]) for r in scored_rows[mid:]]
        psi_map[col] = calculate_psi(ref_vals, cur_vals)

    max_psi = max(psi_map.values()) if psi_map else 0.0
    drift_status = "CRITICAL_DRIFT" if max_psi >= 0.25 else ("WARNING_DRIFT" if max_psi >= 0.10 else "STABLE")

    metrics = ModelEvaluationMetrics(
        model_version=model_version,
        optimal_threshold=best_th,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        total_fraud_brl=total_fraud_brl,
        captured_fraud_brl=captured_brl,
        false_positives_count=fp,
        net_savings_brl=round(best_net_savings, 2),
        feature_psi_drift=psi_map,
        drift_status=drift_status,
    )
    return metrics, scored_rows


def metrics_to_dict(metrics: ModelEvaluationMetrics) -> dict[str, Any]:
    return asdict(metrics)
