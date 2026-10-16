"""Simulador realista de transações Pix com perfis comportamentais e 4 tipologias de fraude.

Tipologias de golpe injetadas com gabarito (`is_fraud` e `fraud_type`):
1. `GOLPE_FALSO_PARENTE`: chave recém-criada (`receiver_key_age_days <= 2`), valor muito acima
   da média do cliente, horário noturno/madrugada.
2. `INVASAO_CONTA_ATO` (Account Takeover): dispositivo novo (`is_new_device=True`), troca
   recente de senha (`password_reset_mins_ago <= 30`) e drenagem de saldo (`balance_drain_ratio >= 0.80`).
3. `REDE_CONTA_LARANJA` (Money Mule Ring): dezenas de pagadores enviando valores para um
   pequeno grupo de contas laranja (`MULE_001`..`MULE_008`), que escoam para uma conta chefe (`BOSS_001`).
4. `SMURFING_ESTRUTURACAO`: rajada de transferências seguidas logo abaixo de R$ 1.000 em poucos minutos.

Também valida o contrato de schema (estilo Avro/Schema Registry) e envia eventos malformados
para a Dead Letter Queue (`dlq_events`).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class CustomerPixProfile:
    account_id: str
    persona: str  # ASSALARIADO | AUTONOMO | COMERCIANTE | ESTUDANTE
    avg_pix_amount: float
    typical_balance: float
    usual_device_id: str


@dataclass
class PixEvent:
    tx_id: str
    event_ts: str
    sender_account_id: str
    sender_persona: str
    receiver_account_id: str
    receiver_pix_key: str
    receiver_key_age_days: int
    amount: float
    sender_balance_before: float
    device_id: str
    is_new_device: bool
    password_reset_mins_ago: int | None
    channel: str
    is_fraud: bool = False
    fraud_type: str = "LEGITIMO"


@dataclass
class SimulationBatch:
    events: list[dict[str, Any]] = field(default_factory=list)
    dlq_events: list[dict[str, Any]] = field(default_factory=list)
    profiles: dict[str, CustomerPixProfile] = field(default_factory=dict)
    known_mule_accounts: list[str] = field(default_factory=list)


def validate_avro_contract(raw_event: dict[str, Any]) -> tuple[bool, str | None]:
    """Valida contrato estrito de schema (estilo Confluent Schema Registry / Avro)."""
    required_fields = (
        "tx_id", "event_ts", "sender_account_id", "receiver_account_id",
        "receiver_pix_key", "amount", "sender_balance_before",
    )
    for col in required_fields:
        if raw_event.get(col) is None or str(raw_event.get(col)).strip() == "":
            return False, f"MISSING_REQUIRED_FIELD:{col}"
    try:
        amt = float(raw_event["amount"])
        if amt <= 0:
            return False, "NON_POSITIVE_AMOUNT"
    except (TypeError, ValueError):
        return False, "INVALID_AMOUNT_TYPE"
    return True, None


def simulate_pix_stream(
    n_customers: int = 200,
    n_legit_tx: int = 1200,
    seed: int = 42,
) -> SimulationBatch:
    """Gera stream determinístico de transações Pix legítimas + 4 tipologias de fraude + DLQ."""
    rng = random.Random(seed)
    batch = SimulationBatch()

    personas = [
        ("ASSALARIADO", 140.0, 2800.0),
        ("AUTONOMO", 320.0, 5500.0),
        ("COMERCIANTE", 680.0, 14000.0),
        ("ESTUDANTE", 55.0, 650.0),
    ]

    for i in range(1, n_customers + 1):
        acc_id = f"ACC_{i:05d}"
        p_name, base_avg, base_bal = rng.choice(personas)
        avg_amt = round(base_avg * rng.uniform(0.7, 1.4), 2)
        bal = round(base_bal * rng.uniform(0.6, 1.6), 2)
        dev_id = f"DEV_{uuid.UUID(int=rng.getrandbits(128), version=4).hex[:10]}"
        batch.profiles[acc_id] = CustomerPixProfile(
            account_id=acc_id,
            persona=p_name,
            avg_pix_amount=avg_amt,
            typical_balance=bal,
            usual_device_id=dev_id,
        )

    mule_accounts = [f"MULE_{i:03d}" for i in range(1, 7)]
    boss_account = "MULE_BOSS_001"
    batch.known_mule_accounts = [*mule_accounts, boss_account]

    base_ts = datetime(2026, 3, 1, 8, 0, 0)
    account_ids = list(batch.profiles.keys())
    raw_candidates: list[dict[str, Any]] = []

    # 1. Transações legítimas com sazonalidade diurna
    for idx in range(n_legit_tx):
        sender_id = rng.choice(account_ids)
        receiver_id = rng.choice(account_ids)
        while receiver_id == sender_id:
            receiver_id = rng.choice(account_ids)
        prof = batch.profiles[sender_id]
        offset_mins = rng.randint(0, 48 * 60)
        ts = base_ts + timedelta(minutes=offset_mins, seconds=rng.randint(0, 59))
        # Horário comercial muito mais provável para transações legítimas
        if ts.hour < 7 or ts.hour > 22:
            ts = ts.replace(hour=rng.randint(8, 20))

        amt = round(max(5.0, rng.lognormvariate(0.0, 0.45) * prof.avg_pix_amount), 2)
        bal_before = round(max(amt * 1.5, prof.typical_balance * rng.uniform(0.8, 1.3)), 2)

        ev = PixEvent(
            tx_id=f"PIX_L_{idx + 1:06d}",
            event_ts=ts.isoformat(timespec="seconds"),
            sender_account_id=sender_id,
            sender_persona=prof.persona,
            receiver_account_id=receiver_id,
            receiver_pix_key=f"chave_{receiver_id.lower()}@novabank.com.br",
            receiver_key_age_days=rng.randint(30, 900),
            amount=amt,
            sender_balance_before=bal_before,
            device_id=prof.usual_device_id,
            is_new_device=False,
            password_reset_mins_ago=None,
            channel="APP_MOBILE",
            is_fraud=False,
            fraud_type="LEGITIMO",
        )
        raw_candidates.append(asdict(ev))

    # 2. Fraude Tipo 1: Golpe do Falso Parente / WhatsApp (madrugada, chave nova, valor alto)
    for idx in range(35):
        sender_id = rng.choice(account_ids)
        prof = batch.profiles[sender_id]
        ts = base_ts + timedelta(days=rng.randint(0, 1), hours=rng.choice([23, 0, 1, 2, 3, 4]), minutes=rng.randint(0, 59))
        amt = round(prof.avg_pix_amount * rng.uniform(6.5, 14.0), 2)
        bal_before = round(amt * rng.uniform(1.05, 1.35), 2)
        scam_rec = f"SCAM_FP_{idx + 1:03d}"
        ev = PixEvent(
            tx_id=f"PIX_FP_{idx + 1:05d}",
            event_ts=ts.isoformat(timespec="seconds"),
            sender_account_id=sender_id,
            sender_persona=prof.persona,
            receiver_account_id=scam_rec,
            receiver_pix_key=f"aleatoria_{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
            receiver_key_age_days=rng.randint(0, 2),
            amount=amt,
            sender_balance_before=bal_before,
            device_id=prof.usual_device_id,
            is_new_device=False,
            password_reset_mins_ago=None,
            channel="APP_MOBILE",
            is_fraud=True,
            fraud_type="GOLPE_FALSO_PARENTE",
        )
        raw_candidates.append(asdict(ev))

    # 3. Fraude Tipo 2: Invasão de Conta / Account Takeover (dispositivo novo + reset de senha + dreno de saldo)
    for idx in range(35):
        sender_id = rng.choice(account_ids)
        prof = batch.profiles[sender_id]
        ts = base_ts + timedelta(hours=rng.randint(1, 40), minutes=rng.randint(0, 59))
        bal_before = round(prof.typical_balance * rng.uniform(1.1, 2.0), 2)
        amt = round(bal_before * rng.uniform(0.85, 0.99), 2)
        ato_rec = rng.choice(mule_accounts)
        ev = PixEvent(
            tx_id=f"PIX_ATO_{idx + 1:05d}",
            event_ts=ts.isoformat(timespec="seconds"),
            sender_account_id=sender_id,
            sender_persona=prof.persona,
            receiver_account_id=ato_rec,
            receiver_pix_key=f"pix_{ato_rec.lower()}@laranja.org",
            receiver_key_age_days=rng.randint(1, 5),
            amount=amt,
            sender_balance_before=bal_before,
            device_id=f"NEW_DEV_{idx:04d}",
            is_new_device=True,
            password_reset_mins_ago=rng.randint(2, 20),
            channel="APP_MOBILE",
            is_fraud=True,
            fraud_type="INVASAO_CONTA_ATO",
        )
        raw_candidates.append(asdict(ev))

    # 4. Fraude Tipo 3: Rede de Contas Laranja (Fan-In de vítimas -> Mules -> Fan-Out para Boss)
    seq_mule = 0
    for mule_id in mule_accounts:
        victims = rng.sample(account_ids, k=8)
        t0 = base_ts + timedelta(hours=rng.randint(5, 30))
        collected = 0.0
        for v_id in victims:
            seq_mule += 1
            prof = batch.profiles[v_id]
            amt = round(rng.uniform(850.0, 2600.0), 2)
            collected += amt
            t_in = t0 + timedelta(minutes=rng.randint(1, 45))
            raw_candidates.append(
                asdict(
                    PixEvent(
                        tx_id=f"PIX_MULE_IN_{seq_mule:05d}",
                        event_ts=t_in.isoformat(timespec="seconds"),
                        sender_account_id=v_id,
                        sender_persona=prof.persona,
                        receiver_account_id=mule_id,
                        receiver_pix_key=f"chave_{mule_id.lower()}@pix.com",
                        receiver_key_age_days=rng.randint(1, 4),
                        amount=amt,
                        sender_balance_before=round(amt * 1.2, 2),
                        device_id=prof.usual_device_id,
                        is_new_device=rng.choice([True, False]),
                        password_reset_mins_ago=rng.choice([None, 12]),
                        channel="APP_MOBILE",
                        is_fraud=True,
                        fraud_type="REDE_CONTA_LARANJA",
                    )
                )
            )
        # Escoamento da conta laranja para a conta chefe (Boss) e entre laranjas (anel)
        seq_mule += 1
        t_out = t0 + timedelta(minutes=55)
        raw_candidates.append(
            asdict(
                PixEvent(
                    tx_id=f"PIX_MULE_OUT_{seq_mule:05d}",
                    event_ts=t_out.isoformat(timespec="seconds"),
                    sender_account_id=mule_id,
                    sender_persona="AUTONOMO",
                    receiver_account_id=boss_account,
                    receiver_pix_key="boss_central@crypto-offramp.io",
                    receiver_key_age_days=2,
                    amount=round(collected * 0.92, 2),
                    sender_balance_before=round(collected, 2),
                    device_id=f"MULE_DEV_{mule_id}",
                    is_new_device=True,
                    password_reset_mins_ago=None,
                    channel="APP_MOBILE",
                    is_fraud=True,
                    fraud_type="REDE_CONTA_LARANJA",
                )
            )
        )

    # 5. Fraude Tipo 4: Smurfing / Estruturação (rajada de 5 Pix de R$ 980 em 3 minutos)
    for idx in range(8):
        sender_id = rng.choice(account_ids)
        prof = batch.profiles[sender_id]
        t_burst = base_ts + timedelta(hours=10 + idx, minutes=15)
        target_mule = rng.choice(mule_accounts)
        for step in range(5):
            raw_candidates.append(
                asdict(
                    PixEvent(
                        tx_id=f"PIX_SMURF_{idx + 1:02d}_{step + 1}",
                        event_ts=(t_burst + timedelta(seconds=35 * step)).isoformat(timespec="seconds"),
                        sender_account_id=sender_id,
                        sender_persona=prof.persona,
                        receiver_account_id=target_mule,
                        receiver_pix_key=f"chave_{target_mule.lower()}@pix.com",
                        receiver_key_age_days=2,
                        amount=round(rng.uniform(960.0, 998.0), 2),
                        sender_balance_before=6000.0 - step * 980.0,
                        device_id=f"SMURF_DEV_{idx}",
                        is_new_device=True,
                        password_reset_mins_ago=15,
                        channel="APP_MOBILE",
                        is_fraud=True,
                        fraud_type="SMURFING_ESTRUTURACAO",
                    )
                )
            )

    # 6. Injeta eventos malformados para testar Schema Registry & Dead Letter Queue (DLQ)
    raw_candidates.append({"tx_id": "PIX_BAD_001", "event_ts": base_ts.isoformat(), "amount": -50.0})
    raw_candidates.append({"tx_id": "PIX_BAD_002", "event_ts": base_ts.isoformat(), "sender_account_id": ""})

    # Ordena eventos válidos por timestamp e separa DLQ
    for cand in raw_candidates:
        ok, err = validate_avro_contract(cand)
        if ok:
            batch.events.append(cand)
        else:
            batch.dlq_events.append({**cand, "_dlq_reason": err})

    batch.events.sort(key=lambda e: (e["event_ts"], e["tx_id"]))
    return batch
