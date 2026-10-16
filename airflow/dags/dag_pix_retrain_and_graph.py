"""DAG do Apache Airflow: Atualização diária do Grafo de Contas Laranja (PageRank) e Retreino Semanal MLflow."""

from __future__ import annotations

from datetime import datetime, timedelta

from pix_fraud_radar.pipeline import PixFraudRadarPlatform

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None  # type: ignore[assignment]
    PythonOperator = None  # type: ignore[assignment]


def run_graph_and_retrain_pipeline() -> dict:
    radar = PixFraudRadarPlatform(n_customers=200, n_legit_tx=1000, seed=42)
    return radar.summary()


if DAG is not None:
    with DAG(
        dag_id="novabank_pix_fraud_graph_and_mlops",
        description="Executa PageRank no grafo de transferências Pix, avalia PSI Drift e publica modelo Champion",
        schedule_interval="0 3 * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args={"owner": "ml-engineering-novabank", "retries": 1, "retry_delay": timedelta(minutes=5)},
        tags=["pix", "fraud", "mlops", "graph", "streaming"],
    ) as dag:
        PythonOperator(
            task_id="compute_graph_pagerank_and_evaluate_champion",
            python_callable=run_graph_and_retrain_pipeline,
        )
