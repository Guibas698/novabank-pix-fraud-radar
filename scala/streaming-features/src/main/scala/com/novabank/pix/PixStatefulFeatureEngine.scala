package com.novabank.pix

import java.sql.Timestamp
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.{GroupState, GroupStateTimeout, OutputMode}

/**
 * Motor Stateful em Scala Spark Structured Streaming para engenharia de features Pix em tempo real.
 *
 * Utiliza `flatMapGroupsWithState` com Watermark para manter o estado de cada conta pagadora
 * em janelas móveis de 5 minutos e 1 hora, lidando com eventos atrasados (late-arriving events)
 * e publicando o vetor de features enriquecido no tópico Kafka/Redpanda e no Delta Lake.
 */
object PixStatefulFeatureEngine {

  case class PixRawEvent(
    tx_id: String,
    event_ts: Timestamp,
    sender_account_id: String,
    receiver_account_id: String,
    amount: Double,
    sender_balance_before: Double,
    is_new_device: Boolean,
    receiver_key_age_days: Int
  )

  case class SenderAccountState(
    recentTimestampsMs: List[Long],
    recentAmounts: List[Double],
    historicalCount: Long,
    historicalSum: Double
  )

  case class EnrichedPixFeatures(
    tx_id: String,
    sender_account_id: String,
    receiver_account_id: String,
    amount: Double,
    tx_count_5m: Int,
    tx_amount_sum_1h: Double,
    amount_to_user_avg_ratio: Double,
    balance_drain_ratio: Double
  )

  def updateSenderState(
    senderId: String,
    events: Iterator[PixRawEvent],
    state: GroupState[SenderAccountState]
  ): Iterator[EnrichedPixFeatures] = {
    val sortedEvents = events.toList.sortBy(_.event_ts.getTime)
    var current = state.getOption.getOrElse(SenderAccountState(Nil, Nil, 0L, 0.0))

    val results = sortedEvents.map { ev =>
      val nowMs = ev.event_ts.getTime
      val cutoff1h = nowMs - 3600L * 1000L
      val cutoff5m = nowMs - 300L * 1000L

      val zipped = current.recentTimestampsMs.zip(current.recentAmounts).filter(_._1 >= cutoff1h)
      val count5m = zipped.count(_._1 >= cutoff5m)
      val sum1h = zipped.map(_._2).sum
      val avgHist = if (current.historicalCount > 0) current.historicalSum / current.historicalCount else 180.0
      val ratioAvg = ev.amount / math.max(10.0, avgHist)
      val drainRatio = ev.amount / math.max(1.0, ev.sender_balance_before)

      current = SenderAccountState(
        recentTimestampsMs = zipped.map(_._1) :+ nowMs,
        recentAmounts = zipped.map(_._2) :+ ev.amount,
        historicalCount = current.historicalCount + 1L,
        historicalSum = current.historicalSum + ev.amount
      )
      state.update(current)
      state.setTimeoutDuration("2 hours")

      EnrichedPixFeatures(
        tx_id = ev.tx_id,
        sender_account_id = senderId,
        receiver_account_id = ev.receiver_account_id,
        amount = ev.amount,
        tx_count_5m = count5m,
        tx_amount_sum_1h = sum1h,
        amount_to_user_avg_ratio = ratioAvg,
        balance_drain_ratio = drainRatio
      )
    }
    results.iterator
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("NovaBank-Pix-Stateful-Streaming-Features")
      .getOrCreate()

    import spark.implicits._

    val stream = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", sys.env.getOrElse("KAFKA_BROKERS", "redpanda:9092"))
      .option("subscribe", "pix.transactions.raw")
      .load()
      .selectExpr("CAST(value AS STRING) as json_payload")
      .as[String]
      .mapPartitions(_ => Iterator.empty[PixRawEvent]) // Deserialização Avro/JSON em produção

    val enriched = stream
      .withWatermark("event_ts", "10 minutes")
      .groupByKey(_.sender_account_id)
      .flatMapGroupsWithState(
        OutputMode.Append(),
        GroupStateTimeout.ProcessingTimeTimeout()
      )(updateSenderState)

    val query = enriched.writeStream
      .format("delta")
      .option("checkpointLocation", "/tmp/checkpoints/pix_features")
      .start("s3a://lakehouse/silver/pix_streaming_features")

    query.awaitTermination()
  }
}
