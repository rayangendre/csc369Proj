import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import vegas.{Point, Quantitative, Vegas}

import scala.util.Random

object manualLinearRegression {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder()
      .appName("Project")
      .master("local[4]")
      .getOrCreate()

    val flightsData = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/data/flights.csv")
      .filter("MONTH = 6")

    // ((features), target)
    val data = flightsData.select("DAY", "DAY_OF_WEEK", "MONTH", "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY")
      .rdd
      .filter(row => !row.anyNull)
      .map(row => (List(row.getInt(0).toDouble, row.getInt(1).toDouble, row.getInt(2).toDouble,
        row.getInt(3).toDouble, row.getInt(4).toDouble,row.getInt(5).toDouble,row.getInt(6).toDouble),
        row.getInt(7).toDouble))

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), seed = Random.nextLong())

    val n = trainData.count()
    val (meanFeatures, meanTarget) = data
      .map { case (x, y) => (x, y) }
      .reduce { case ((x1, y1), (x2, y2)) =>
      (x1.zip(x2).map { case (xi1, xi2) => xi1 + xi2 }, y1 + y2)
    }
    val meanFeatureArr = meanFeatures.map(x => x/n)
    val meanTargetRes = meanTarget / n

    val beta = calculateCoefficients(data, meanFeatureArr, meanTargetRes)
      .map(value => if (value.isNaN) 0.0 else value)

    println(beta)

    val predictionsVsActual = testData.map { case (x, actual) =>
      val prediction = beta(0) + x.zip(meanFeatureArr).zip(beta.drop(1)).map { case ((xi, meanXi), betaI) => (xi - meanXi) * betaI }.sum
      (prediction, actual)
    }.filter { case (prediction, actual) =>
      prediction != null && actual != null
    }

    val visData = predictionsVsActual.collect().map { case (value1, value2) =>
      Map("predicted" -> value1, "actual" -> value2)
    }

//    visData.foreach{println}

    val plot = Vegas("Predicted vs Actual")
      .withData(visData)
      .mark(Point)
      .encodeX("predicted", Quantitative)
      .encodeY("actual", Quantitative)



    def render = {
      plot.window.show
    }

    render

  }

  def calculateCoefficients(data: org.apache.spark.rdd.RDD[(List[Double], Double)], meanX: List[Double], meanY: Double): List[Double] = {
    val numerator = data.map { case (x, y) =>
      val xCentered = x.zip(meanX).map { case (xi, meanXi) => xi - meanXi }
      val xy = xCentered.zipWithIndex.map { case (xCentered, i) => xCentered * (y - meanY) }
      xy
    }.reduce((a, b) => a.zip(b).map { case (x1, x2) => x1 + x2 })

    val denominator = data.map { case (x, _) =>
      x.zip(meanX).map { case (xi, meanXi) => math.pow(xi - meanXi, 2) }
    }.reduce((a, b) => a.zip(b).map { case (x1, x2) => x1 + x2 })

    val beta = numerator.zip(denominator).map { case (num, denom) => num / denom }
    val beta0 = meanY - (meanX.zip(beta).map { case (meanXi, betaI) => meanXi * betaI }).sum
    beta0 +: beta
  }
}
