import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{Encoder, Encoders, Row, SparkSession}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import vegas._



object dataCleaning {
  
    def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder()
      .appName("Project")
      .master("local[4]")
      .getOrCreate()

    val flightsData = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv("data/flights.csv")
      .filter("MONTH = 6")

    val indexer = new StringIndexer()
      .setInputCol("AIRLINE")
      .setOutputCol("AIRLINE_INDEX")

    val indexedData = indexer.fit(flightsData).transform(flightsData)

    println("Airline Index Mapping:")
    indexedData.select("AIRLINE", "AIRLINE_INDEX").distinct().show()


    val encoder = new OneHotEncoder()
      .setInputCol("AIRLINE_INDEX")
      .setOutputCol("AIRLINE_ONE_HOT")

    val encodedData = encoder.transform(indexedData)

    val airlineOneHotValues = encodedData.select("AIRLINE_ONE_HOT").distinct().collect()

    airlineOneHotValues.foreach { row =>
      println(s"Airline Encoding: ${row.getAs[SparseVector](0)}")
    }

    val assembler = new VectorAssembler()
      .setInputCols(Array("DAY", "DAY_OF_WEEK", "AIRLINE_ONE_HOT", "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE", "SCHEDULED_ARRIVAL"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val assembledData = assembler.transform(encodedData).select("features", "ARRIVAL_DELAY").na.drop()
    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2))

    val lr = new LinearRegression()
      .setLabelCol("ARRIVAL_DELAY")
      .setFeaturesCol("features")

    val lrModel = lr.fit(trainingData)


    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

    val predictions = lrModel.transform(testData)
    val predictedActualDF = predictions.select("prediction", "ARRIVAL_DELAY").toDF("predicted", "actual")
    val predictedActualData = predictedActualDF.collect()
    val predictedActualDataSeq: Seq[Map[String, Any]] = predictedActualData.map(row =>
      Map("predicted" -> row.getAs[Double](0), "actual" -> row.getAs[Double](1))
    )

    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

    println("Model Summary:")
    println(lrModel.summary.toString)

    println("Objective History:")
    println(lrModel.summary.objectiveHistory.mkString("Array(", ", ", ")"))

    println(s"Number of Iterations: ${lrModel.summary.totalIterations}")

    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")

    println(s"MSE: ${lrModel.summary.meanSquaredError}")

    println(s"MAE: ${lrModel.summary.meanAbsoluteError}")

    println(s"Explained Variance: ${lrModel.summary.explainedVariance}")

    println(s"R^2: ${lrModel.summary.r2}")

    println("Predictions DataFrame:")
    lrModel.summary.predictions.show()

    val featureNames = assembler.getInputCols

    val coefficientsWithDescription = featureNames.zip(lrModel.coefficients.toArray)

    coefficientsWithDescription.foreach { case (featureName, coefficient) =>
      println(s"Coefficient for $featureName: $coefficient")
    }

    val plot = Vegas("Predicted vs Actual")
      .withData(predictedActualDataSeq)
      .mark(Point)
      .encodeX("predicted", Quantitative)
      .encodeY("actual", Quantitative)



    def render = {
      plot.window.show
    }

    render
  }

}
