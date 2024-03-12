import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{Encoder, Encoders, Row, SparkSession}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import vegas._
import org.apache.spark.sql.catalyst.encoders.RowEncoder

object UseKNN {

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
      .sample(false, 0.1)
      .limit(15000)

    val airlineIndexer = new StringIndexer()
      .setInputCol("AIRLINE")
      .setOutputCol("AIRLINE_INDEX")

    val originAirportIndexer = new StringIndexer()
      .setInputCol("ORIGIN_AIRPORT")
      .setOutputCol("ORIGIN_AIRPORT_INDEX")

    val destinationAirportIndexer = new StringIndexer()
      .setInputCol("DESTINATION_AIRPORT")
      .setOutputCol("DESTINATION_AIRPORT_INDEX")

    val indexedData = airlineIndexer.fit(flightsData)
      .transform(originAirportIndexer.fit(flightsData)
        .transform(destinationAirportIndexer.fit(flightsData)
          .transform(flightsData)))


    indexedData.select("AIRLINE", "AIRLINE_INDEX").distinct().show()


    val assembler = new VectorAssembler()
      .setInputCols(Array("AIRLINE_INDEX", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE",
        "ORIGIN_AIRPORT_INDEX", "SCHEDULED_TIME",
        "DISTANCE", "SCHEDULED_ARRIVAL", "DESTINATION_AIRPORT_INDEX"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val assembledData = assembler.transform(indexedData).select("features", "ARRIVAL_DELAY").na.drop()
    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.8, 0.2))

    //    convert training features into array for KNN
    val trainFeatures: Array[Row] = trainingData.select("features").collect()
    val trainFeaturesArrays: Array[Array[Double]] = trainFeatures.map { row =>
      val denseVector: DenseVector = row.getAs[DenseVector](0)
      denseVector.toArray
    }

    // convert training labels (ARRIVAL_DELAY) into array for KNN
    // 1 = delayed (if greater than threshold), else 0 = on time
    val trainLabels: Array[Row] = trainingData.select("ARRIVAL_DELAY").collect()
    val trainLabelsArray: Array[Int] = trainLabels.map { row =>
      val threshold = 15.0
      val delay = row.getAs[Integer](0)
      if (delay > threshold) {
        1 // delayed
      } else {
        0 // on time
      }
    }

    //    convert testing features into array for KNN
    val testFeatures: Array[Row] = testData.select("features").collect()
    val testFeaturesArrays: Array[Array[Double]] = testFeatures.map { row =>
      val denseVector: DenseVector = row.getAs[DenseVector](0)
      denseVector.toArray
    }

    val testLabels: Array[Row] = testData.select("ARRIVAL_DELAY").collect()
    val testLabelsArray: Array[Int] = testLabels.map { row =>
      val threshold = 15.0
      val delay = row.getAs[Integer](0)
      if (delay > threshold) {
        1 // delayed
      } else {
        0 // on time
      }
    }

    val totalTestInstances = testFeaturesArrays.length
    var correctPredictions = 0

    val knn = new KNN(2)
    knn.train(trainFeaturesArrays, trainLabelsArray)

    println(testFeaturesArrays.length)

    testFeaturesArrays.zip(testLabelsArray).foreach { case (testFeature, testLabel) =>
      val predictedLabel = knn.predict(testFeature)
      if (predictedLabel == testLabel) {
        correctPredictions += 1
      }
    }

    val accuracy = correctPredictions.toDouble / totalTestInstances
    val error = 1 - accuracy

    println(s"Accuracy: $accuracy")
    println(s"Error rate: $error")



  }

  def seqToVector(seq: Seq[Double]): Vector = Vectors.dense(seq.toArray)


}
