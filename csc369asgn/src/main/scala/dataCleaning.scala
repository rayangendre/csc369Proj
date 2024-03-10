import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import vegas._


object dataCleaning {
  
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder()
      .appName("Project")
      .master("local[4]")
      .getOrCreate()

    // Define case classes for airlines and airports
    case class Airline(airlineCode: String, airlineName: String)
    case class Airport(airportCode: String, name: String)




    // Load airlines and airports data
    val airlines = spark.sparkContext.textFile("src/main/data/airlines.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airlineCode, airlineName) = line.split(",")
        Airline(airlineCode.trim, airlineName.trim)
      })

    val airports = spark.sparkContext.textFile("src/main/data/airports.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airportCode, name, _, _, _, _, _) = line.split(",") // Assuming only need airport code and name
         Airport(airportCode, name)
      })

    // Join flights with airlines and airports
    val flights = spark.sparkContext.textFile("src/main/data/flights.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .filter(_.split(",").length >= 31)
      .map { line =>
          val parts = line.split(",")
          val airline = parts(4).trim
          val originAirport = parts(7).toLowerCase
          val destinationAirport = parts(8).toLowerCase
          val year = parts(0).toLowerCase
          val month = parts(1).toLowerCase
          val day = parts(2).toLowerCase
          val dayOfWeek = parts(3).toLowerCase
          val flightNumber = parts(6).toLowerCase
          //val tailNumber = parts(6).toLowerCase
          val scheduledDeparture = parts(9).toLowerCase
          val departureTime = parts(10).toLowerCase
          //val departureDelay = parts(11).toLowerCase
          //val taxiOut = parts(12).toLowerCase
          //val wheelsOff = parts(13).toLowerCase
          val scheduledTime = parts(14).toLowerCase
          //val elapsedTime = parts(15).toLowerCase
          val airTime = parts(16).toLowerCase
          val distance = parts(17).toLowerCase
          //val wheelsOn = parts(18).toLowerCase
          //val taxiIn = parts(19).toLowerCase
          val scheduledArrival = parts(20).toLowerCase
          //val arrivalTime = parts(21).toLowerCase
          val arrivalDelay = parts(22).toLowerCase
          //val diverted = parts(23).toLowerCase
          val cancelled = parts(24).toLowerCase
          val cancellationReason = parts(25).toLowerCase
          val airSystemDelay = parts(26).toLowerCase
          val securityDelay = parts(27).toLowerCase
          val airlineDelay = parts(28).toLowerCase
          val lateAircraftDelay = parts(29).toLowerCase
          val weatherDelay = parts(30).toLowerCase

          //Some((year, month, day, dayOfWeek, airline, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime,  scheduledTime,  airTime, distance, scheduledArrival,  arrivalDelay,  cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay))
          (year, month, day, dayOfWeek, airline, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime,  scheduledTime,  airTime, distance, scheduledArrival,  arrivalDelay,  cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)
      }

    // ex: delay by airline, month, dayofWeek
    val airlineDelayRDD = flights.map{
      case (year, month, day, dayOfWeek, airline, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime,  scheduledTime,  airTime, distance, scheduledArrival,  arrivalDelay,  cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)
      =>
        ((airline, month, dayOfWeek), arrivalDelay.toInt)
    }.groupByKey()
      .mapValues(values => values.sum / values.size.toDouble)

    airlineDelayRDD.sortByKey().collect().foreach(println)



//    try using Vega on Linear Regression Model
    val flightsData = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/data/flights.csv")
      .limit(300)

    val assembler = new VectorAssembler()
      .setInputCols(Array("MONTH", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE",
        "DEPARTURE_TIME", "SCHEDULED_TIME", "AIR_TIME",
      "DISTANCE", "SCHEDULED_ARRIVAL"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val assembledData = assembler.transform(flightsData).select("features", "ARRIVAL_DELAY").na.drop()
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
