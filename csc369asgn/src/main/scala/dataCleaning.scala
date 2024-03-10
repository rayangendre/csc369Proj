import org.apache.spark.sql.SparkSession

object dataCleaning {

  def main(args: Array[String]): Unit = {

    import org.apache.spark.sql.SparkSession

    val spark = SparkSession.builder()
      .appName("Project")
      .master("local[4]")
      .getOrCreate()

    // Define case classes for airlines and airports
    case class Airline(airlineCode: String, airlineName: String)
    case class Airport(airportCode: String, name: String)


    val airports = spark.sparkContext.textFile("data/airports.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airportCode, name, _, _, _, _, _) = line.split(",") // Assuming only need airport code and name
        Airport(airportCode, name)
      })


    // Join flights with airlines and airports
    val flights = spark.sparkContext.textFile("data/flights.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .flatMap { line =>
        val parts = line.split(",")
        if (parts.length >= 31) {
          val airlineCode = parts(4).toLowerCase
          val originAirport = parts(7).toLowerCase
          val destinationAirport = parts(8).toLowerCase
          val year = parts(1).toLowerCase
          val month = parts(2).toLowerCase
          val day = parts(3).toLowerCase
          val dayOfWeek = parts(0).toLowerCase
          val flightNumber = parts(5).toLowerCase
          val scheduledDeparture = parts(9).toLowerCase
          val departureTime = parts(10).toLowerCase
          val scheduledTime = parts(14).toLowerCase
          val airTime = parts(16).toLowerCase
          val distance = parts(17).toLowerCase
          val scheduledArrival = parts(20).toLowerCase
          val arrivalDelay = parts(22).toLowerCase
          val cancelled = parts(24).toLowerCase
          val cancellationReason = parts(25).toLowerCase
          val airSystemDelay = parts(26).toLowerCase
          val securityDelay = parts(27).toLowerCase
          val airlineDelay = parts(28).toLowerCase
          val lateAircraftDelay = parts(29).toLowerCase
          val weatherDelay = parts(30).toLowerCase

          Some((airlineCode, (year, month, day, dayOfWeek, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime, scheduledTime, airTime, distance, scheduledArrival, arrivalDelay, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)))
        } else {
          None
        }
      }

    // Join flights with airports for origin and destination
    val flightsWithAirports = flights
      .flatMap { case (airlineCode, (year, month, day, dayOfWeek, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime, scheduledTime, airTime, distance, scheduledArrival, arrivalDelay, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)) =>
        // Make sure both origin and destination airports are available
        if (originAirport.nonEmpty && destinationAirport.nonEmpty) {
          Some((originAirport, (airlineCode, year, month, day, dayOfWeek, flightNumber, scheduledDeparture, departureTime, scheduledTime, airTime, distance, scheduledArrival, arrivalDelay, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)),
            (destinationAirport, (airlineCode, year, month, day, dayOfWeek, flightNumber, scheduledDeparture, departureTime, scheduledTime, airTime, distance, scheduledArrival, arrivalDelay, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)))
        } else {
          None
        }
      }

    // Join flightsWithAirports with airports
    val joinedDataSet = flightsWithAirports.join(airports)

    // Output the result
    joinedDataSet.take(10).foreach { case (airportCode, ((airlineCode, year, month, day, dayOfWeek, flightNumber, scheduledDeparture, departureTime, scheduledTime, airTime, distance, scheduledArrival, arrivalDelay, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay), airportName)) =>
      println(s"Airport: $airportCode, Airline: $airlineCode, Airport Name: $airportName")
    }

    // Convert airlines RDD
    val airlines = spark.sparkContext.textFile("data/airlines.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airlineCode, airlineName) = line.split(",")
        (airlineCode.trim.toLowerCase, airlineName.trim.toLowerCase)
      })

    val joinedRDD = flights.join(airlines)

    val totalDelayByAirline = flights
      .map { case (airlineCode, (_, _, _, _, _, _, _, _, _, _, _, _, _, arrivalDelay,_, cancelled, _, _, _, _, _)) =>
        (airlineCode, arrivalDelay.toDouble)
      }
      .reduceByKey(_ + _)

    val totalDelayByAirports = flights
      .map { case (airlineCode, (_, _, _, _, _, originAirport, _, _, _, _, _, _, _, arrivalDelay, _, cancelled, _, _, _, _, _)) =>
        (originAirport, arrivalDelay.toDouble)
      }
      .reduceByKey(_ + _)


    // Join with airlines to get airline names
    val totalDelayByAirlineWithNames = totalDelayByAirline.join(airlines)

    // Displaying the results
    totalDelayByAirlineWithNames.collect().foreach { case (airlineCode, (totalDelay, airlineName)) =>
      val hours = totalDelay / 60.0
      println(s"Airline: $airlineName, Total Delay: $hours hours")
    }

    totalDelayByAirports.collect().foreach { case (airportCode, totalDelay) =>
      val hours = totalDelay / 60.0
      println(s"Airport: $airportCode, Total Delay: $hours hours")
    }


  }


}
