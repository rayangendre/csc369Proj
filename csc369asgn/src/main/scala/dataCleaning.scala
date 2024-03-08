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

    case class Flight(year: String,
                      month: String,
                      day: String,
                      dayOfWeek: String,
                      airline: String,
                      flightNumber: String,
                      tailNumber: String,
                      originAirport: String,
                      destinationAirport: String,
                      scheduledDeparture: String,
                      departureTime: String,
                      departureDelay: String,
                      taxiOut: String,
                      wheelsOff: String,
                      scheduledTime: String,
                      elapsedTime: String,
                      airTime: String,
                      distance: String,
                      wheelsOn: String,
                      taxiIn: String,
                      scheduledArrival: String,
                      arrivalTime: String,
                      arrivalDelay: String,
                      diverted: String,
                      cancelled: String,
                      cancellationReason: String,
                      airSystemDelay: String,
                      securityDelay: String,
                      airlineDelay: String,
                      lateAircraftDelay: String,
                      weatherDelay: String)


    // Load airlines and airports data
    val airlines = spark.sparkContext.textFile("data/airlines.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airlineCode, airlineName) = line.split(",")
        Airline(airlineCode.trim, airlineName.trim)
      })

    val airports = spark.sparkContext.textFile("data/airports.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airportCode, name, _, _, _, _, _) = line.split(",") // Assuming only need airport code and name
        Airport(airportCode, name)
      })

    // Join flights with airlines and airports
    val flights = spark.sparkContext.textFile("data/flights.txt")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(year, month, day, dayOfWeek, airline, flightNumber, tailNumber, originAirport, destinationAirport, scheduledDeparture, departureTime, departureDelay, taxiOut, wheelsOff, scheduledTime, elapsedTime, airTime, distance, wheelsOn, taxiIn, scheduledArrival, arrivalTime, arrivalDelay, diverted, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay) = line.split(",")
        Flight(year, month, day, dayOfWeek, airline.toLowerCase, flightNumber, tailNumber, originAirport.toLowerCase, destinationAirport.toLowerCase, scheduledDeparture, departureTime, departureDelay, taxiOut, wheelsOff, scheduledTime, elapsedTime, airTime, distance, wheelsOn, taxiIn, scheduledArrival, arrivalTime, arrivalDelay, diverted, cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay)
      })

    val flightsWithAirlineInfo = flights.map(flight => (flight.airline, flight)).join(airlines.map(airline => (airline.airlineCode, airline))).map {
      case (_, (flight, airline)) => (flight, airline)
    }

    val flightsWithAirportInfo = flightsWithAirlineInfo.map(flightWithAirline => (flightWithAirline._1.originAirport, flightWithAirline)).join(airports.map(airport => (airport.airportCode, airport))).map {
      case (_, (flightWithAirline, airport)) => (flightWithAirline._1, flightWithAirline._2, airport)
    }

    // Now flightsWithAirportInfo contains flight data joined with both airline and airport information


  }
}
