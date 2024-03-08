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
      .flatMap { line =>
        val parts = line.split(",")
        if (parts.length >= 31) {
          val airline = parts(4).toLowerCase
          val originAirport = parts(7).toLowerCase
          val destinationAirport = parts(8).toLowerCase
          val year = parts(1).toLowerCase
          val month = parts(2).toLowerCase
          val day = parts(3).toLowerCase
          val dayOfWeek = parts(0).toLowerCase
          val flightNumber = parts(5).toLowerCase
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

          Some((year, month, day, dayOfWeek, airline, flightNumber, originAirport, destinationAirport, scheduledDeparture, departureTime,  scheduledTime,  airTime, distance, scheduledArrival,  arrivalDelay,  cancelled, cancellationReason, airSystemDelay, securityDelay, airlineDelay, lateAircraftDelay, weatherDelay))
        } else {
          None
        }
      }


    flights.collect().foreach(println(_))




  }


}
