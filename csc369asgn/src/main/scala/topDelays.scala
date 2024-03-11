import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.ml.linalg.SparseVector
import vegas._
import org.apache.spark.sql.SparkSession

object topDelays {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    val spark = SparkSession.builder()
      .appName("Project")
      .master("local[4]")
      .getOrCreate()

    val airlines = spark.sparkContext.textFile("data/airlines.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airlineCode, airlineName) = line.split(",")
        (airlineCode.trim, airlineName.trim)
      })

    val airports = spark.sparkContext.textFile("data/airports.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .map(line => {
        val Array(airportCode, name, _, _, _, _, _) = line.split(",") // Assuming only need airport code and name
        (airportCode, name)
      })

    val flights = spark.sparkContext.textFile("data/flights.csv")
      .mapPartitionsWithIndex { (index, iter) => if (index == 0) iter.drop(1) else iter }
      .filter(_.split(",").length >= 31)
      .map { line =>
        val parts = line.split(",")
        val airline = parts(4).trim
        val originAirport = parts(7).toLowerCase
        val arrivalDelay = parts(22).toDouble
        val cancelled = parts(24).toLowerCase
        (airline, originAirport, arrivalDelay, cancelled)
      }

    val totalDelaysByAirline = flights.filter(_._4 != "1.00")
      .map { case (airline, _, arrivalDelay, _) => (airline, arrivalDelay) }
      .reduceByKey(_ + _)

    val totalFlightsByAirline = flights
      .map { case (airline, _, _, _) => (airline, 1) }
      .reduceByKey(_ + _)

    val averageDelayByAirline = totalDelaysByAirline.join(totalFlightsByAirline)
      .mapValues { case (totalDelay, totalFlights) => (totalDelay / 60) / totalFlights }

    val sortedAirlinesByDelay = averageDelayByAirline.collect().sortBy(_._2)

    sortedAirlinesByDelay.foreach { case (airline, avgDelay) =>
      println(s"Airline: $airline, Average Delay: $avgDelay hours")
    }

    val filteredFlights = flights.filter { case (_, airport, _, _) => airport.forall(_.isLetter) }

    val totalDelaysByAirport = filteredFlights.filter(_._4 != "1.00")
      .map { case (_, airport, arrivalDelay, _) => (airport, arrivalDelay.toDouble) }
      .reduceByKey(_ + _)

    val totalFlightsByAirport = filteredFlights
      .map { case (_, airport, _, _) => (airport, 1) }
      .reduceByKey(_ + _)

    val averageDelayByAirport = totalDelaysByAirport.join(totalFlightsByAirport)
      .mapValues { case (totalDelay, totalFlights) => (totalDelay / 60) / totalFlights }


    val sortedAirportsByDelay = averageDelayByAirport.collect().sortBy(_._2)


    sortedAirportsByDelay.foreach { case (airport, avgDelay) =>
      println(s"Airport: $airport, Average Delay: $avgDelay hours")
    }


    val reversedSortedAirportsByDelay = averageDelayByAirport.collect().sortBy(-_._2)

    val topTenDelayedAirports = reversedSortedAirportsByDelay.take(10)


    val delayedAirportsData = topTenDelayedAirports.map { case (airport, avgDelay) =>
      Map("Airport" -> airport, "Average Delay (hours)" -> avgDelay)
    }

    Vegas("Top Ten Delayed Airports")
      .withData(delayedAirportsData)
      .encodeX("Airport", Nominal)
      .encodeY("Average Delay (hours)", Quantitative) 
      .mark(Bar)
      .show

    val delayedFlightsData = sortedAirlinesByDelay.map { case (airline, avgDelay) =>
      Map("Airline" -> airline, "Average Delay (hours)" -> avgDelay)
    }

    Vegas("Most Delayed Flights")
      .withData(delayedFlightsData)
      .encodeX("Airline", Nominal)
      .encodeY("Average Delay (hours)", Quantitative)
      .mark(Bar)
      .show


