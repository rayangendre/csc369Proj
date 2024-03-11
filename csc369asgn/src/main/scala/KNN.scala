class KNN(k: Int) {

  // Load training data (features and labels)
  var trainingData: (Array[Array[Double]], Array[Int]) = null

  def train(features: Array[Array[Double]], labels: Array[Int]): Unit = {
    trainingData = (features, labels)
  }

  def predict(point: Array[Double]): Int = {
    // Calculate distances to all training points
    val distances = trainingData._1.map(p => (euclideanDistance(point, p), p))

    // Sort by distances (ascending order) and get the k nearest neighbors
    val nearestNeighbors = distances.sortBy(_._1).take(k)

    // Get the majority vote label from neighbors
    val neighborLabels = nearestNeighbors.map { case (distance, neighborPoint) =>
      // Find the index of the neighborPoint in the training data features
      val labelIndex = trainingData._1.indexOf(neighborPoint)
      // Access the corresponding label using the index
      trainingData._2(labelIndex)
    }

    val labelCounts = neighborLabels.groupBy(identity).mapValues(_.length)
    labelCounts.maxBy(_._2)._1
  }

  def euclideanDistance(point1: Array[Double], point2: Array[Double]): Double = {
    (point1 zip point2).map { case (x1, x2) =>
      math.pow(x1 - x2, 2)
    }.sum
  }
}
