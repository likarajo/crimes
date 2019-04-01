import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object Crimes {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Crimes")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    // Specify data file
    val filePath = "data/crime_data.csv"

    val crimes = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

    // Get the columns that we need further
    val df = crimes.select("Murder","Assault", "Robbery", "Drugs")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Murder", "Assault", "Robbery", "Drugs"))
      .setOutputCol("features")

    val data = assembler.transform(crimes)

    val kmeans = new KMeans()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setK(4) // set clusters = 4

    val model = kmeans.fit(data)

    val predictions = model.transform(data)
    predictions.select("_c0","crime$cluster", "prediction").show(10)

    // model evaluation
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)

    println("Cluster Centres:")
    model.clusterCenters.foreach(println)

    println("Silhouette: ", silhouette)

    spark.stop()
    println("Disconnected from Spark")

  }

}
