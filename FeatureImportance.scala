import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, BisectingKMeansSummary}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel, KMeansSummary}
import org.apache.spark.ml.classification.RandomForestClassifier


//import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
//import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
//import org.apache.spark.ml.{ Pipeline, PipelineStage }
//import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

///////////////////////////////////////////////////
//establish clustering classes to retrieve cluster outputs on our parsedData stored in a .csv locally.
//will instantiate these classes later with hyperparameters for two clustering algorithms and retrieve
//the kmPredictions and bkmPredictions DataFrame objects with a new column named predictions that stores
//the assigned cluster index [0,1,2,3,4,...)
///////////////////////////////////////////////////
object ClusteringClasses [args: Array[String]]{

  val trainOutputDir = "C:/Users/Documents/SparkClusters/trainOutputLocation"
  //assuming that general data has been collected in parquet format and stored in directory .../trainOutputDir
  val parsedData = spark.read.csv(trainOutputLocation)
  parsedData.registerTempTable("temp1")
  parsedData.printSchema()
  parsedData.show()

  val numClusters = 4
  val numIterations = 125


  class bkmClusterOutputs (data: org.apache.spark.sql.DataFrame, numClusters: Integer = 3 , numIterations: Integer = 20 , numSeed: Long = 1L){
  	println("Generating Cluster Model Outputs")

  	//creating a methodthat returns Tuple2 of cluster centers and the cost function output
  	def bkmDataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer , numIterations: Integer, numSeed: Long ): Tuple4[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double, org.apache.spark.sql.DataFrame] = {
  		//model params
  		val bkm = new org.apache.spark.ml.clustering.BisectingKMeans().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed);
  		//obtain model
  		val bkmModel = bkm.fit(data);
  		//obtain and display cluster Centers
  		val clusterIndexes = bkmModel.clusterCenters;
  		//clusterIndexes.foreach(println)
  		//obtain model predictions
  		val bkmPredictions = bkmModel.transform(data);
  		//obtain cost function output WCSS score for KMeans
  		val WCSS = bkmModel.computeCost(bkmPredictions);
  		//obtain cluster sizes
          val bkmModelSummary = bkmModel.summary;
  		val bkmClusterSizes = bkmModelSummary.clusterSizes;
  		//place cluster centers, cluster sizes, and cost output into Tuple3 collection
  		val bkmoutput: Tuple4[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double, org.apache.spark.sql.DataFrame] = (clusterIndexes, bkmClusterSizes, WCSS, bkmPredictions)
  		return bkmoutput
  	}

  	val model = bkmDataCenters(data, numClusters, numIterations, 1L)
  	//class fields
  	val clusterIndexes = model._1 //Returns Array[org.apache.spark.ml.linalg.Vector] containing cluster centers
  	val clusterSizes  = model._2 //Returns Array[Long] of cluster sizes
  	val cost = model._3 //Returns double representing cost
  	val predictions = model._4 //returns original dataframe with predictions column appended

  }

  class kmClusterOutputs (data: org.apache.spark.sql.DataFrame, numClusters: Integer = 3 , numIterations: Integer = 20 , numSeed: Long = 1L){
  	println("Generating Cluster Model Outputs")

  	//creating a methodthat returns Tuple2 of cluster centers and the cost function output
  	def kmDataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer , numIterations: Integer, numSeed: Long ): Tuple4[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double, org.apache.spark.sql.DataFrame] = {
  		//model params
  		val km = new org.apache.spark.ml.clustering.KMeans().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed);
  		//obtain model
  		val kmModel = km.fit(data);
  		//obtain and display cluster Centers
  		val clusterIndexes = kmModel.clusterCenters;
  		//clusterIndexes.foreach(println)
  		//obtain model predictions
  		val kmPredictions = kmModel.transform(data);
  		//obtain cost function output WCSS score for KMeans
  		val WCSS = kmModel.computeCost(kmPredictions);
  		//obtain cluster sizes
      val kmModelSummary = kmModel.summary;
  		val kmClusterSizes = kmModelSummary.clusterSizes;
  		//place cluster centers, cluster sizes, and cost output into Tuple3 collection
  		val kmoutput: Tuple4[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double, org.apache.spark.sql.DataFrame] = (clusterIndexes, kmClusterSizes, WCSS, kmPredictions)
  		return kmoutput
  	}

  	val model = kmDataCenters(data, numClusters, numIterations, 1L)
  	//class fields
  	val clusterIndexes = model._1 //Returns Array[org.apache.spark.ml.linalg.Vector] containing cluster centers
  	val clusterSizes  = model._2 //Returns Array[Long] of cluster sizes
  	val cost = model._3 //Returns double representing cost
  	val predictions = model._4 //returns original dataframe with predictions column appended

  }

  //Display clusterIndexes
  def bkmShow(parsedData: org.apache.spark.sql.DataFrame, numClusters: Int, numIterations: Int): org.apache.spark.sql.DataFrame = {
    //instantiate Bisecting K Means Model
    val bkmModel = new bkmClusterOutputs(parsedData, numClusters,numIterations)

    val bkmClusterLabels = bkmModel.predictions.withColumnRenamed("prediction","label")

    bkmModel.clusterIndexes.foreach(println)
    //Display cluster sizes
    val a = bkmModel.clusterSizes;
    for((e,count) <- a.zipWithIndex ){
      println(s"Cluster $count contains $e members")
    }
    //Display Cost Function Score
    println("Bisecting Kmeans WCSS = "+bkmModel.cost)

    return bkmClusterLabels
  }

  def kmShow(parsedData: org.apache.spark.sql.DataFrame, numClusters: Int, numIterations: Int): org.apache.spark.sql.DataFrame = {
    val kmModel = new kmClusterOutputs(parsedData, numClusters,numIterations)
    val kmClusterLabels = kmModel.predictions.withColumnRenamed("prediction","label")

    //Display clusterIndexes
    kmModel.clusterIndexes.foreach(println)
    //Display cluster sizes
    val a = kmModel.clusterSizes;
    for((e,count) <- a.zipWithIndex ){
        println(s"Cluster $count contains $e members")
    }
    //Display Cost Function Score
    println("Bisecting Kmeans WCSS = "+kmModel.cost)

    return kmClusterLabels
  }

  //Method to return Random Forest feature importance
  def rfFeatureImportance (clusterLabels:org.apache.spark.sql.DataFrame, maxDepth: Int = 5, numTrees: Int = 20, randomSeed: Int = 5043) {
      //val splitSeed = 5043
      //val Array(trainingData, testData) = bkmClusterLabels.randomSplit(Array(0.80, 0.20), splitSeed)
      val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(maxDepth).setNumTrees(numTrees).setFeatureSubsetStrategy("auto").setSeed(randomSeed)
      val model = classifier.fit(bkmClusterLabels)
      val featureImportanceResults = model.featureImportances //return 'feature importantance vector' from Random Forest Model
      //Recall Features vector contains: "indexedDevice","has_favorites","time_spent","ESPNPlus_Minutes","WatchESPN_Minutes","content_starts","NumberofActiveDays"
      for ( (importance,feature) <- featureImportanceResults.toArray.zipWithIndex){
          println(s"For feature $feature importance is $importance")
      }
  }

  def main(args: Array[String]){


  }


}
