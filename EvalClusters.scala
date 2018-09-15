"label","indexedDevice","time_spent","Video_Minutes","content_starts"

//currently the version of spark on this cluster does not support ml.ClucsteringEvaluator
//import org.apache.spark.ml.evaluation.ClusteringEvaluator //will become available when Spark version 2.3.0 is available on Qubole. Currently experimental library.
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.util.MLUtils._

//Data transformation libraries
import sqlContext.implicits._
import spark.implicits._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}



//////////////////////////////////////////////
//K Means Clustering
import org.apache.spark.ml.clustering.{KMeans, KMeansModel, KMeansSummary}
//class to obtain cluster model outputs
class kmClusterOutputs (data: org.apache.spark.sql.DataFrame, numClusters: Integer = 3 , numIterations: Integer = 20 , numSeed: Long = 1L){
	println("Generating Cluster Model Outputs")

	//creating a methodthat returns Tuple2 of cluster centers and the cost function output
	def kmDataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer , numIterations: Integer, numSeed: Long ): Tuple3[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double] = {
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
		val kmoutput: Tuple3[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double] = (clusterIndexes, kmClusterSizes, WCSS)
		return kmoutput
	}

	val model = kmDataCenters(data, numClusters, numIterations, 1L)
	//class fields
	val clusterIndexes = model._1 //Returns Array[org.apache.spark.ml.linalg.Vector] containing cluster centers
	val clusterSizes  = model._2 //Returns Array[Long] of cluster sizes
	val cost = model._3 //Returns double representing cost

}

//create new kmClusterOutputs instance
var test  = new kmClusterOutputs(data = parsedData, numClusters = 3, numiterations =  30 , numSeed = 1L)
//Display clusterIndexes
test.clusterIndexes.foreach(println)
//Display cluster sizes
val a = test.clusterSizes;
for((e,count) <- a.zipWithIndex ){
    println(s"Cluster $count contains $e members")
}
//Display Cost Function Score
println("Kmeans WCSS = "+test.cost)

//////////////////////////////////////////////
//Bisecting K Means Clustering
import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, BisectingKMeansSummary}

//class to obtain cluster model outputs
class bkmClusterOutputs (data: org.apache.spark.sql.DataFrame, numClusters: Integer = 3 , numIterations: Integer = 20 , numSeed: Long = 1L){
	println("Generating Cluster Model Outputs")

	//creating a methodthat returns Tuple2 of cluster centers and the cost function output
	def bkmDataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer , numIterations: Integer, numSeed: Long ): Tuple3[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double] = {
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
		val bkmoutput: Tuple3[Array[org.apache.spark.ml.linalg.Vector],Array[Long],Double] = (clusterIndexes, bkmClusterSizes, WCSS)
		return bkmoutput
	}

	val model = bkmDataCenters(data, numClusters, numIterations, 1L)
	//class fields
	val clusterIndexes = model._1 //Returns Array[org.apache.spark.ml.linalg.Vector] containing cluster centers
	val clusterSizes  = model._2 //Returns Array[Long] of cluster sizes
	val cost = model._3 //Returns double representing cost

}

//create new bkmClusterOutputs instance
var bkmTest  = new bkmClusterOutputs(data = parsedData, numClusters = 3, numIterations =  30 , numSeed = 1L)
//Display clusterIndexes
bkmTest.clusterIndexes.foreach(println)
//Display cluster sizes
val a = bkmTest.clusterSizes;
for((e,count) <- a.zipWithIndex ){
    println(s"Cluster $count contains $e members")
}
//Display Cost Function Score
println("Bisecting Kmeans WCSS = "+bkmtest.cost)




//////////////////////////////////////////////
//Gaussian Mixture Clustering
 import org.apache.spark.ml.clustering.{GaussianMixture, GaussianMixtureModel, GaussianMixtureSummary}

//class to obtain cluster model outputs
class gmClusterOutputs (data: org.apache.spark.sql.DataFrame, numClusters: Integer = 3 , numIterations: Integer = 20 , numSeed: Long = 1L){
	println("Generating Cluster Model Outputs")

	//creating a methodthat returns Tuple2 of cluster centers and the cost function output
	def gmDataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer , numIterations: Integer, numSeed: Long ): Tuple3[Array[org.apache.spark.ml.stat.distribution.MultivariateGaussian],Array[Long],Double] = {
		//model params
		val gm = new org.apache.spark.ml.clustering.GaussianMixture().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed);
		//obtain model
		val gmModel = gm.fit(data);
		//obtain and display cluster Centers
		val clusterIndexes = gmModel.gaussians;
		//clusterIndexes.foreach(println)
		//obtain model predictions
		//val gmPredictions = gmModel.transform(data);
		//obtain cluster sizes
    val gmModelSummary = gmModel.summary;
		val gmClusterSizes = gmModelSummary.clusterSizes;
		//obtain total log likelihood output for Gaussian Mixture Model ~= cost function
		val LL = gmModelSummary.logLikelihood;
		//place cluster centers, cluster sizes, and LL output into Tuple3 collection
		val gmoutput: Tuple3[Array[org.apache.spark.ml.stat.distribution.MultivariateGaussian],Array[Long],Double] = (clusterIndexes, gmClusterSizes, LL)
		return gmoutput
	}

	val model = gmDataCenters(data, numClusters, numIterations, 1L)
	//class fields
	val clusterIndexes = model._1 //Returns Array[org.apache.spark.ml.stat.distribution.MultivariateGaussian].
	val clusterSizes  = model._2 //Returns Array[Long] of cluster sizes
	val cost = model._3 //Returns double representing cost

}

//create new gmClusterOutputs instance
var gmtest  = new gmClusterOutputs(data = parsedData, numClusters = 3, numIterations =  20 , numSeed = 1L)
//Display clusterIndexes
gmtest.clusterIndexes.foreach(println)
//Display cluster sizes
val a = gmtest.clusterSizes;
for((e,count) <- a.zipWithIndex ){
    println(s"Cluster $count contains $e members")
}
//Display Cost Function Score
println("Gaussian Mixture Log Likelihood = "+gmtest.cost)


//With model object  can output specific parts of Gaussian distribution
val gm = new org.apache.spark.ml.clustering.GaussianMixture().setK(3).setMaxIter(20).setSeed(1l);
val gmModel = gm.fit(parsedData);

// output parameters of mixture model model
    for (i <- 0 until gmModel.getK) {
      println(s"Gaussian $i:\nweight=${gmModel.weights(i)}\n" +
          s"mu=${gmModel.gaussians(i).mean}\nsigma=\n${gmModel.gaussians(i).cov}\n")
    }


// Save and load model
clusters.save(sc, "")
val sameModel = KMeansModel.load(sc, "")
