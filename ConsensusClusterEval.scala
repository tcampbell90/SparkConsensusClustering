import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}
import org.apache.spark.ml.clustering._
import org.apache.spark.sql._
import scala.collection.mutable.{Map,HashMap,HashSet}
import scala.collection.immutable.{Set}
import scala.util.control.Breaks._
import scala.util.Random._

abstract class ConsensusCluster {
    //Used instantiate a new Breaks into Consensus Cluster class. Breaks are used in place of continue key word from Java to exit for{} structures
    val mybreaks = new scala.util.control.Breaks
    import mybreaks.{break, breakable}

    //Num clusters to construct as chosen by previous clustering evaluation methods
    val numClusters = 5;

    // # iterations to determine clusters
    val numIterations = 125;

    //Random seed number (stored in Long format) for ml lib K Means algorithm
    val numSeed = 1L

    //Param to control how many loops the consensus cluster algorithm will run
    val numRuns = 20

    // Stores how many times two data points were assigned to the same cluster
    // across all iterations.
    //
    // A data point is represented as a Row object. Key to the HashMap is a data point
    // (outer point) and the corresponding value is a HashMap that maps a data point
    // (inner point) to an integer. Hence, to find how many times two data points were
    // assigned to the same cluster across all iterations pass one of those data points
    // (outer point) to countSameCluster to get the corresponding HashMap. Then, pass
    // the second data point(inner point) to that HashMap to get the integer value,
    // which is how many times those data points were assigned to the same cluster.
    var countSameCluster = new scala.collection.mutable.HashMap[String, scala.collection.mutable.HashMap[String, Integer]];

    // Stores how many times two data points were assigned to the same data sample
    // across all iterations.

    var countSameSample = new scala.collection.mutable.HashMap[String, scala.collection.mutable.HashMap[String, Integer]];

    //Create histogram class to store counts of percentage that point pairs ended up in same cluster
    class Histogram extends scala.collection.mutable.HashMap {
    var histogram = new scala.collection.mutable.HashMap[Integer, Integer]
    // Initialize with all 0's.
    histogram.put(0, 0);
    histogram.put(1, 0);
    histogram.put(2, 0);
    histogram.put(3, 0);
    histogram.put(4, 0);
    histogram.put(5, 0);
    histogram.put(6, 0);
    histogram.put(7, 0);
    histogram.put(8, 0);
    histogram.put(9, 0);
    }

    // * Initialize 'countSameCluster' & 'countSameCluster' from full data set
    // *
    // * param fullData: full data set

    def initializeCounts(fullData: org.apache.spark.sql.DataFrame){
        val fullDataRDD = fullData.rdd.map(_.mkString(","));
        val pointsOuter = fullDataRDD.collect();
        val pointsInner = fullDataRDD.collect();

        for (pointOuter <- pointsOuter){
          var map = new scala.collection.mutable.HashMap[String, Integer];
          for (pointInner <- pointsInner){
            map.put(pointInner, 0);
          }
          countSameCluster.put(pointOuter, map);
        }

        for (pointOuter <- pointsOuter){
          var map = new scala.collection.mutable.HashMap[String, Integer];
          for (pointInner <- pointsInner){
            map.put(pointInner, 0);
          }
          countSameSample.put(pointOuter, map);
        }
    }

    // Given a subsample update the 'countSameSample' data structure.
    // param collectedSampledData - Data points in a subsample
    def updateSameSampleCounts(collectedSampledData: Array[String]) {
        val clonedData = collectedSampledData
        var allPoints = new scala.collection.mutable.HashMap[String, scala.collection.mutable.HashMap[String, Integer]];

        for ( s <- collectedSampledData) {
          // Get all the points in 'countSameSample' for the particular
          // data point s in subsample
          var tempMap = countSameSample.get(s).get //second get in get(s).get is to get the value from the type Option[HashMap[Integer,String]]
          for ( c <- clonedData) {
              breakable {
                  if(s == c) {
                      break() // ignore self
                  }
                  else {
                      // Increment # times c & s were together in a subsample
                      var currentVal = tempMap.get(c).get;
                      currentVal += 1;
                      tempMap.update(c, currentVal);
                  }
              }
          }
          allPoints.update(s, tempMap)
        }

      for ( (key,value) <- allPoints) {
        countSameSample.update(key,value)
      }
    }

  // * Given a cluster update 'countSameCluster' data structure
  // * from data points in a cluster
  // * @param clusterPoints - Data points in a cluster

  def updateSameClusterCounts(clusterPoints: scala.collection.mutable.HashMap[Integer, scala.collection.mutable.Set[String]] ) {
    //Each element in keys corresponds to a particular cluster, 0, 1, 2
    val keys = clusterPoints.keySet;
    var allPoints = new scala.collection.mutable.HashMap[String, scala.collection.mutable.HashMap[String, Integer]];

    for ( i <- keys) {
        // Obtain points in that cluster
        val pointsOuter = clusterPoints.get(i).get;
        val pointsInner = clusterPoints.get(i).get;

        for (pointOuter <- pointsOuter) {
            // Get all the points in 'countSameCluster' for the
            // particular data point pointOuter in cluster
            var tempMap = countSameCluster.get(pointOuter).get //returns HashMap[String,Integer];

            for (pointInner <- pointsInner) {
                breakable {
                    if(pointOuter == pointInner){
                        break() ; // ignore self
                    }
                    else {
                        // Increment # times pointInner & pointOuter were together in a cluster
                        var currentValue = tempMap.get(pointInner).get;
                        currentValue +=1 ;
                        tempMap.put(pointInner, currentValue);
                    }
                }
            }
            allPoints.update(pointOuter,tempMap)
        }
    }

    for ( (key,value) <- allPoints) {
        countSameCluster.update(key,value)
    }
  }

    // * At end of an iteration, update global data structures based on the subsample used in the iteration
    // * and data points in the clusters created in that iteration
    // *
    // * param collectedSampledData - An ordered collection storing data points where point at i-th element
    // * belongs to the cluster defined in i-th element of collectedClusterIndexes
    // * param collectedClusterIndexes - An ordered collection where each element represents a cluster, 0, 1, 2
    // * param clusterCenters - Coordinates of each cluster

    def updateHistogramData(collectedSampledData: Array[String],  collectedClusterIndexes: Array[Integer]){

    // Update the 'countSameSample' data structure
    updateSameSampleCounts(collectedSampledData);

    // Key to 'clusterPoints' is a cluster identifier, e.g. 0, 1, 2
    // The value is a Set where each element is a data point in the corresponding
    // cluster; data point is represented by its coordinates - a string of whitespace separated
    // numbers
    var clusterPoints = new scala.collection.mutable.HashMap[Integer, scala.collection.mutable.Set[String]];

    for ( (cluster,index) <- collectedClusterIndexes.zipWithIndex) {
        var point = collectedSampledData(index)
      clusterPoints.get(cluster) match {
          case Some(_) => {
              var tempSet = clusterPoints.apply(cluster)
              tempSet += point
              clusterPoints.update(cluster,tempSet) }
          case _ => clusterPoints.put(cluster,scala.collection.mutable.Set(point))
      }
    }
    // Update the 'countSameCluster' data structure
    updateSameClusterCounts(clusterPoints);
  }
    // * Calculate and print out histogram data. Data for 10 bins will be printed out:
// * 0.0 <number to display in bin [0.0,0.1)>
// * 0.1 <number to display in bin [0.1,0.2)>
// * 0.2 <number to display in bin [0.2,0.3)>
// * ...
// * 0.9 <number to display in bin [0.9,1.0]>

def generateHistogram(histogramParam: Histogram ) {
    for (sOuter <- countSameCluster.keySet) {
        // sOuter is a particular data point.
        // inSameCluster stores how many times a particular data point was in the same cluster as sOuter
        // inSameSample stores how many times a particular data point was in the same subsample as sOuter
        val inSameCluster = countSameCluster.apply(sOuter);
        val inSameSample = countSameSample.apply(sOuter);

        for ( sInner <- inSameCluster.keySet) {
            // sInner is a particular data point that was in the same cluster as sOuter
            breakable {
                if (sOuter == sInner) {
                    break(); // Ignore self
                }
                else {
                    // how many times sInner and sOuter were in the same cluster
                    val numTimesInSameCluster = inSameCluster.apply(sInner).toDouble;

                    // how many times sInner and sOuter were in the same subsample
                    val numTimesInSameSample = inSameSample.apply(sInner).toDouble;

                    // Calculate the ratio and place into the corresponding bin
                    var ratio = 0D

                    if (numTimesInSameSample != 0) ratio = (numTimesInSameCluster/numTimesInSameSample).toDouble else ratio = 0D

                    if (ratio >= 0  && ratio < 0.1) {
                    var value = histogramParam.histogram.apply(0);
                    value += 1;
                    histogramParam.histogram.update(0, value); }
                    else if (ratio >= 0.1  && ratio < 0.2) {
                    var value = histogramParam.histogram.apply(1);
                    value += 1;
                    histogramParam.histogram.update(1, value); }
                    else  if (ratio >= 0.2 && ratio < 0.3) {
                    var value = histogramParam.histogram.apply(2);
                    value += 1;
                    histogramParam.histogram.update(2, value); }
                    else if (ratio >= 0.3 && ratio < 0.4) {
                    var value = histogramParam.histogram.apply(3);
                    value += 1;
                    histogramParam.histogram.update(3, value); }
                    else if (ratio >= 0.4 && ratio < 0.5) {
                    var value = histogramParam.histogram.apply(4);
                    value += 1;
                    histogramParam.histogram.update(4, value); }
                    else if (ratio >= 0.5 && ratio < 0.6) {
                    var value = histogramParam.histogram.apply(5)
                    value += 1;
                    histogramParam.histogram.update(5, value); }
                    else if (ratio >= 0.6 && ratio < 0.7) {
                    var value = histogramParam.histogram.apply(6);
                    value += 1;
                    histogramParam.histogram.update(6, value); }
                    else if (ratio >= 0.7 && ratio < 0.8) {
                    var value = histogramParam.histogram.apply(7);
                    value += 1;
                    histogramParam.histogram.update(7, value); }
                    else if (ratio >= 0.8 && ratio < 0.9) {
                    var value = histogramParam.histogram.apply(8);
                    value += 1;
                    histogramParam.histogram.update(8, value); }
                    else if (ratio >= 0.9 && ratio <= 1) {
                    var value = histogramParam.histogram.apply(9);
                    value += 1;
                    histogramParam.histogram.update(9, value); }
          }
        }
      }
    }
  }

  //dataCenters is a trait that contains abstract method dataCenters that will be extended in later algorithm specific implementations
  trait dataCenters {
  def dataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer, numIterations: Integer, numSeed: Long = 1L ): org.apache.spark.rdd.RDD[Integer]
  }

  // This is the main method that performs all the tasks.
  abstract class iterations(var histogramParam: Histogram, val data:org.apache.spark.sql.DataFrame, val numClusters:Integer = 5,val numIterations:Integer = 125,val numSeed:Long = 1L,val numRuns:Integer = 20) extends dataCenters {
    println("Number of non-unique data points:  "+ data.count.toString);
    // Remove any duplicates
    val fullData = data.distinct();
    println("Number of unique data points: " + fullData.count.toString);

    // Initialize global data structures
    initializeCounts(fullData);

    // Instantiate the histogram data structure
    //var histogram = new Histogram

    // Each execution of this loop corresponds to an iteration
    for ( iteration <- 0 until numRuns) {
        // Obtain a random subsample, consisting of 90% of the original data set
        val rando = new scala.util.Random()
        val sampledData = fullData.sample(false, 0.9, rando.nextLong() );
        val sampledDataRDD = sampledData.rdd.map(_.mkString(",")) //Convert to RDD[String]

        // Rely on concrete subclasses for this method. This is where clusters are
        // created by a specific algorithm
        val clusterIndexes = dataCenters(sampledData, numClusters, numIterations, numSeed);

        // Bring all data to driver node for displaying results.
        // 'collectedSampledData' and 'collectedClusterIndexes' are ordered
        // collections with the same size: i-th element in 'collectedSampledData'
        // is a data point that belongs to the cluster defined in the i-th element
        // in 'collectedClusterIndexes'
        val collectedSampledData = sampledDataRDD.collect() //convert to Array[String]
        val collectedClusterIndexes = clusterIndexes.collect() //convert to Array[Integer]
        println("Number of data points in random sample:"+collectedSampledData.size);
        // Update global data structures based on cluster data
        updateHistogramData(collectedSampledData, collectedClusterIndexes);
        // Bucket consensus cluster results in histogram
        generateHistogram(histogramParam)
    }
    // Display histogram data
    def showResults {
        println("HISTOGRAM");
        for ( i <- 0 to 9 ) {
            println((i).toString + " " + histogramParam.histogram.apply(i).toString);
      }
    }
  }

//////////////////////////
// KMeans Implementation
//////////////////////////

class KMeansClustering(histogramParam: Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) extends iterations( histogramParam:Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) {


    /**
	* This is the abstract method defined in ConsensusCluster whose implementation
	* is left to child classes. Here we first create a KMeans object and then call
	* its run() method on data to obtain a KMeansModel object. Then, we call the
	* predict method on KMeansModel object to obtain a data structure, clusterIndexes,
	* that gives the cluster indexes for the input parameter data. The clusterIndexes
	* and data have the same number of elements. The elements in the clusterIndexes and
	* data have reciprocal sequence in the sense that the i-th element of clusterIndexes
	* defines which cluster the i-th element of data belongs to, one of 0, 1, 2 ... etc.
	*
	* param data - Data points to partition into clusters
	* param numClusters - number of clusters desired
	* param numIterations - maximum # of iterations to perform during clustering
	*/
	override def dataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer, numIterations: Integer, numSeed: Long = 1L ): org.apache.spark.rdd.RDD[Integer] = {
		val km = new org.apache.spark.ml.clustering.KMeans().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed).setPredictionCol("predict");
		val clusters = km.fit(data);
		val clusterIndexes = clusters.transform(data);
		val clusterIndexesRDD = clusterIndexes.select("predict").rdd.map(row => row.getAs[Integer](0))
		return clusterIndexesRDD;
	}
}

///////////////////////////////////
// Bisecting KMeans Implementation
///////////////////////////////////

class BKMeansClustering(histogramParam: Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) extends iterations(histogramParam: Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) {
    /**
	* This is the abstract method defined in ConsensusCluster whose implementation
	* is left to child classes. Here we first create a KMeans object and then call
	* its run() method on data to obtain a KMeansModel object. Then, we call the
	* predict method on KMeansModel object to obtain a data structure, clusterIndexes,
	* that gives the cluster indexes for the input parameter data. The clusterIndexes
	* and data have the same number of elements. The elements in the clusterIndexes and
	* data have reciprocal sequence in the sense that the i-th element of clusterIndexes
	* defines which cluster the i-th element of data belongs to, one of 0, 1, 2 ... etc.
	*
	* param data - Data points to partition into clusters
	* param numClusters - number of clusters desired
	* param numIterations - maximum # of iterations to perform during clustering
	*
	*/
	override def dataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer, numIterations: Integer, numSeed: Long = 1L ): org.apache.spark.rdd.RDD[Integer] = {
		val km = new org.apache.spark.ml.clustering.BisectingKMeans().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed).setPredictionCol("predict");
		val clusters = km.fit(data);
		val clusterIndexes = clusters.transform(data);
		val clusterIndexesRDD = clusterIndexes.select("predict").rdd.map(row => row.getAs[Integer](0))
		return clusterIndexesRDD;
	}
}

/////////////////////////////////////////
// Gaussian Mixture Model Implementation
/////////////////////////////////////////

class GMMClustering(histogramParam: Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) extends iterations(histogramParam: Histogram, data:org.apache.spark.sql.DataFrame, numClusters:Integer, numIterations:Integer , numSeed:Long , numRuns:Integer ) {
    /**
	* This is the abstract method defined in ConsensusCluster whose implementation
	* is left to child classes. Here we first create a GaussianMixture object and then call
	* its run() method on data to obtain a GaussianMixtureModel object. Then, we call the
	* predict method on GaussianMixtureModel object to obtain a data structure, clusterIndexes,
	* that gives the cluster indexes for the input parameter data. The clusterIndexes
	* and data have the same number of elements. The elements in the clusterIndexes and
	* data have reciprocal sequence in the sense that the i-th element of clusterIndexes
	* defines which cluster the i-th element of data belongs to, one of 0, 1, 2 ... etc.
	*
	* @param data - Data points to partition into clusters
	* @param numClusters - number of clusters desired
	* @param numIterations - maximum # of iterations to perform during clustering
	*
	*/
	override def dataCenters(data: org.apache.spark.sql.DataFrame, numClusters: Integer, numIterations: Integer, numSeed: Long = 1L ): org.apache.spark.rdd.RDD[Integer] = {
		val km = new org.apache.spark.ml.clustering.GaussianMixture().setK(numClusters).setMaxIter(numIterations).setSeed(numSeed).setPredictionCol("predict");
		val clusters = km.fit(data);
		val clusterIndexes = clusters.transform(data);
		val clusterIndexesRDD = clusterIndexes.select("predict").rdd.map(row => row.getAs[Integer](0))
		return clusterIndexesRDD;
	}
}



///////////////////////////////////////////
// How to create a new KMeans cluster test
///////////////////////////////////////////

//KM Test
var histogramKM = new Histogram
val testKM = new KMeansClustering(histogramKM,parsedData,numClusters,numIterations,numSeed, 20)
testKM.showResults

//BKM Test
var histogramBKM = new Histogram
val testBKM = new BKMeansClustering(histogramBKM,parsedData,numClusters,numIterations,numSeed, 20)
testBKM.showResults

//GMM Test
var histogramGMM = new Histogram
val testGMM = new GMMClustering(histogramGMM,parsedData,numClusters,numIterations,numSeed, 20)
testGMM.showResults
