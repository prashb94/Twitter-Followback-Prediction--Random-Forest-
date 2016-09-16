import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy

val f = sc.textFile("graph_cb.txt")
val m = sc.textFile("interaction_list_all.txt")
val follow = f.map(line => {
	var arr = line.split(' ')
	((arr(0).toLong,arr(1).toLong),arr(2).toInt)
	})
val mention = m.map(line => {
	var arr = line.split(' ')
	((arr(0).toLong,arr(1).toLong),arr(2).toInt)
	}).reduceByKey(_+_)

val numberFollowing = f.map(line => {
	var arr = line.split(' ')
	(arr(0).toLong,1)
	}).reduceByKey(_+_).collectAsMap

val numberOfFollowers = f.map(line => {
	var arr = line.split(' ')
	(arr(1).toLong,1)
	}).reduceByKey(_+_).collectAsMap

val numberOfMentionsComplete = m.map(line => {
	var arr = line.split(' ')
	(arr(1).toLong,1)
	}).reduceByKey(_+_).collectAsMap

val mentionMap = mention.collectAsMap

val followMap = follow.collectAsMap

val data = follow.map{case((id1,id2),n) => {
var finalNum = 0
var label = -1.0
var mentionOneTwo = 0
var mentionTwoOne = 0
var mentionOne = 0
var mentionTwo = 0
var oneFollowing = 0
var twoFollowing = 0
var oneFollowers = 0
var twoFollowers = 0
if(followMap.contains((id2,id1)))
{
var x = followMap((id2,id1)) - n

if(x==0) 
{finalNum = 200}
else
{finalNum = scala.math.abs(x)}
}
else 
{finalNum = -1}

if(finalNum == -1) 
{label = 0.0}
else 
{label = 1.0}

if(numberFollowing.contains(id1)) {oneFollowing = numberFollowing(id1)}
if(numberFollowing.contains(id2)) {twoFollowing = numberFollowing(id2)}

if(numberOfFollowers.contains(id1)) {oneFollowers = numberOfFollowers(id1)}
if(numberOfFollowers.contains(id2)) {twoFollowers = numberOfFollowers(id2)}

if(numberOfMentionsComplete.contains(id1)) {mentionOne = numberOfMentionsComplete(id1)}
if(numberOfMentionsComplete.contains(id2)) {mentionTwo = numberOfMentionsComplete(id2)}

if(mentionMap.contains((id1,id2))) {mentionOneTwo = mentionMap((id1,id2))}
if(mentionMap.contains((id2,id1))) {mentionTwoOne = mentionMap((id2,id1))}

(LabeledPoint(label,Vectors.dense(oneFollowers,twoFollowers,oneFollowing,twoFollowing,mentionOne,mentionTwo,mentionOneTwo,mentionTwoOne,finalNum)))
}
}

/* RANDOM FOREST CLASSIFIER */

val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
val treeStrategy = Strategy.defaultStrategy("Classification")
val numTrees = 9
val featureSubsetStrategy = "auto"
val model = RandomForest.trainClassifier(trainingData,
  treeStrategy, numTrees, featureSubsetStrategy, seed = 12345)
val testErr = testData.map { point =>
  val prediction = model.predict(point.features)
  if (point.label == prediction) 1.0 else 0.0
}.mean()
println("Test Error = " + testErr)
println("Learned Random Forest:n" + model.toDebugString)
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
val precision = metrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}
val recall = metrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}
val PRC = metrics.pr
val f1Score = metrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)
val thresholds = precision.map(_._1)
val roc = metrics.roc
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)