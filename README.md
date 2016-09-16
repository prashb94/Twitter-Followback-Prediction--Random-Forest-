/*
Twitter Follow-Back Prediction

Tsinghua University - Big Data Summer Camp 2016

Country - India

Name - Prashanth Balasubramanian

Algorithm Used - Random Forest Classifier

Features Used/Model Performance/Sample Input/Output - **SEE BELOW**

*/


/*

INSTRUCTIONS TO EXECUTE 

1. spark-shell --executor-memory 10G or just spark-shell
2. :paste
3. Copy and paste the entire code into the shell
4. CTRL+D

Note - Make sure that graph_cb.txt and interaction_list_all.txt are copied onto root at HDFS

*/


/* FEATURES USED

1. Number of users that are following USER#1 (Popularity metric #1)
2. Number of users that are following USER#2 (Popularity metric #1)
3. Number of users that USER#1 is following (Popularity metric #2)
4. Number of users that USER#2 is following (Popularity metric #2)
5. Number of times that USER#1 has been mentioned on Twitter (Popularity metric #3)
6. Number of times that USER#2 has been mentioned on Twitter (Popularity metric #3)
7. Number of times USER#1 has mentioned USER#2 
8. Number of times USER#2 has mentioned USER#1
9. Number of days after which a connection between USER#1 and USER#2 occured (Case 1 - If both users connected on the same day => Score = 200; Case 2 - If both users connected after X days => Score = abs(X); Case 3 - If both users did not connect at all in the past => Score = -1)

Label -> 1.0 if there will be a connection between USER#2 and USER#1, given that USER#1 has already followed USER#2 
         0.0 otherwise


*/


/* FINAL INPUT WITH 9 FEATURES - SAMPLE

(1.0,[3.0,6.0,5.0,6.0,3.0,1.0,0.0,1.0,200.0])
(1.0,[3.0,2.0,5.0,3.0,3.0,1.0,0.0,0.0,200.0])
(0.0,[3.0,12.0,5.0,9.0,3.0,35.0,0.0,0.0,-1.0])
(0.0,[3.0,10.0,5.0,2.0,3.0,0.0,0.0,0.0,-1.0])
(0.0,[3.0,4.0,5.0,1.0,3.0,4.0,0.0,0.0,-1.0])
(0.0,[0.0,47.0,1.0,11.0,0.0,0.0,0.0,0.0,-1.0])
(0.0,[3.0,6.0,3.0,1.0,0.0,0.0,0.0,0.0,-1.0])
(1.0,[3.0,3.0,3.0,1.0,0.0,0.0,0.0,0.0,200.0])
(1.0,[3.0,5.0,3.0,4.0,0.0,2.0,0.0,0.0,200.0])
(1.0,[1.0,2.0,8.0,7.0,1.0,19.0,1.0,1.0,200.0])
.
.
.

*/


/* MODEL PERFORMANCE

testErr: Double = 1.0

scala> precision.foreach { case (t, p) =>
     |   println(s"Threshold: $t, Precision: $p")
     | }

Threshold: 1.0, Precision: 1.0
Threshold: 0.0, Precision: 0.6063996042409183

scala> model
res21: org.apache.spark.mllib.tree.model.RandomForestModel =
TreeEnsembleModel classifier with 9 trees

scala> val recall = metrics.recallByThreshold
recall: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[376] at map at BinaryClassificationMetrics.scala:216

scala> recall.foreach { case (t, r) =>
     |   println(s"Threshold: $t, Recall: $r")
     | }
Threshold: 1.0, Recall: 1.0
Threshold: 0.0, Recall: 1.0

scala> val PRC = metrics.pr
PRC: org.apache.spark.rdd.RDD[(Double, Double)] = UnionRDD[379] at union at BinaryClassificationMetrics.scala:111

scala> val f1Score = metrics.fMeasureByThreshold
f1Score: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[380] at map at BinaryClassificationMetrics.scala:216

scala> f1Score.foreach { case (t, f) =>
     |   println(s"Threshold: $t, F-score: $f, Beta = 1")
     | }
Threshold: 1.0, F-score: 1.0, Beta = 1
Threshold: 0.0, F-score: 0.7549797729531488, Beta = 1

scala> val auPRC = metrics.areaUnderPR
auPRC: Double = 1.0

scala> println("Area under precision-recall curve = " + auPRC)
Area under precision-recall curve = 1.0

scala> val thresholds = precision.map(_._1)
thresholds: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[386] at map at <console>:77

scala> val roc = metrics.roc
roc: org.apache.spark.rdd.RDD[(Double, Double)] = UnionRDD[390] at UnionRDD at BinaryClassificationMetrics.scala:92

scala> val auROC = metrics.areaUnderROC
auROC: Double = 1.0

scala> println("Area under ROC = " + auROC)
Area under ROC = 1.0

*/



/* RANDOM FOREST WITH 9 TREES

.
.
.
.
 Tree 6:
   If (feature 3 <= 2.0)
    If (feature 3 <= 0.0)
     Predict: 0.0
    Else (feature 3 > 0.0)
     If (feature 8 <= -1.0)
      Predict: 0.0
     Else (feature 8 > -1.0)
      Predict: 1.0
   Else (feature 3 > 2.0)
    If (feature 5 <= 0.0)
     If (feature 0 <= 1.0)
      If (feature 2 <= 1.0)
       If (feature 4 <= 1.0)
        If (feature 8 <= -1.0)
         Predict: 0.0
        Else (feature 8 > -1.0)
         Predict: 1.0
       Else (feature 4 > 1.0)
        If (feature 0 <= 0.0)
         Predict: 0.0
        Else (feature 0 > 0.0)
         If (feature 7 <= 1.0)
          If (feature 3 <= 3.0)
           If (feature 1 <= 2.0)
            Predict: 0.0
           Else (feature 1 > 2.0)
            If (feature 8 <= -1.0)
             Predict: 0.0
            Else (feature 8 > -1.0)
             Predict: 1.0
          Else (feature 3 > 3.0)
           Predict: 1.0
         Else (feature 7 > 1.0)
          If (feature 8 <= -1.0)
           Predict: 0.0
          Else (feature 8 > -1.0)
           Predict: 1.0
      Else (feature 2 > 1.0)
       If (feature 7 <= 2.0)
        If (feature 8 <= -1.0)
         Predict: 0.0
        Else (feature 8 > -1.0)
         Predict: 1.0
       Else (feature 7 > 2.0)
        If (feature 8 <= -1.0)
         Predict: 0.0
        Else (feature 8 > -1.0)
         Predict: 1.0
     Else (feature 0 > 1.0)
      If (feature 8 <= -1.0)
       Predict: 0.0
      Else (feature 8 > -1.0)
       Predict: 1.0
    Else (feature 5 > 0.0)
     If (feature 0 <= 1.0)
      If (feature 8 <= -1.0)
       Predict: 0.0
      Else (feature 8 > -1.0)
       Predict: 1.0
     Else (feature 0 > 1.0)
      If (feature 6 <= 0.0)
       If (feature 8 <= -1.0)
        Predict: 0.0
       Else (feature 8 > -1.0)
        Predict: 1.0
      Else (feature 6 > 0.0)
       If (feature 6 <= 2.0)
        If (feature 0 <= 5.0)
         If (feature 2 <= 8.0)
          If (feature 8 <= -1.0)
           Predict: 0.0
          Else (feature 8 > -1.0)
           Predict: 1.0
         Else (feature 2 > 8.0)
          If (feature 8 <= -1.0)
.
.
.
.
.
.*/

