from pyspark.sql import SparkSession

from pyspark.ml.clustering import KMeans, GaussianMixture, PowerIterationClustering, LDA, BisectingKMeans

spark = SparkSession.builder.appName("pyspark-notebook").getOrCreate()
data = [(1, 0, 0.5),
        (2, 0, 0.5), (2, 1, 0.7),
        (3, 0, 0.5), (3, 1, 0.7), (3, 2, 0.9),
        (4, 0, 0.5), (4, 1, 0.7), (4, 2, 0.9), (4, 3, 1.1),
        (5, 0, 0.5), (5, 1, 0.7), (5, 2, 0.9), (5, 3, 1.1), (5, 4, 1.3)]
df = spark.createDataFrame(data).toDF("src", "dst", "weight").repartition(1)
pic = PowerIterationClustering(k=2, weightCol="weight")
assignments = pic.assignClusters(df)
assignments.sort(assignments.id).show(truncate=False)