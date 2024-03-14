from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, \
    RandomForestRegressor, GBTRegressor, IsotonicRegression, AFTSurvivalRegression, FMRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, \
    GBTClassifier, MultilayerPerceptronClassifier, LinearSVC, NaiveBayes, FMClassifier
from pyspark.ml.clustering import KMeans, GaussianMixture, PowerIterationClustering, LDA, BisectingKMeans

spark = SparkSession.builder.appName("pyspark-notebook").getOrCreate()
data = spark.read.csv("./dataset/iris.csv", header=True, inferSchema=True)
data = data.filter(data["target"] < 2)
data = data.withColumn("censor", lit(1))

assembler = VectorAssembler(inputCols=[
    'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'censor'
], outputCol="features")
data = assembler.transform(data)

train, test = data.randomSplit([0.7, 0.3])

model = LDA(k=2)
model = model.fit(train)

predictions = model.transform(test)
predictions.show(100, truncate=False)
spark.stop()
