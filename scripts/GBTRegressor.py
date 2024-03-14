from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, \
    RandomForestRegressor, GBTRegressor, IsotonicRegression, AFTSurvivalRegression, FMRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, \
    GBTClassifier, MultilayerPerceptronClassifier, LinearSVC, OneVsRest, NaiveBayes, FMClassifier

spark = SparkSession.builder.appName("pyspark-notebook").getOrCreate()
data = spark.read.csv("./dataset/diabetes.csv", header=True, inferSchema=True)
data = data.withColumn("censor", lit(1))

assembler = VectorAssembler(inputCols=[
    'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'
], outputCol="features")
data = assembler.transform(data)

train, test = data.randomSplit([0.7, 0.3], seed=41)

model = GBTRegressor(featuresCol="features", labelCol="target")
model = model.fit(train)

predictions = model.transform(test)
predictions.select("target", "prediction").show(100, truncate=False)
spark.stop()
