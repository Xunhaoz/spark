import logging
import warnings
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor


def data_preprocess(data, X=False):
    data = data.withColumn("id", col("id").cast("string"))
    for column_name in data.columns[1:]:
        data = data.withColumn(column_name, col(column_name).cast("float"))

    if X:
        assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaler_features", withMean=True, withStd=True)
        pipeline = Pipeline(stages=[assembler, scaler])
        data = pipeline.fit(data).transform(data)
        data = data.select("id", "scaler_features")

    return data


warnings.filterwarnings('ignore')

spark = SparkSession.builder \
    .appName("LinearRegression") \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
log4jLogger = spark._jvm.org.apache.log4j
log4jLogger.LogManager.getLogger("org").setLevel(log4jLogger.Level.ERROR)

data_X = spark.read.csv("data/data_X.csv", header=True)
data_X = data_preprocess(data_X, X=True)

data_Y = spark.read.csv("data/data_Y.csv", header=True)
data_Y = data_preprocess(data_Y)

data = data_X.join(data_Y, "id", "inner")

growth_rate_lr = DecisionTreeRegressor(featuresCol="scaler_features", labelCol="growth_rate")
length_weight_lr = DecisionTreeRegressor(featuresCol="scaler_features", labelCol="length_weight")

growth_rate_model = growth_rate_lr.fit(data)
length_weight_model = length_weight_lr.fit(data)

# growth_rate_model.write().overwrite().save("C:/Users/Xunhaoz/Desktop/spark/models/growth_rate_linear_regression")
# length_weight_model.write().overwrite().save("C:/Users/Xunhaoz/Desktop/spark/models/length_weight_linear_regression")

print(growth_rate_model.summary)
print(length_weight_model.summary)

spark.stop()
