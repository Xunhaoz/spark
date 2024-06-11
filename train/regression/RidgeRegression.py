import warnings
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


spark = SparkSession.builder \
    .appName("RidgeRegression") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

data_X = spark.read.csv("data/data_X.csv", header=True)
data_X = data_preprocess(data_X, X=True)

data_Y = spark.read.csv("data/data_Y.csv", header=True)
data_Y = data_preprocess(data_Y)

data = data_X.join(data_Y, "id", "inner")

growth_rate = LinearRegression(featuresCol="scaler_features", labelCol="growth_rate", elasticNetParam=0.0, regParam=0.1)
length_weight = LinearRegression(featuresCol="scaler_features", labelCol="length_weight", elasticNetParam=0.0,
                                 regParam=0.1)

growth_rate_model = growth_rate.fit(data)
length_weight_model = length_weight.fit(data)

# growth_rate_model.write().overwrite().save(
#     "./models/growth_rate_ridge_regression"
# )
# length_weight_model.write().overwrite().save(
#     "./models/length_weight_ridge_regression"
# )
#
# growth_rate_model = LinearRegressionModel.load(
#     "./models/growth_rate_ridge_regression"
# )
# length_weight_model = LinearRegressionModel.load(
#     "./models/length_weight_ridge_regression"
# )

growth_rate = growth_rate_model.transform(data)
length_weight = length_weight_model.transform(data)

growth_rate.show()
length_weight.show()

spark.stop()
