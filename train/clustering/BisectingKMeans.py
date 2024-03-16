import warnings
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans, BisectingKMeansModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def pandas_2_spark(data_frame, spark, X=False):
    index = data_frame['id']
    data_frame = data_frame.iloc[:, 1:]

    if X:
        columns = data_frame.columns
        data_frame = StandardScaler().fit_transform(data_frame)
        data_frame = MinMaxScaler().fit_transform(data_frame)
        data_frame = pd.DataFrame(data_frame, columns=columns)

    data_frame['id'] = index
    data = spark.createDataFrame(data_frame)

    if X:
        assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol='scaler_features')
        data = assembler.transform(data)
        data = data.select('id', 'scaler_features')
    return data


spark = SparkSession.builder.appName("KMeans").getOrCreate()

data_X = pd.read_csv("data/data_X.csv")
data_X = pandas_2_spark(data_X, spark, X=True)

growth_rate = BisectingKMeans(featuresCol="scaler_features", k=2)
length_weight = BisectingKMeans(featuresCol="scaler_features", k=2)

growth_rate_model = growth_rate.fit(data_X)
length_weight_model = length_weight.fit(data_X)

growth_rate_model.write().overwrite().save(
    "./models/growth_rate_bisecting_kmeans"
)
length_weight_model.write().overwrite().save(
    "./models/length_weight_bisecting_kmeans"
)

growth_rate_model = BisectingKMeansModel.load(
    "./models/growth_rate_bisecting_kmeans"
)
length_weight_model = BisectingKMeansModel.load(
    "./models/length_weight_bisecting_kmeans"
)

growth_rate = growth_rate_model.transform(data_X)
length_weight = length_weight_model.transform(data_X)

growth_rate.show()
length_weight.show()

spark.stop()
