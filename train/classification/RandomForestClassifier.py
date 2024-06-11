import warnings
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

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


spark = SparkSession.builder.appName("RandomForestClassifier").getOrCreate()

data_X = pd.read_csv("data/data_X.csv")
data_X = pandas_2_spark(data_X, spark, X=True)

data_Y = pd.read_csv("data/classification_data_Y.csv")
data_Y = pandas_2_spark(data_Y, spark)

data = data_X.join(data_Y, "id", "inner")

growth_rate = RandomForestClassifier(featuresCol="scaler_features", labelCol="growth_rate")
length_weight = RandomForestClassifier(featuresCol="scaler_features", labelCol="length_weight")

growth_rate_model = growth_rate.fit(data)
length_weight_model = length_weight.fit(data)

# growth_rate_model.write().overwrite().save(
#     "./models/growth_rate_random_forest_classifier"
# )
# length_weight_model.write().overwrite().save(
#     "./models/length_weight_random_forest_classifier"
# )
#
# growth_rate_model = RandomForestClassificationModel.load(
#     "./models/growth_rate_random_forest_classifier"
# )
# length_weight_model = RandomForestClassificationModel.load(
#     "./models/length_weight_random_forest_classifier"
# )

growth_rate = growth_rate_model.transform(data)
length_weight = length_weight_model.transform(data)

growth_rate.show()
length_weight.show()

spark.stop()
