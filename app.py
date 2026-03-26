
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Ride Demand Pipeline") \
    .getOrCreate()

df = spark.read.csv("ncr_ride_bookings.csv", header=True, inferSchema=True,nullValue='null')
df.show(5)

from pyspark.sql.functions import to_timestamp, concat_ws, hour, dayofweek

df = df.withColumn(
    "datetime",
    to_timestamp(df["Time"])
)

df = df.withColumn("hour", hour("datetime")) \
       .withColumn("day", dayofweek("datetime"))

from pyspark.sql.functions import count

df_grouped = df.groupBy("hour", "day", "Pickup Location") \
               .agg(count("Booking ID").alias("trip_count"))

from pyspark.sql.functions import when

df_grouped = df_grouped.withColumn(
    "demand_label",
    when(df_grouped["trip_count"] < 20, 0)   # Low
    .when(df_grouped["trip_count"] < 50, 1)  # Medium
    .otherwise(2)                            # High
)



from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

assembler = VectorAssembler(
    inputCols=["hour", "day"],
    outputCol="features"
)

data = assembler.transform(df_grouped)

train, test = data.randomSplit([0.8, 0.2])

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="demand_label"
)

model = rf.fit(train)

predictions = model.transform(test)
predictions.select("hour", "day", "prediction").show(6)
import pickle

pickle.dump(model, open("model.pkl", "wb"))


import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()

model = RandomForestClassificationModel.load("ride_demand_model")

st.title("🚕 Ride Demand Predictor (India)")

hour = st.slider("Select Hour", 0, 23)
day = st.selectbox("Select Day (1=Sun ... 7=Sat)", [1,2,3,4,5,6,7])

input_df = pd.DataFrame([[hour, day]], columns=["hour", "day"])

prediction = model.predict(input_df)[0]

if prediction == 0:
    st.success("🟢 Low Demand")
elif prediction == 1:
    st.warning("🟡 Medium Demand")
else:
    st.error("🔴 High Demand (Surge likely)")
