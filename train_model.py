from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, hour, dayofweek, count, when

spark = SparkSession.builder.appName("Ride Demand Pipeline").getOrCreate()

df = spark.read.csv("ncr_ride_bookings.csv", header=True, inferSchema=True, nullValue='null')

# Feature engineering
df = df.withColumn("datetime", to_timestamp(df["Time"]))
df = df.withColumn("hour", hour("datetime")) \
       .withColumn("day", dayofweek("datetime"))

# Aggregate demand
df_grouped = df.groupBy("hour", "day") \
               .agg(count("Booking ID").alias("trip_count"))

# Label
df_grouped = df_grouped.withColumn(
    "demand_label",
    when(df_grouped["trip_count"] < 20, 0)
    .when(df_grouped["trip_count"] < 50, 1)
    .otherwise(2)
)

# Convert to pandas
pdf = df_grouped.select("hour", "day", "demand_label").toPandas()

# Train sklearn model
from sklearn.ensemble import RandomForestClassifier
import pickle

X = pdf[["hour", "day"]]
y = pdf["demand_label"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model saved as model.pkl")
