from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
import time


# Create a Spark session
spark = SparkSession.builder.appName("MinMaxCalculator") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Start time
start_time = time.time()

# Define the schema for the Iris dataset
iris_schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
])

# Load the Iris dataset with the specified schema
iris_df = spark.read.csv("iris10Kx.csv", header=False, inferSchema=True, schema=iris_schema)

# Calculate the minimum and maximum for each column
min_max_df = iris_df.agg(
    F.min("sepal_length").alias("min_sepal_length"),
    F.max("sepal_length").alias("max_sepal_length"),
    F.min("sepal_width").alias("min_sepal_width"),
    F.max("sepal_width").alias("max_sepal_width"),
    F.min("petal_length").alias("min_petal_length"),
    F.max("petal_length").alias("max_petal_length"),
    F.min("petal_width").alias("min_petal_width"),
    F.max("petal_width").alias("max_petal_width")
)

# Display the results
min_max_df.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark...")

# Stop the Spark session
spark.stop()
