from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
import time


# Create a Spark session
spark = SparkSession.builder.appName("MeanCalculator") \
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
iris_df = spark.read.csv("iris10Kx.csv", header=False, schema=iris_schema)

# Calculate the mean for each column
means_df = iris_df.agg(
    F.mean("sepal_length").alias("mean_sepal_length"),
    F.mean("sepal_width").alias("mean_sepal_width"),
    F.mean("petal_length").alias("mean_petal_length"),
    F.mean("petal_width").alias("mean_petal_width")
)

# Display the results
means_df.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark...")

# Stop the Spark session
spark.stop()
