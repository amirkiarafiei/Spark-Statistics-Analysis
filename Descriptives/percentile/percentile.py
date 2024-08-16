from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, percentile_approx
import time

# Create a Spark session
spark = SparkSession.builder.appName("PercentileCalculator") \
        .master("local[1]") \
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
iris_df = spark.read.csv("../../iris10Kx.csv", header=False, inferSchema=True, schema=iris_schema)

# Define the desired percentile
percentile = 25  # Change this to the desired percentile

# Iterate over all four columns
for col_name in iris_df.columns:
    percentile_value = iris_df.select(percentile_approx(col_name, percentile/100)).collect()[0][0]
    print(f"{percentile}th Percentile for column {col_name}: {percentile_value}")

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

# Stop the Spark session
spark.stop()
