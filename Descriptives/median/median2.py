from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
import time

# Create a Spark session
spark = SparkSession.builder.appName("MedianCalculator") \
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
iris_df = spark.read.csv("../../iris100Kx.csv", header=False, inferSchema=True, schema=iris_schema)

# Calculate the median for each column using Spark SQL median function
median_df = iris_df.select(
   F.median("sepal_length").alias("median_sepal_length"),
   F.median("sepal_width").alias("median_sepal_width"),
   F.median("petal_length").alias("median_petal_length"),
   F.median("petal_width").alias("median_petal_width")
)

# Display the results
median_df.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark...")

# Stop the Spark session
spark.stop()
