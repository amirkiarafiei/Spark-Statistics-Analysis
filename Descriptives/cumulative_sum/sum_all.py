from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
import time



# Create a Spark session
spark = SparkSession.builder.appName("CumulativeSum") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Start time
start_time = time.time()

# Replace 'your_dataset' with your actual dataset
iris_schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("class", DoubleType(), True),  # Assuming the class column is the fifth column (index 4)
])

# Load the Iris dataset with the specified schema
iris_df = spark.read.csv("../../iris100Kx.csv", header=False, inferSchema=True, schema=iris_schema)

# Assuming the class column is the fifth column (index 4)
class_index = 4

# Calculate cumulative sum for each column
cumulative_sums = iris_df.groupBy("class").agg(
    F.sum("sepal_length").alias("cumulative_sum_sepal_length"),
    F.sum("sepal_width").alias("cumulative_sum_sepal_width"),
    F.sum("petal_length").alias("cumulative_sum_petal_length"),
    F.sum("petal_width").alias("cumulative_sum_petal_width")
)

# Display results
cumulative_sums.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop Spark
spark.stop()
