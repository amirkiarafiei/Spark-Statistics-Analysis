from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
import time


# Create a Spark session
spark = SparkSession.builder.appName("IrisFeatureHistogram") \
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
iris_df = spark.read.csv("iris.csv", header=False, inferSchema=True, schema=iris_schema)

# This will create histograms for all features in the dataset
for col_name in iris_df.columns:
    feature_histogram = iris_df.groupBy(col_name).count()

    print(f"Histogram for Feature {col_name}:")
    feature_histogram.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark...")

# Stop the Spark session
spark.stop()
