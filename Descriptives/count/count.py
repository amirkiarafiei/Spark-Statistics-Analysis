from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.sql import functions as F
import time

# Create a Spark session
spark = SparkSession.builder.appName("CountEntriesByClass") \
   .master("local[3]") \
   .config("spark.executor.memory", "2g") \
   .config("spark.executor.memoryOverhead", "1g") \
   .getOrCreate()

# Start time
start_time = time.time()

# Define the schema for the Iris dataset (assuming the class label is in the 5th column)
iris_schema = StructType([
   StructField("sepal_length", DoubleType(), True),
   StructField("sepal_width", DoubleType(), True),
   StructField("petal_length", DoubleType(), True),
   StructField("petal_width", DoubleType(), True),
   StructField("class", StringType(), True)  # Assuming class label is a string
])

# Load the Iris dataset with the specified schema
iris_df = spark.read.csv("../../iris100Kx.csv", header=False, inferSchema=True, schema=iris_schema)

# Count entries for each class
count_df = iris_df.groupBy("class").count()

# Display the results
count_df.show()

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark...")

# Stop the Spark session
spark.stop()
