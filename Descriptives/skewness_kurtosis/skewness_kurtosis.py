from pyspark.sql import SparkSession
from pyspark.sql.functions import col, skewness, kurtosis
import time



# Initialize a Spark session
spark = SparkSession.builder.appName("IrisSkewnessKurtosis") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()


# Start time
start_time = time.time()

# Provide column names manually
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Read the CSV file without inferring the schema and provide column names
iris_data = spark.read.csv("../../iris100Kx.csv", header=False, inferSchema=True).toDF(*column_names)

# Calculate skewness and kurtosis for each column
skewness_values = iris_data.select([skewness(col(c)).alias(c) for c in iris_data.columns])
kurtosis_values = iris_data.select([kurtosis(col(c)).alias(c) for c in iris_data.columns])

# Display the results
print("Skewness:")
skewness_values.show()

print("Kurtosis:")
kurtosis_values.show()


# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop the Spark session
spark.stop()
