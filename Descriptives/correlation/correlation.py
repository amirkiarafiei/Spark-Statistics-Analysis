from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import time



# Create a Spark session
spark = SparkSession.builder.appName("CorrelationCalculator") \
    .master("local[]") \
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
iris_data = spark.read.csv("../../iris100Kx.csv", header=False, schema=iris_schema)

# Select relevant columns (sepal length, sepal width, petal length, petal width)
selected_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_data = iris_data.select(selected_columns)

# Assemble features into a single vector column
vector_assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")
iris_data = vector_assembler.transform(iris_data)

# Calculate the correlation matrix
correlation_matrix = Correlation.corr(iris_data, "features").head()

# Extract correlations for sepal length (index 0) with other features (index 1, 2, 3)
sepal_length_correlations = correlation_matrix[0].toArray()[0:4]


# Display the result
print(f"Correlation between feature 0 and others: {sepal_length_correlations[0]}")
print(f"Correlation between feature 1 and others: {sepal_length_correlations[1]}")
print(f"Correlation between feature 2 and others: {sepal_length_correlations[2]}")
print(f"Correlation between feature 3 and others: {sepal_length_correlations[3]}")


# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop Spark session
spark.stop()
