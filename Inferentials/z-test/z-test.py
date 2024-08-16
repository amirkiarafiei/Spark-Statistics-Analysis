from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StructType, StructField, StringType
from scipy.stats import norm
import time

# Set the PYSPARK_PYTHON environment variable
import os
os.environ["PYSPARK_PYTHON"] = "C:\\Users\\HP\\anaconda3\\envs\\BigDataProject\\python.exe"


# Create a Spark session
spark = SparkSession.builder.appName("ZTestIris") \
    .master("local[2]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Start time
start_time = time.time()

# Define the schema for the Iris dataset
iris_schema = StructType([
    StructField("sepal_length", FloatType(), True),
    StructField("sepal_width", FloatType(), True),
    StructField("petal_length", FloatType(), True),
    StructField("petal_width", FloatType(), True),
    StructField("species", StringType(), True)  # Assuming the last column is the species
])

# Load the Iris dataset without a header
iris_df = spark.read.csv("iris10Kx.csv", header=False, schema=iris_schema)

# Select data for setosa and versicolor
setosa_data = iris_df.filter(iris_df["species"] == "Iris-setosa").select("sepal_length")
versicolor_data = iris_df.filter(iris_df["species"] == "Iris-versicolor").select("sepal_length")

# Convert PySpark DataFrame to Pandas DataFrame
setosa_pandas = setosa_data.toPandas()
versicolor_pandas = versicolor_data.toPandas()

# Calculate sample means and standard deviations using Pandas
mean_setosa = setosa_pandas["sepal_length"].mean()
mean_versicolor = versicolor_pandas["sepal_length"].mean()
std_setosa = setosa_pandas["sepal_length"].std()
std_versicolor = versicolor_pandas["sepal_length"].std()

# Population standard deviation (theoretical scenario)
population_std = 1.0

# Sample sizes
n_setosa = len(setosa_pandas)
n_versicolor = len(versicolor_pandas)

# Z-test
z_score = (mean_setosa - mean_versicolor) / (population_std / (n_setosa**0.5))

# Two-tailed test at 0.05 significance level
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Z-score: {z_score}")
print(f"P-value: {p_value}")

# Check if the null hypothesis is rejected
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in sepal length between setosa and versicolor.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in sepal length between setosa and versicolor.")

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

spark.stop()