from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from scipy.stats import ttest_ind
import time



# Create a Spark session
spark = SparkSession.builder.appName("TTestExample") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Start time
start_time = time.time()

# Define the schema
schema = StructType([
    StructField("SepalLengthCm", FloatType(), True),
    StructField("SepalWidthCm", FloatType(), True),
    StructField("PetalLengthCm", FloatType(), True),
    StructField("PetalWidthCm", FloatType(), True),
    StructField("Species", StringType(), True)
])

# Read the CSV file with the specified schema
iris_data = spark.read.csv("iris10Kx.csv", header=False, schema=schema)

# Select SepalLengthCm and Species columns
selected_data = iris_data.select("SepalLengthCm", "Species")

# Create separate DataFrames for each species
setosa_data = selected_data.filter(col("Species") == "Iris-setosa").select("SepalLengthCm").rdd.flatMap(lambda x: x)
versicolor_data = selected_data.filter(col("Species") == "Iris-versicolor").select("SepalLengthCm").rdd.flatMap(lambda x: x)

# Check if there is variation in the data (non-zero standard deviation)
std_dev_setosa = setosa_data.stdev()
std_dev_versicolor = versicolor_data.stdev()

# Perform t-test only if there is variation in both groups
if std_dev_setosa != 0 and std_dev_versicolor != 0:
    # Perform t-test
    t_stat, p_value = ttest_ind(setosa_data.collect(), versicolor_data.collect())

    # Set significance level (e.g., 0.05 for 95% confidence interval)
    significance_level = 0.05

    # Compare p-value with significance level
    if p_value < significance_level:
        print(f"P-value: {p_value}")
        print("Reject the null hypothesis. Means are significantly different.")
    else:
        print(f"P-value: {p_value}")
        print("Fail to reject the null hypothesis. Means are not significantly different.")
else:
    print("No variation in one or both groups. Unable to perform t-test.")

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop Spark session
spark.stop()
