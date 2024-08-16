from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
import time



# Create a Spark session
spark = SparkSession.builder.appName("ChiSquareTestExample") \
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
    StructField("species", StringType(), True)
])

# Load the Iris dataset with the specified schema
iris_data = spark.read.csv("../../iris100Kx.csv", header=False, schema=iris_schema)

# Convert species labels to numerical indices
class_indexer = StringIndexer(inputCol="species", outputCol="label")
indexed_data = class_indexer.fit(iris_data).transform(iris_data)

# Assemble features and label into a single feature vector
feature_columns = ["petal_width"]
assembler = VectorAssembler(inputCols=feature_columns + ["label"], outputCol="features")
assembled_data = assembler.transform(indexed_data)

# Apply ChiSquareTest
chi_sq_test_result = ChiSquareTest.test(assembled_data, "features", "label").head()

# Display the ChiSquareTest result
print("ChiSquareTest Result:")
print("pValues: " + str(chi_sq_test_result.pValues))
print("degreesOfFreedom: " + str(chi_sq_test_result.degreesOfFreedom))
print("statistics: " + str(chi_sq_test_result.statistics))

# Interpretation of the result
p_value = chi_sq_test_result.pValues[0]
alpha = 0.05  # significance level

print("\nInterpretation:")
print(f"The p-value is {p_value:.4f}.")

if p_value < alpha:
    print("The p-value is less than the significance level (alpha).")
    print("Reject the null hypothesis.")
    print("There is evidence of an association between species and petal width.\n\n")
else:
    print("The p-value is greater than or equal to the significance level (alpha).")
    print("Fail to reject the null hypothesis.")
    print("There is no significant evidence of an association between species and petal width.\n\n")



# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop Spark session
spark.stop()
