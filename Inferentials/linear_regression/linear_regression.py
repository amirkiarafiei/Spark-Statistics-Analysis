from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import time



# Create a Spark session
spark = SparkSession.builder.appName("IrisLinearRegression") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Start time
start_time = time.time()

# Define the schema for the Iris dataset
schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("class", StringType(), True)
])

# Load the Iris dataset with the defined schema
iris_data = spark.read \
    .format("csv") \
    .option("header", "false") \
    .schema(schema) \
    .load("../../iris100Kx.csv")  # Replace with the actual path to your Iris dataset

# Prepare the data for linear regression
feature_columns = ["sepal_width", "petal_length", "petal_width"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = vector_assembler.transform(iris_data).select("features", "sepal_length")

# Split the data into training and testing sets
(training_data, test_data) = assembled_data.randomSplit([0.8, 0.2], seed=1234)

# Create a linear regression model
lr = LinearRegression(labelCol="sepal_length", featuresCol="features")

# Fit the model to the training data
lr_model = lr.fit(training_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Display the predicted sepal length alongside the actual sepal length
predictions.select("prediction", "sepal_length", "features").show()


# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

input("Press any key to stop Spark.")

# Stop the Spark session
spark.stop()
