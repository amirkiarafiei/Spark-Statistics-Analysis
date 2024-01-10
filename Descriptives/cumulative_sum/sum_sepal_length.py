from pyspark import SparkContext, SparkConf


# Create a Spark context
conf = SparkConf().setAppName("CumulativeSum")
sc = SparkContext(conf=conf)

# Replace 'your_dataset' with your actual dataset
data = sc.textFile("iris.csv")

# Assuming the class column is the fifth column (index 4)
class_index = 4
feature_index = 0

# Map Phase: Parse and emit key-value pairs
mapped_data = data.map(lambda line: (
    line.split(',')[class_index],  # Key is the class
    float(line.split(',')[feature_index])  # Value is the first column value as float
))

# Reduce Phase: Calculate cumulative sum for each class
cumulative_sums = mapped_data.reduceByKey(lambda x, y: x + y)

# Collect and display results
results = cumulative_sums.collect()
for key, cumulative_sum in results:
    print(f"Class: {key}, Cumulative Sum for the Sepal Length: {cumulative_sum}")

# Stop Spark
sc.stop()
