from math import sqrt
from numpy import array
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ngrams", help="some useful description.")
args = parser.parse_args()

if args.ngrams:
    ngrams = args.ngrams

ngrams_value = int(2)
print(ngrams_value)


if ngrams_value <= 3:
    node_count = 1
elif 3 < ngrams_value <= 6:
    node_count = 2
elif 6 < ngrams_value <= 9:
    node_count = 3
else:
    raise ValueError("ngrams değeri 1 ile 9 arasında olmalıdır.")



spark = SparkSession.builder.appName("CountEntriesByClass") \
   .master("local") \
   .config("spark.executor.memory", "2g") \
   .config("spark.executor.memoryOverhead", "1g") \
   .getOrCreate()

Path = "events_small.csv" #hdfs:///customers.csv
customers = spark.read.format("csv").load(Path , header=True, inferSchema=True)

features_lst = ['View Count(scaled)', 'Purchase Count(scaled)']

vector_assembler = VectorAssembler(inputCols=features_lst,
                                             outputCol='feature_vector',handleInvalid="error")

customers = vector_assembler.transform(customers)
data_size = 100
# customers 1/3 ile işlem
if ngrams_value in [1,4,7]:
	customers, discard = customers.randomSplit([0.1 , 0.9], 24)
	data_size = 10
elif ngrams_value in [2,5,8]:
# customers 2/3 ile işlem
	customers, discard = customers.randomSplit([0.6 , 0.4], 24)
	data_size = 60
# Tamamı için üstteki iki satırı da yorum satırına al

customers_rdd = customers.rdd
parsed_rdd = customers_rdd.map(lambda row: row.feature_vector).map(lambda x: list(x))

start_time = time.time()
model = KMeans.train(rdd = parsed_rdd, k = 3, maxIterations=10000)
end_time = time.time()

model_name = f"kmeans_model_k=3_{end_time}"
model.save(spark, model_name)

# Training time hdfs'e yazdırılacak
training_time = end_time - start_time

# customers = customers.toPandas()

# predictions = model.predict(parsed_rdd).collect()
# customers["cluster"] = predictions

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = model.centers[model.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsed_rdd.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

computeCost = model.computeCost(parsed_rdd)
print("Within Set Sum of Squared Errors = " + str(computeCost))


current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# with open("output_file.txt", 'a') as file:
#     file.write(f"Algoritma: KMeans, Veri Büyüklüğü: %{data_size}, Node Sayısı: {node_count}, Çalışma Süresi: {training_time:.4f} saniye, Weights: -, WSSSE: {WSSSE}, ComputeCost: {computeCost}, Çalışma Tarihi: {current_datetime}\n")

# ---ekrana yazılacaklar---
# training_time, WSSSE, computeCost

# ---plot yazdırma---
x_values = [vector[0] for vector in model.clusterCenters]  # x değerleri (ilk eleman)
y_values = [vector[1] for vector in model.clusterCenters]  # y değerleri (ikinci eleman)
plt.figure(figsize=(8,8))
plt.title("Clustering Users by their Activities")
sns.scatterplot(x=customers["View Count(scaled)"], y=customers["Purchase Count(scaled)"], hue=customers['cluster'],palette="deep")
plt.scatter(x_values, y_values, marker='o', s=100, c='red', label='Cluster Centers')
plt.legend()
plt.savefig(model_name + '.png', dpi=300, bbox_inches='tight')
plt.show()


