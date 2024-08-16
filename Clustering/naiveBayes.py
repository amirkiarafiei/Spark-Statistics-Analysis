
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
from pyspark.sql.functions import col

# Spark Oturumu Başlatma
spark = SparkSession.builder.appName("NaiveBayes") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Veri Setlerini Okuma
events = spark.read.csv("events_small.csv", header=True, inferSchema=True)

# Zaman özellikleri oluşturma fonksiyonu
def create_time_features(data):
    print("\nHERE2\n")
    # timestamp sütununu Unix zaman damgasından timestamp türüne dönüştürme
    data = data.withColumn("timestamp", F.from_unixtime(F.col("timestamp")))
    data = data.withColumn("hour", F.hour(F.col("timestamp")))
    data = data.withColumn("day", F.dayofmonth(F.col("timestamp")))
    data = data.withColumn("month", F.month(F.col("timestamp")))
    data = data.withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
    return data

# VectorAssembler kullanarak features sütununu oluşturma fonksiyonu
def create_features_column(data):
    assembler = VectorAssembler(inputCols=["hour", "day", "month", "day_of_week", "visitorid", "itemid"],
                                outputCol="features")
    return assembler.transform(data)

# StringIndexer kullanarak label sütununu dönüştürme fonksiyonu
def index_label_column(data):
    indexer = StringIndexer(inputCol
                            ="event", outputCol="label")
    return indexer.fit(data).transform(data)

# Naive Bayes modeli
def naive_bayes():
    start_time = time.time()

    # Zaman özelliklerini ekleme
    events_with_time_features = create_time_features(events)

    # Label sütununu dönüştürme ve features sütununu oluşturma
    nb_data = create_features_column(index_label_column(events_with_time_features))

    # Naive Bayes modelini eğitme
    (train_data_nb, test_data_nb) = nb_data.randomSplit([0.8, 0.2], seed=42)
    nb = NaiveBayes(labelCol="label", featuresCol="features", smoothing=0.5)
    nb_model = nb.fit(train_data_nb)

    print("\nHERE6\n")
    # Naive Bayes modelinin performansını değerlendirme
    nb_predictions = nb_model.transform(test_data_nb)
    # print(nb_predictions.select("prediction", "label").show(5))
    nb_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                         metricName="accuracy")

    nb_accuracy = nb_evaluator.evaluate(nb_predictions)
    print(f"Accuracy: {nb_accuracy}")
    end_time = time.time()
    # predictionAndLabels = nb_predictions.select("prediction", "label").rdd


    # predictionAndLabels = nb_predictions.select("prediction", "label").rdd.map(lambda row: (row.prediction, row.label))
    predictionAndLabels = nb_predictions.select("prediction", "label").rdd
    print("\nHERE7\n")

    metrics = MulticlassMetrics(predictionAndLabels)
    print("\nHERE8\n")

    # Confusion Matrix
    confusion_matrix = metrics.confusionMatrix().toArray()

    # Diğer metrikler
    indexed = index_label_column(events)
    num_classes = indexed.select("label").distinct().count()

    # Precision, Recall ve F1-Score hesaplaması
    precision = []
    recall = []
    f1Score = []

    for label in range(num_classes):
        precision.append(metrics.precision(float(label)))
        recall.append(metrics.recall(float(label)))
        f1Score.append(metrics.fMeasure(float(label)))

    print("\nHERE9\n")

    # Precision, Recall ve F1-Score sonuçlarını yazdırma
    for label in range(num_classes):
        print(f"Class {label} - Precision: {precision[label]}, Recall: {recall[label]}, F1-Score: {f1Score[label]}")

    print("Linear SVM modeli doğruluğu:", nb_accuracy)
    print("Çalışma süresi:", end_time - start_time, "saniye")
    print("Confusion Matrix:\n", confusion_matrix)

    return confusion_matrix, nb_accuracy, (end_time - start_time)

confusion_matrix2 = naive_bayes()
print("Çıkış yapılıyor...")

print("\nHERE10\n")
# Spark Oturumu Kapatma
spark.stop()
print(confusion_matrix2)

print("\nHERE11\n")
