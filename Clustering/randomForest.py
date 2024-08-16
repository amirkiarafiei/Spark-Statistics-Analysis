
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
import time

# Spark Oturumu Başlatma
spark = SparkSession.builder.appName("RandomForest") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Veri Setlerini Okuma
events = spark.read.csv(f"events_demo.csv", header=True, inferSchema=True)

# VectorAssembler kullanarak features sütununu oluşturma fonksiyonu
def create_features_column(data):
    assembler = VectorAssembler(inputCols=["timestamp", "visitorid", "itemid"], outputCol="features")
    return assembler.transform(data)

    # StringIndexer kullanarak label sütununu dönüştürme fonksiyonu
def index_label_column(data):
    indexer = StringIndexer(inputCol="event", outputCol="label")
    return indexer.fit(data).transform(data)

    # Random Forest modeli
def randomForest():
    start_time = time.time()
    rf_data = create_features_column(index_label_column(events))

    # Random Forest modelini eğitme
    (train_data_rf, test_data_rf) = rf_data.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    rf_model = rf.fit(train_data_rf)

    # Random Forest modelinin performansını değerlendirme
    rf_predictions = rf_model.transform(test_data_rf)
    rf_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                         metricName="accuracy")
    rf_accuracy = rf_evaluator.evaluate(rf_predictions)
    end_time = time.time()

    predictionAndLabels = rf_predictions.select("prediction", "label").rdd
    metrics = MulticlassMetrics(predictionAndLabels)

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

    # Precision, Recall ve F1-Score sonuçlarını yazdırma
    for label in range(num_classes):
        print(f"Class {label} - Precision: {precision[label]}, Recall: {recall[label]}, F1-Score: {f1Score[label]}")

    print("Random Forest modeli doğruluğu:", rf_accuracy)
    print("Çalışma süresi:", end_time - start_time, "saniye")
    print("Confusion Matrix:\n", confusion_matrix)

    return confusion_matrix, rf_accuracy, (end_time - start_time)

confusion_matrix2 = randomForest()
print("Çıkış yapılıyor...")

# Spark Oturumu Kapatma
spark.stop()
print(confusion_matrix2)
