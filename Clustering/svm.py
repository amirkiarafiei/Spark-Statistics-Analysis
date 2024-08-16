
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import time

# Spark Oturumu Başlatma
spark = SparkSession.builder.appName("CountEntriesByClass") \
    .master("local[3]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()

# Veri Setlerini Okuma
events = spark.read.csv("events_demo.csv", header=True, inferSchema=True)

# Zaman özellikleri oluşturma fonksiyonu
def create_time_features(data):
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
    indexer = StringIndexer(inputCol="event", outputCol="label")
    return indexer.fit(data).transform(data)

    # Linear SVM modeli
def linear_svc_optimized():
    start_time = time.time()

    # Zaman özelliklerini ekleme
    events_with_time_features = create_time_features(events)

    # Label sütununu dönüştürme ve features sütununu oluşturma
    svm_data = create_features_column(index_label_column(events_with_time_features))

    # Özelliklerin ölçeklendirilmesi
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    scaler_model = scaler.fit(svm_data)
    svm_data = scaler_model.transform(svm_data).cache()  # Önbelleğe alma (kısa sürmesi için)

    # Linear SVM modelini eğitme
    (train_data_svm, test_data_svm) = svm_data.randomSplit([0.8, 0.2], seed=42)
    lsvc = LinearSVC(maxIter=1, regParam=0.01, featuresCol="scaledFeatures", labelCol="label")
    ovr_classifier = OneVsRest(classifier=lsvc)
    ovr_model = ovr_classifier.fit(train_data_svm)

    # Linear SVM modelinin performansını değerlendirme
    svm_predictions = ovr_model.transform(test_data_svm)
    svm_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                          metricName="accuracy")
    svm_accuracy = svm_evaluator.evaluate(svm_predictions)
    end_time = time.time()
    predictionAndLabels = svm_predictions.select("prediction", "label").rdd
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

    print("Linear SVM modeli doğruluğu:", svm_accuracy)
    print("Çalışma süresi:", end_time - start_time, "saniye")
    print("Confusion Matrix:\n", confusion_matrix)

    return confusion_matrix, svm_accuracy, (end_time - start_time)

confusion_matrix2 = linear_svc_optimized()
print("Çıkış yapılıyor...")

# Spark Oturumu Kapatma
spark.stop()
print(confusion_matrix2)
