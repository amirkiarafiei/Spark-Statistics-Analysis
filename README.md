# Statistics Analysis using Spark on Iris Dataset

## Overview

This project focuses on implementing descriptive and exploratory statistical functions using Apache Spark (PySpark) to analyze the performance when running on a cluster of nodes. The original Iris dataset and a 10K times enlarged version are utilized to simulate big data scenarios. The data is visualized using Apache Superset, providing insights into the statistical analysis results.

An academic report accompanies this repository, offering a comprehensive overview of the project. The report includes details on the workflow, different stages of the project, methodology, visualizations, and additional information. Performance analysis tables are also presented in the report.


## Configurations

Modify the following code snippet to configure the Spark cluster:
```python 
# Create a Spark session
spark = SparkSession.builder.appName("MeanCalculator") \
    .master("local[2]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "1g") \
    .getOrCreate()
```
Modify the code snippet below to choose the desired Iris dataset version:
```pyhton 
# Load the Iris dataset with the specified schema
iris_data = spark.read.csv("iris.csv", header=False, schema=iris_schema)
#iris_data = spark.read.csv("iris10Kx.csv", header=False, schema=iris_schema)

```

## Project Deployment
Step 1: Clone the repository to your local machine
```bash
git clone https://github.com/username/repo.git

```

Step 2: Activate the Conda environment using the provided .yml file
```bash
conda env create -f environment.yml
conda activate your-env
```

Step 3: Install Java and Spark on your machine. Set the necessary environment variables and update the PATH variable. For detailed instructions, refer to the [Spark official documentation](https://spark.apache.org/downloads.html)

## Superset and Visualization
To visualize the data, follow these steps:

Step 1: Clone the Superset repository to your local machine
```bash
git clone https://github.com/apache/superset.git
```

Step 2: Run Superset on Docker containers using the following commands

```bash
cd superset
sudo docker compose build
sudo docker compose up
```

Step 3 (Optional): Use the Portainer tool or Docker commands to check the status of running containers. You can access Portainer through web browser at localhost:9000.

Step 4: Log in to Superset Dashboard and create charts.
Access the Superset dashboard through your web browser at localhost:8088.
Create desired charts by connecting Superset to an Iris database (I used PostgreSQL DB). Follow the [Superset documentation](https://superset.apache.org/docs/intro) for detailed instructions.


## Acknowledgement
Special thanks to Prof. Mehmet Sıddık AKTAŞ for supervising this project.

## License
This project is licensed under the MIT License.

