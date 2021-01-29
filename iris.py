import sys
import tempfile
import requests

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Nice way to write a tmp file onto the system
    temp_csv_file = tempfile.mktemp()
    with open(temp_csv_file, mode="wb") as f:
        data_https = requests.get(
            "https://teaching.mrsharky.com/data/iris.data"
        )
        f.write(data_https.content)

    iris_df = spark.read.csv(temp_csv_file, inferSchema="true", header="true")
    iris_df = iris_df.toDF(
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class")

    iris_df.createOrReplaceTempView("iris")
    iris_df.persist(StorageLevel.DISK_ONLY)

    # Simple SQL
    results = spark.sql("SELECT * FROM iris")
    results.show()

    # Average for each of the 4
    average_overall = spark.sql(
        """
        SELECT
                AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM iris
        """
    )
    average_overall.show()

    # Average for each of the 4 by class
    average_by_class = spark.sql(
        """
        SELECT
                class
                , AVG(sepal_length) AS avg_sepal_length
                , AVG(sepal_width) AS avg_sepal_width
                , AVG(petal_length) AS avg_petal_length
                , AVG(petal_width) AS avg_petal_width
            FROM iris
            GROUP BY class
        """
    )
    average_by_class.show()

    # Add a new column
    iris_df = iris_df.withColumn("rand", rand(seed=42))
    iris_df.createOrReplaceTempView("iris")
    results = spark.sql("SELECT * FROM iris ORDER BY rand")
    results.show()

    return

if __name__ == "__main__":
    sys.exit(main())
