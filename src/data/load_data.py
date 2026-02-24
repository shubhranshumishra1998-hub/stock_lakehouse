from pyspark.sql import SparkSession
from pyspark.sql.function import *
from pyspark.sql.type import *



df= spark.read.format("parquet").load(