from pyathena import connect
import pandas as pd

conn = connect(
    s3_staging_dir="s3://bucket2-curated-stock/athena_staging/",
    region_name="us-east-2"
)

df = pd.read_sql(
    "SELECT * FROM stock_lakehouse.stock_iceberg_v2 LIMIT 10",
    conn
)

print(df)