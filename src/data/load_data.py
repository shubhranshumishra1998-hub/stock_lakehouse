import pandas as pd
from pyathena import connect

def load_from_athena(query, database):
    conn = connect(region_name="us-east-2")
    return pd.read_sql(query, conn)
