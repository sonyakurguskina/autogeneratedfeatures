import pandas as pd
import featuretools as ft

# Load CSV file
data = pd.read_csv('datasets/lintest.csv')

# Create an EntitySet
es = ft.EntitySet(id='data')
es = es.add_dataframe(
    dataframe_name="data",
    dataframe=data,
    index="y",
)

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                      groupby_trans_primitives=["cum_sum", "cum_count"],
                                      max_depth=1)

print(feature_matrix.head())
