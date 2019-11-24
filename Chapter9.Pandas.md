# Pandas

- [Pandas](#pandas)
  - [Series](#series)
  - [DataFrame](#dataframe)


## Series

```py
import numpy as np
import pandas as pd

s1=pd.Series([10, 20, 3., 40])
# 0    10.0
# 1    20.0
# 2     3.0
# 3    40.0
# dtype: float64

s1.values
# array([10., 20.,  3., 40.])
s2.index
# RangeIndex(start=0, stop=4, step=1)

s2=pd.Series(np.arange(10))
s2[s2>5]
# 6    6
# 7    7
# 8    8
# 9    9
# dtype: int64

s3=pd.Series({'1':10, '2':20, '3':30})
# 1    10
# 2    20
# 3    30
# dtype: int64
s3.values
# array([10, 20, 30])
s3.index
# Index(['1', '2', '3'], dtype='object')

s4=pd.Series([11, 22, 33, 44], index=['A', 'B', 'C', 'D'])
# A    11
# B    22
# C    33
# D    44
# dtype: int64
s4['B']
# 22
s4.to_dict()
# {'A': 11, 'B': 22, 'C': 33, 'D': 44}

s5=pd.Series(s4, index=['A', 'B', 'C', 'D', 'E'])
# A    11.0
# B    22.0
# C    33.0
# D    44.0
# E     NaN
# dtype: float64
pd.isnull(s5)
# A    False
# B    False
# C    False
# D    False
# E     True
# dtype: bool
pd.notnull(s5)
# A     True
# B     True
# C     True
# D     True
# E    False
# dtype: bool
s5.name='demo'
# A    11.0
# B    22.0
# C    33.0
# D    44.0
# E     NaN
# Name: demo, dtype: float64
s5.index.name='grey'
# grey
# A    11.0
# B    22.0
# C    33.0
# D    44.0
# E     NaN
# Name: demo, dtype: float64
```

## DataFrame

[dataframe io](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)

| Format Type | Data Description     | Reader         | Writer       |
|-------------|----------------------|----------------|--------------|
| text        | CSV                  | read_csv       | to_csv       |
| text        | JSON                 | read_json      | to_json      |
| text        | HTML                 | read_html      | to_html      |
| text        | Local clipboard      | read_clipboard | to_clipboard |
| binary      | MS Excel             | read_excel     | to_excel     |
| binary      | OpenDocument         | read_excel     |              |
| binary      | HDF5 Format          | read_hdf       | to_hdf       |
| binary      | Feather Format       | read_feather   | to_feather   |
| binary      | Parquet Format       | read_parquet   | to_parquet   |
| binary      | Msgpack              | read_msgpack   | to_msgpack   |
| binary      | Stata                | read_stata     | to_stata     |
| binary      | SAS                  | read_sas       |              |
| binary      | Python Pickle Format | read_pickle    | to_pickle    |
| SQL         | SQL                  | read_sql       | to_sql       |
| SQL         | Google Big Query     | read_gbq       | to_gbq       |

```py
import numpy as np
import pandas as pd

# read all table form TOIBE
df0=pd.read_html('https://www.tiobe.com/tiobe-index/')[0]

type(df0)
# pandas.core.frame.DataFrame
df0
# ...

df0.columns
# Index(['Nov 2019', 'Nov 2018', 'Change', 'Programming Language', 'Ratings', 'Change.1'],dtype='object')

# 如果column name无空格
df0.Ratings.values
# array([...])
# 如果column name有空格
df0['Nov 2019'].values1
# array([...])

type(df0['Nov 2019'])
# pandas.core.series.Series
df_new=pd.DataFrame(df0, columns=['Programming Language', 'Ratings','Change.1'])
# ...

# add new column "Share"
df_new2=pd.DataFrame(df0, columns=['Programming Language', 'Ratings','Change.1', 'Share'])
# 3 methods to set column
df_new2['Share']=range(20)
# df_new2['Share']=np.arange(20)
# df_new2['Share']=pd.Series(np.arange(20))

# modify column(not recommended)
df_new2['Share'][3:5]=[33, 44]
# modify column(recommended)
df_new2['Share'].values[3:5]=[33,44]

# replace whole column
df_new2['Share']=pd.Series([55, 66], index=[5, 6])
```