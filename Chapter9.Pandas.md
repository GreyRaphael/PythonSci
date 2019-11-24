# Pandas

- [Pandas](#pandas)
  - [Series](#series)
  - [DataFrame](#dataframe)
  - [reindex](#reindex)
  - [NAN](#nan)
  - [multiindex](#multiindex)
  - [map](#map)
  - [replace](#replace)
  - [bins](#bins)


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

pandas indexing

```py
# some function
df0.shape
df0.head() # first 5 records
df0.head(10) # first 10 records
df0.tail() # last 5 records
df0.to_records(index=False) # to list of tuples

# indexing dataframe
df0['Ratings']
df0[['Ratings', 'Nov 2019']]
df0[['Ratings', 'Nov 2019']][10:15]
df0.iloc[10:15,4:6]
df0.loc[10:15, ['Ratings', 'Nov 2019']] 
df0.loc[10:15, :"Ratings"] 
```

example: iloc vs loc

```py
df1=df0[10:20]
# Nov 2019 	Nov 2018 	Change 	Programming Language 	Ratings 	Change.1
# 10 	11 	16 	NaN 	Ruby 	1.261% 	+0.17%
# 11 	12 	11 	NaN 	Objective-C 	1.195% 	-0.28%
# 12 	13 	13 	NaN 	Delphi/Object Pascal 	1.142% 	-0.28%
# 13 	14 	25 	NaN 	Groovy 	1.099% 	+0.50%
# 14 	15 	15 	NaN 	Assembly language 	1.022% 	-0.09%
# 15 	16 	14 	NaN 	R 	0.980% 	-0.43%
# 16 	17 	20 	NaN 	Visual Basic 	0.957% 	+0.10%
# 17 	18 	23 	NaN 	D 	0.927% 	+0.25%
# 18 	19 	17 	NaN 	MATLAB 	0.890% 	-0.14%
# 19 	20 	10 	NaN 	Go 	0.853% 	-0.64%

# 严格根据index索引
df1.iloc[5:8] 
#  	Nov 2019 	Nov 2018 	Change 	Programming Language 	Ratings 	Change.1
# 15 	16 	14 	NaN 	R 	0.980% 	-0.43%
# 16 	17 	20 	NaN 	Visual Basic 	0.957% 	+0.10%
# 17 	18 	23 	NaN 	D 	0.927% 	+0.25%

# 根据label索引
df1.loc[15:17]
#  	Nov 2019 	Nov 2018 	Change 	Programming Language 	Ratings 	Change.1
# 15 	16 	14 	NaN 	R 	0.980% 	-0.43%
# 16 	17 	20 	NaN 	Visual Basic 	0.957% 	+0.10%
# 17 	18 	23 	NaN 	D 	0.927% 	+0.25%
```

## reindex

Series reindex

```py
import numpy as np
import pandas as pd

s0=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# a    1
# b    2
# c    3
# d    4
# dtype: int64
s1=s0.reindex(index=['a', 'b', 'C', 'D', 'E'], fill_value=10)
# a     1
# b     2
# C    10
# D    10
# E    10
# dtype: int64
s1.reindex(index=['a', 'b'])
# a     1
# b     2
# dtype: int64
s1.drop('a')
# b     2
# C    10
# D    10
# E    10
# dtype: int64

s2=pd.Series(['a', 'b', 'c'], index=[1, 5, 10])
# 1     a
# 5     b
# 10    c
# dtype: object

s3=s2.reindex(index=range(12), method='ffill')
# 0     NaN
# 1       a
# 2       a
# 3       a
# 4       a
# 5       b
# 6       b
# 7       b
# 8       b
# 9       b
# 10      c
# 11      c
# dtype: object
```

DataFrame reindex

```py
import numpy as np
import pandas as pd

df0=pd.DataFrame(np.random.rand(16).reshape(4, 4), index=['a', 'b', 'd', 'e'], columns=['c1', 'c2', 'c3', 'c4'])
#     c1 	c2 	c3 	c4
# a 	0.819308 	0.079193 	0.924312 	0.866187
# b 	0.148369 	0.459822 	0.059087 	0.812420
# d 	0.869805 	0.124662 	0.825912 	0.922507
# e 	0.679514 	0.325763 	0.709922 	0.144940
df1=df0.reindex(index=['a', 'b', 'c', 'd', 'e'])
df2=df0.reindex(columns=['c1', 'c2', 'c3', 'c4', 'c5'])
df3=df1=df0.reindex(index=['a', 'b', 'c', 'd', 'e'], columns=['c1', 'c2', 'c3', 'c4', 'c5'])

df3.reindex(index=['a', 'd', 'e'])
df3.reindex(columns=['c1', 'c2', 'c4'])
df3.drop(index=['a', 'b'], axis=0)
df3.drop(columns=['c1', 'c2'], axis=1)
```

## NAN

> tip: jupyter中`Shift+Tab`查看docstring

```py
import numpy as np
import pandas as pd

a=np.nan
type(a) # float
a+1 # nan

# Series with nan
s0=pd.Series([1, 2, np.nan, 4])
# 0    1.0
# 1    2.0
# 2    NaN
# 3    4.0
# dtype: float64
s0.isnull()
s0.notnull()
s0.dropna() # 不改变s0

# DataFrame with nan
df0=pd.DataFrame([[1, 2, 3], [5, np.nan,6], [7, np.nan, np.nan]])
# 	0 	1 	2
# 0 	1 	2.0 	3.0
# 1 	5 	NaN 	6.0
# 2 	7 	NaN 	NaN
df0.isnull()
df0.notnull()
df0.dropna()
df0.dropna(axis=0)
df0.dropna(axis=1)
df0.dropna(axis=0, how='all')

# nan数目>=2,会被删除
df0.dropna(thresh=2, axis=1)
df0.fillna(value=1)

# column 0 fill with 0, column 1 fill with 11, column 2 fill with 22
df0.fillna(value={0:0, 1:11, 2:22})
#  	0 	1 	2
# 0 	1 	2.0 	3.0
# 1 	5 	11.0 	6.0
# 2 	7 	11.0 	22.0
```

## multiindex

多级index让Series具有二维结构

```py
import numpy as np
import pandas as pd

s0=pd.Series(np.random.randn(6), index=[['1', '1', '1', '2', '2', '2'], ['a', 'b', 'c', 'a', 'b', 'c']])
# 1  a    2.003880
#    b    0.102215
#    c    0.587213
# 2  a   -1.305691
#    b   -0.205484
#    c    0.964941
# dtype: float64
s0['1']['c']
s0[:,'a']
df0=s0.unstack()
# 	a 	b 	c
# 1 	2.003880 	0.102215 	0.587213
# 2 	-1.305691 	-0.205484 	0.964941
# or
df1=pd.DataFrame([s0['1'], s0['2']])

s2=df1.unstack()
# a  0    2.003880
#    1   -1.305691
# b  0    0.102215
#    1   -0.205484
# c  0    0.587213
#    1    0.964941
# dtype: float64
s3=df1.T.unstack()
# 0  a    2.003880
#    b    0.102215
#    c    0.587213
# 1  a   -1.305691
#    b   -0.205484
#    c    0.964941
# dtype: float64

# 多级index的DataFrame
df2=pd.DataFrame(np.random.randn(16).reshape(4, 4), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['BJ', 'BJ', 'SH', 'SZ'], [8, 9, 8, 8]])
#  		BJ 	SH 	SZ
# 		8 	9 	8 	8
# a 	1 	-1.303202 	-0.884724 	0.830207 	-0.274237
# 2 	0.772264 	-1.293091 	0.550154 	0.239424
# b 	1 	0.635577 	-0.184073 	1.162522 	-0.264281
# 2 	0.578760 	-0.111816 	-2.348368 	0.302795
df2['BJ']
```

## map

```py
import numpy as np
import pandas as pd

df0=pd.DataFrame({'city':['BJ', 'SH', 'SZ'], 'population':[1000, 2000, 1500]})
# 	city 	population
# 0 	BJ 	1000
# 1 	SH 	2000
# 2 	SZ 	1500

gdp_map={'BJ':2, 'SH': 10, 'SZ':8}
df0['GDP']=df0['city'].map(gdp_map)

# 仅仅适用于没有指定index的情形
df0['GDP']=pd.Series([20, 100, 80])

df1=pd.DataFrame({'city':['BJ', 'SH', 'SZ'], 'population':[1000, 2000, 1500]}, index=['a', 'b', 'c'])
# 必须指定index,否则全是NaN
df1['GDP']=pd.Series([20, 100, 80], index=['a', 'b', 'c'])
```

## replace

```py
import numpy as np
import pandas as pd

s0=pd.Series(range(10))
s0.replace(1, 11)
s0.replace({2:22, 3:33})
s0.replace([5, 6, 7], [55, 66, 77])
```

## bins

```py
import numpy as np
import pandas as pd

score_list=np.random.randint(100, size=50)
# (0, 59], (59, 70], (70, 80], (80, 100]
bins=[0, 59, 70, 80, 100]
score_cut=pd.cut(score_list, bins)
pd.value_counts(score_cut)


df=pd.DataFrame()
df['score']=score_list
df['student']=[pd.util.testing.rands(4) for _ in range(50)]

df['category']=pd.cut(df['score'], bins, labels=['low', 'ok', 'good', 'great'])
```