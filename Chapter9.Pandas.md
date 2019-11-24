# Pandas

- [Pandas](#pandas)
  - [Series](#series)


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