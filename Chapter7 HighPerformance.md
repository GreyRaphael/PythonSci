# High Performance

<!-- TOC -->

- [High Performance](#high-performance)
    - [Jupyter](#jupyter)
        - [Jupyter设置远程访问](#jupyter设置远程访问)
    - [IPython](#ipython)
        - [IPython Magic Commands](#ipython-magic-commands)
    - [numpy数据类型系统](#numpy数据类型系统)
    - [Cython](#cython)
        - [Install Cython on windows](#install-cython-on-windows)
        - [use cython](#use-cython)
        - [Compare Optimization](#compare-optimization)
            - [recursive fibonacci optimization](#recursive-fibonacci-optimization)
            - [iteration fibonacci optimization](#iteration-fibonacci-optimization)
            - [cached fibonacci optimization](#cached-fibonacci-optimization)
            - [summary](#summary)
        - [numpy optimization](#numpy-optimization)
    - [`ipyparallel` module](#ipyparallel-module)

<!-- /TOC -->

## Jupyter

Jupyter是**交互执行框架**, 通过不同的[kernel](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)执行不同的编程语言; 主要支持的语言: Python, R, Julia, C++;
> 对于Python语言的kernel是IPython;

### Jupyter设置远程访问

```bash
C:\ProgramData\Anaconda3\Scripts\jupyter notebook --generate-config
# 生成配置文件: ~/.jupyter/jupyter_notebook_config.py

C:\ProgramData\Anaconda3\Scripts\jupyter notebook password
# 生成加密的密码: ~/.jupyter/jupyter_notebook_config.json
```

```python
# jupyter_notebook_config.py
c.NotebookApp.ip = '' 
# 阿里云需要填写 私有ip
# 将jupyter_notebook_config.json里面的密码复制到这
c.NotebookApp.password = "sha1:9ba3e4304a4a:d322d173f6f43a74e3445c75aa02c218f2759a1c"
c.NotebookApp.open_browser = False
```

> Jupyter notebook, JupyterLab是基于浏览器的服务;  
> 在浏览器输入`222.29.69.149:8888`配合密码就可以访问

Linux后台运行: 不采用service

```bash
# being jupyter
nohup jupyter-notebook ~/JupyterWork/ &
# terminate jupyter
lsof nohup.out
kill -9 <PID>
```

## IPython

IPython因为有宏命令机制(**魔法命令**)比原生的Python Shell好用; `ipyparallel`可以简单实现多核并行计算;
> `conda install ipyparallel`

### IPython Magic Commands

- `%xxx`: line magic, 只对一行作用;
- `%%xxx`: cell magic, 对整个cell作用
- `!xxx`: 调用系统(window, linux)的命令, 不同系统命令不同

Jupyter中可以省略`%`不能省略`%%, !`; 为了规范起见，都加上;
> `pylab`是`numpy`与`matplotlib`的合并;

常用`!`:
- `!ls`: for linux; `!dir`: for windows; 为了通用一般被`%ls`代替;

常用`%`, `%%`: 
- `%lsmagic`: list all magics
- `%xxx?`, `%%xxx?`: 该magic的help
- `%pwd`, `%cd`
- `%matplotlib`: 一般采用`%matplotlib inline`才能在第一行输出图片
    ```python
    # 这一行保证了如下内容在同一个cell, 也能显示图片
    %matplotlib inline

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['figure.dpi'] = 100

    x=np.linspace(-2*np.pi, 2*np.pi, 200)
    y1=np.sin(x)/x
    y2=np.sin(x)**2

    fig = plt.figure(figsize=(8, 3))
    ax1=fig.add_subplot(121)
    ax1.plot(x,y1)
    ax2=fig.add_subplot(122)
    ax2.plot(x,y2)

    # 既可以用plt.savafig(), 也可以用fig.savefig();多个图片分别用fig.savefig()
    # savefig的dpi一般与figure.dpi相同; pdf, svg是矢量图,没有dpi的概念;
    fig.savefig('test.png', dpi=150)
    fig.savefig('test.svg')
    ```
- `%%capture`: 捕捉stdout
    ```python
    # first cell, 将cell的输出保存到captured_data
    %%capture captured_data
    print([1,2,3])
    ```
    ```python
    # second cell,给机器看的输出效果
    captured_data.stdout # '[1, 2, 3]\n'
    ```
    ```python
    # third cell, 给人看的输出效果
    captured_data.show() # [1, 2, 3]
    ```
- `%debug`, `%%debug`: 会在报错时进入pdb调试;

组合技:
- `%%writefile`
- `%load`: 只能是文本文件
- `%run`: 运行`.py`
    ```python
    # 1st cell: 写文件
    %%writefile test.py
    def fib(n):
        if n<2:
            return n
        return fib(n-1)+fib(n-2)

    print(fib(30))
    ```
    ```python
    # 2nd cell: 运行文件
    %run test.py
    ```
    ```python
    # 3rd cell, 运行并且计时
    %run -t test.py
    ```
    ```python
    # 4th cell: 读取文件到cell
    %load test.py
    ```
    ```python
    # 5th cell: timeit
    %timeit fib(30)
    ```

性能测试工具: [一般用法](https://blog.csdn.net/xiemanR/article/details/72763234)
- PyCharm自带的[Profiling](https://blog.csdn.net/xiemanr/article/details/69398057)
- `%timeit`, `%%timeit`: 多次运行时间平均值; 没有详细的统计;
- `%time`, `%%time`: 运行时间; window, linux显示的项目不同;
- `%prun`, `%%prun`: 使用python的built-in模块`cProfile`;
    ```python
    # cell
    def func1(num):
        return num**3

    def func2():
        sum=0
        for i in range(1000):
            sum+=func1(i)
        return sum

    %prun func2()
    ```
    ```bash
    # result
    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1000    0.002    0.000    0.002    0.000 <ipython-input-2-3306a652762e>:1(func1)
        1    0.001    0.001    0.003    0.003 <ipython-input-2-3306a652762e>:4(func2)
        1    0.000    0.000    0.003    0.003 {built-in method builtins.exec}
        1    0.000    0.000    0.003    0.003 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    # tottime: 除去sub-functions的总调用时间
    # percall: tottime/ncalls
    # cumtime: 包含sub-functions的总调用时间
    # percall: cumtime/ncalls
    # filename: linenumber(function)
    # 其中builtins.exec和<module>表示执行该段代码
    # 其中disable()表示Stop collecting profiling information.
    ```
    ```python
    # 递归特例
    def fib(n):
        if n<2:
            return n
        return fib(n-1)+fib(n-2)

    %prun fib(13)
    ```
    ```bash
    # 753/1分别表示sub-fuction和主调函数的ncalls; 只不过两者都是同一个函数所以这么写;
    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    753/1    0.001    0.000    0.001    0.001 <ipython-input-1-eaec75342a65>:1(fib)
        1    0.000    0.000    0.001    0.001 {built-in method builtins.exec}
        1    0.000    0.000    0.001    0.001 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    ```
- 还有一个纯python实现的与`cProfile`的api完全相同的`profile`没有被IPython采用;
- `%lprun`: 统计每行代码的执行次数和执行时间; `conda install line_profiler`
    ```python
    # cell
    def fib(n):
        if n<2:
            return n
        return fib(n-1)+fib(n-2)
    ```
    ```python
    # cell
    %load_ext line_profiler
    %lprun -f fib fib(20)
    ```
    ```bash
    # result
    Line      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        1                                           def fib(n):
        2     21891      36792.0      1.7     49.0      if n<2:
        3     10946      15157.0      1.4     20.2          return n
        4     10945      23063.0      2.1     30.7      return fib(n-1)+fib(n-2)
    ```
- `%memit`, `mprun`:
    ```python
    # cell
    def fib(n):
        if n<2:
            return n
        return fib(n-1)+fib(n-2)
    ```
    ```python
    # cell: memit
    %load_ext memory_profiler
    # 粗粒度内存检查
    %memit fib(20)
    # peak memory: 49.25 MiB, increment: 0.09 MiB
    ```
    ```python
    # mprun是对于文件操作的
    # cell
    %%writefile tmp.py
    import numpy as np
    def func():
        sum=np.arange(1000000)
        return sum
    ```
    ```python
    # cell
    from tmp import func
    # 细粒度内存检查
    %mprun -f func func()
    ```
    ```bash
    # result
    Filename: N:\Jupyter\tmp.py
    Line    Mem usage    Increment   Line Contents
    ==============================================
    2     58.7 MiB     58.7 MiB   def func():
    3     62.5 MiB      3.8 MiB       sum=np.arange(1000000)
    4     62.5 MiB      0.0 MiB       return sum
    ```

## numpy数据类型系统

其中`np.float`, `np.int`就是python原生的`float`, `int`的alias;不属于numpy数据系统;
> 原生的python没有`uint`

```python
# cell
import numpy as np
np.sctypes
# {'int': [numpy.int8, numpy.int16, numpy.int32, numpy.int64],
#  'uint': [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64],
#  'float': [numpy.float16, numpy.float32, numpy.float64],
#  'complex': [numpy.complex64, numpy.complex128],
#  'others': [bool, object, bytes, str, numpy.void]}
```

```python
# cell: int group for arange
a1=np.arange(10, dtype='int')
a2=np.arange(10, dtype='uint')
a3=np.arange(10, dtype='float')
a1.dtype, a2.dtype, a3.dtype
# (dtype('int32'), dtype('uint32'), dtype('float64'))
```

```python
# cell: int
b1=np.int(10)
b2=np.int8(10)
b3=np.int16(10)
b4=np.int32(10)
b5=np.int64(10)
type(b1), type(b2), type(b3), type(b4), type(b5)
# (int, numpy.int8, numpy.int16, numpy.int32, numpy.int64)
```

```python
# cell: uint
c1=np.uint(10)
c2=np.uint8(10)
c3=np.uint16(10)
c4=np.uint32(10)
c5=np.uint64(10)
type(c1), type(c2), type(c3), type(c4), type(c5)
# (numpy.uint32, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64)
```

```python
# cell: float
d1=np.float(10)
d2=np.float16(10)
d3=np.float32(10)
d4=np.float64(10)
type(d1), type(d2), type(d3), type(d4)
# (float, numpy.float16, numpy.float32, numpy.float64)
```

## Cython

CPython vs Cython:
- CPython: 底层是C实现的Python
- Cython: Cython是单独的一门语言，专门用来写在Python里面import用的扩展库。Cython的语法基本上跟Python一致，而Cython有专门的“编译器”先将 Cython代码转变成C（自动加入了一大堆的C-Python API），然后使用C编译器编译出最终的Python可调用的模块。

### Install Cython on windows

Method1: 使用的是mingw编译`.pyx`为`.pyd`

```bash
# Anaconda3; 大约200MB, 不推荐
conda install libpython m2w64-toolchain cython
```

Method2: 使用msvc编译`.pyx`为`.pyd`

```bash
# 安装Microsoft Visual C++ Build tools; 大约4G; 推荐
# https://visualstudio.microsoft.com/vs/older-downloads/
conda install cython
```

### use cython

```python
# 新建helloworld.pyx
print("hello, grey")
```

```python
# 新建setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("helloworld.pyx")
)
```

```bash
./
    helloworld.pyx
    setup.py
```

```bash
python setup.py build_ext --inplace
# 得到helloworld.c, helloworld.cp36-win_amd64.pyd
```

```bash
# 调用module
PS N:\Jupyter> python
>>> import helloworld
hello, grey
```

### Compare Optimization

在Jupyter中测试

#### recursive fibonacci optimization

递归fibonacci优化;

without optimization

```python
# cell
def fib(n):
    if n<2:
        return n
    return fib(n-1)+fib(n-2)

%timeit fib(20)
# 2.05 ms ± 14.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

numba optimization: 43.4倍

```python
# cell
# Optimization with numba: 43.4倍
from numba import jit

@jit
def fib_numba(n):
    if n<2:
        return n
    return fib_numba(n-1)+fib_numba(n-2)

%timeit fib_numba(20)
# 47.2 µs ± 58.8 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

cython optimization: 2.43倍

```python
# cell
%load_ext cython
```
```python
# cell
%%cython
def fib_cython(n):
    if n<2:
        return n
    return fib_cython(n-1)+fib_cython(n-2)
```
```python
# cell
%timeit fib_cython(20)
# 841 µs ± 3.39 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

cython optimization with type(使用了静态编译): 101.0倍

```python
# cell
%load_ext cython
```
```python
%%cython
# %%cython -a
# 显示cython与python interaction的地方;
cpdef long fib_cython_type(long n):
    if n<2:
        return n
    return fib_cython_type(n-1)+fib_cython_type(n-2)
```
```python
%timeit fib_cython_type(20)
# 20.3 µs ± 28 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

cache optimization: 24818.4倍

因为`return`是两个fibonacci递归, 所以第一个算好的, 缓存起来, 可以直接给第二个用;

`maxsize=None`表示缓存没有大小限制

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_cache(n):
    if n<2:
        return n
    return fib_cache(n-1)+fib_cache(n-2)

%timeit fib_cache(20)
# 82.6 ns ± 0.125 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

#### iteration fibonacci optimization

递归改成迭代, 利用两个变量存储过程量; 相对without optimization 1934.0倍;

seq optimization: 以下以此为基准

```python
# cell
def fib_seq(n):
    if n < 2:
        return n
    a,b = 1,0
    for i in range(n-1):
        a,b = a+b,a
    return a

%timeit fib_seq(20)
# 1.06 µs ± 1.81 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

seq numba: 6.8倍

```python
# cell
from numba import jit

@jit
def fib_seq_numba(n):
    if n < 2:
        return n
    a,b = 1,0
    for i in range(n-1):
        a,b = a+b,a
    return a

%timeit fib_seq_numba(20)
# 155 ns ± 0.251 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

seq cython optimization: 1.85倍

```python
%load_ext cython
```
```python
%%cython
def fib_seq_cython(n):
    if n < 2:
        return n
    a,b = 1,0
    for i in range(n-1):
        a,b = a+b,a
    return a
```
```python
%timeit fib_seq_cython(20)
# 573 ns ± 1.91 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

seq cython optimization with type: 19.03倍

```python
%load_ext cython
```
```python
%%cython
cpdef long fib_seq_cython_type(long n):
    if n < 2:
        return n
    cdef long a,b
    a,b = 1,0
    for i in range(n-1):
        a,b = a+b,a
    return a
```
```python
%timeit fib_seq_cython_type(20)
# 55.7 ns ± 0.166 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

seq with cache: 12.9倍

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_seq_cache(n):
    if n < 2:
        return n
    a,b = 1,0
    for i in range(n-1):
        a,b = a+b,a
    return a

%timeit fib_seq_cache(20)
# 82.3 ns ± 0.0663 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

#### cached fibonacci optimization

在cache的基础上, 然后优化, 相对于without optimization的24818.4倍

cached fibanacci: 下面以此为基准

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_cache(n):
    if n<2:
        return n
    return fib_cache(n-1)+fib_cache(n-2)

%timeit fib_cache(20)
# 82.6 ns ± 0.125 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

numba optimization: 0.98倍

```python
from numba import jit
from functools import lru_cache

@lru_cache(maxsize=None)
@jit
def fib_cache_jit(n):
    if n<2:
        return n
    return fib_cache_jit(n-1)+fib_cache_jit(n-2)

%timeit fib_cache_jit(20)
# 84.2 ns ± 0.0726 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

cython optimization: 0.95倍

```python
#cell
%load_ext cython
```
```python
%%cython

from functools import lru_cache
@lru_cache(maxsize=None)
def fib_cache_cython(n):
    if n<2:
        return n
    return fib_cache_cython(n-1)+fib_cache_cython(n-2)
```
```python
%timeit fib_cache_cython(20)
# 86.7 ns ± 0.102 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```

cython optimization无法用type优化

#### summary

||without optimization|numba|cython|cython_type|cache|
|---|---|---|---|---|---|---|---|
recursive fibnacci|2.05ms|47.2μs|841μs|20.3μs|82.6ns
iteration fibnacci|1.06μs|155ns|573ns|55.7ns|82.3ns
cached fibnacci|82.6ns|84.2|86.7ns|/|/

cache之后, 再怎么优化也没有提升;

### numpy optimization

采用`numexpr`对numpy进行优化，主要是优化memory数据存储,一般提升10倍, 主要是解决内存的瓶颈;

```python
# cell
import numpy as np
import numexpr as ne

nx, ny = 200, 200
a = np.linspace(0.,3.1416,nx*ny).reshape(nx,ny)
```
```python
%%timeit
for i in range(100):
    b = np.sin(a+i)**2 + np.cos(a+i)**2 + a**1.5
# 153 ms ± 111 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
```python
%%timeit
for i in range(100):
    b = ne.evaluate("sin(a+i)**2 + cos(a+i)**2 + a**1.5")
# 15.4 ms ± 13.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

## `ipyparallel` module

`conda install ipyparallel`

[ipyparallel](https://github.com/ipython/ipyparallel)主要用于**并行计算**和**分布式计算**

