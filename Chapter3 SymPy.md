# SymPy

- [SymPy](#sympy)


![](res/intro01.png)

SymPy是Python的数学符号计算库，用它可以进行数学公式的符号推导

## simple example

```python
import sympy

# evaluate expression e^(i*pi)+1=0
print((sympy.E**(sympy.I*sympy.pi)+1).evalf()) # 0
print(sympy.N(sympy.E**(sympy.I*sympy.pi)+1)) # 0

# expand
x = sympy.Symbol('x')
print(sympy.expand(sympy.E**(sympy.I*x))) # exp(I*x)
print(sympy.expand(sympy.E**(sympy.I*x), complex=True)) # I*exp(-im(x))*sin(re(x)) + exp(-im(x))*cos(re(x))

# generanl expand
x = sympy.Symbol('x', real=True) # real number
print(sympy.expand(sympy.E**(sympy.I*x), complex=True)) # I*sin(x) + cos(x)

# tayplor series
tmp=sympy.series(sympy.exp(sympy.I*x), x, 0, 10)
print(tmp) # 1 + I*x - x**2/2 - I*x**3/6 + x**4/24 + I*x**5/120 - x**6/720 - I*x**7/5040 + x**8/40320 + I*x**9/362880 + O(x**10)
sympy.pprint(tmp) # in beautiful format

# print real part, image part
sympy.pprint(sympy.re(tmp))
sympy.pprint(sympy.im(tmp))
```

## integration

```python
import sympy

# 不定积分
x = sympy.Symbol("x", real=True)
tmp = sympy.integrate(x*sympy.sin(x), x)
print(tmp) # -x*cos(x) + sin(x)

# 定积分
tmp = sympy.integrate(x*sympy.sin(x), (x, 0, 2*sympy.pi))
print(tmp) # -2*pi
```

```python
# 圆面积 pi*r**2
import sympy

x, y, r = sympy.symbols('x,y,r')
# 必须要指定r>0
r = sympy.Symbol('r', positive=True)
tmp = 2 * sympy.integrate(sympy.sqrt(r*r-x**2), (x, -r, r))
print(tmp) # pi*r**2
```

```python
# 球体体积
import sympy

x, y, r = sympy.symbols('x,y,r')
r = sympy.Symbol('r', positive=True)
circle_area = 2 * sympy.integrate(sympy.sqrt(r*r-x**2), (x, -r, r)) # pi*r**2
# 无数个小圆堆成球体，替换circle_area中的r为(r**2-x**2)**0.5
circle_area = circle_area.subs(r, sympy.sqrt(r**2-x**2))
sphere_volume=sympy.integrate(circle_area, (x, -r, r))
print(sphere_volume) # 4*pi*r**3/3
```

subs函数可以将算式中的符号进行替换，它有3种调用方式：

- `expression.subs(x, y)` : 将算式中的x替换成y
- `expression.subs({x:y,u:v})` : 使用字典进行多次替换
- `expression.subs([(x,y),(u,v)])` : 使用列表进行多次替换

请注意多次替换是顺序执行的，因此：`expression.sub([(x,y),(y,x)])`并不能对两个符号x,y进行交换。
