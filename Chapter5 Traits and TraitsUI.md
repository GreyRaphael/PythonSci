# Python Traits & TraisUI

- [Python Traits & TraisUI](#python-traits-traisui)
    - [Traits](#traits)
        - [What is traits](#what-is-traits)
            - [dynamically add traits](#dynamically-add-traits)
        - [trait Property](#trait-property)
        - [trait notification](#trait-notification)
    - [TraitsUI](#traitsui)
            - [view type](#view-type)

```bash
conda install traits
conda intstll traitsui
```

Traits & TraitsUI [Manual](http://bigsec.net/b52/scipydoc/traits_manual_index.html#)

## Traits

颜色属性可以是：

- 'red'
- 0xff0000
- (255, 0, 0)

也就是可以用string, integer, tuple等类型的值表达颜色；却不是能接受所有的值，比如"abc"、0.5等等就不能很好地表示颜色；对外的接口可以接受各种各样形式的值，但是在内部必须有一个统一的表达方式来简化程序的实现；

用Trait属性可以很好地解决这样的问题， **trait为Python对象的属性增加了类型定义的功能**:

- 它提供一个内部的标准的颜色表达方式
- 它可以接受能表示颜色的各种类型的值
- 当给它赋值为不能表达颜色的值时，能够捕捉到错误，并且报告错误，告诉用户它能够接受什么样的值

要用trait属性的class必须继承`HasTraits`

```python
from traits.api import HasTraits, Color


class Circle(HasTraits):
    # trait属性像类的属性一样定义，像实例的属性一样使用
    color = Color


try:
    print(Circle.color)
except Exception as e:
    print(e)  # type object 'Circle' has no attribute 'color'

c = Circle()
print(c.color.getRgb())  # (255, 255, 255, 255),默认值
c.color = 'red'
print(c.color)  # <PyQt5.QtGui.QColor object at 0x000001E12CD86668>
print(c.color.getRgb())  # (255, 0, 0, 255); 查看Qt的文档
print(c.color.name())  # ff0000

c.color=0x00ff00
print(c.color.getRgb()) # (0, 255, 0, 255)

c.color=(0, 0, 255)
print(c.color.getRgb()) # (0, 0, 255, 255)

# Qt的颜色比较奇特，支持float
c.color=0.5
print(c.color.getRgb()) # (0, 0, 0, 255)
```

```python
from traits.api import HasTraits, Color


class Circle(HasTraits):
    color = Color


c = Circle()
c.configure_traits()  # 弹出颜色dialog
print(c.color.getRgb())  # (38, 168, 255, 255)
```

### What is traits

trait为Python对象的属性增加了类型定义的功能，此外还提供了如下的[额外功能](http://docs.enthought.com/traits/traits_user_manual/intro.html)：

- Initialization：每个trait属性都定义有自己的缺省值，这个缺省值用来初始化属性
- Validation：基于trait的属性都有明确的类型定义，只有满足定义的值才能赋值给属性
- Delegate：trait属性的值可以委托给其他对象的属性
- Notification：trait属性的值的改变可以触发指定的函数的运行
- Visualization：拥有trait属性的对象可以很方便地提供一个用户界面交互式地改变trait属性的值

![](traits\traits01.png)

```python
from traits.api import Delegate, HasTraits, Instance, Int, Str

class Parent(HasTraits):
    # initialization
    last_name=Str('Ge')

class Child(HasTraits):
    age=Int

    # validation, father属性的值必须是Parent类的实例
    father=Instance(Parent)
    # delegate, Child的实例的last_name属性委托给其father属性的last_name
    last_name=Delegate('father', 'last_name')
    # notification, 当age属性的值被修改时，下面的函数将被运行
    def _age_changed(self, old, new):
        print(f"Age changed from {old} to {new}")

p=Parent()
c=Child()
c.father=p
print(c.last_name)
c.age=26
c.print_traits()
# age:       26
# father:    <__main__.Parent object at 0x0000023AE4E51938>
# last_name: 'Ge'
print(c.get()) # {'age': 26, 'father': <__main__.Parent object at 0x0000026B18F219E8>, 'last_name': 'Ge'}
c.set(age=27) # Age changed from 26 to 27
c.configure_traits()
print(c.last_name)
```

#### dynamically add traits

```python
from traits.api import HasTraits, Str, Instance, Delegate

obj1=HasTraits()
obj1.add_trait('prop1', Str('AlphaGrey'))
print(obj1.prop1) # AlphaGrey

obj2=HasTraits()
obj2.add_trait('prop1', Instance(HasTraits))
obj2.prop1=obj1

obj2.add_trait('prop2', Delegate('prop1', 'prop1', modify=True)) # 即obj2.prop2是obj2.prop1.prop1的代理

print(obj2.prop1) # <traits.has_traits.HasTraits object at 0x0000018503221888>
print(obj2.prop2) # AlphaGrey
```

### trait Property

标准的Python提供了Property功能，Property看起来像对象的一个成员变量，但是在获取它的值或者给它赋值的时候实际上是调用了相应的函数。Traits也提供了类似的功能。

```python
from traits.api import HasTraits, Property, Float, cached_property

class Rectangle(HasTraits):
    width=Float(1.0)
    height=Float(2.0)

    #area是一个属性，当width,height的值变化时，它对应的_get_area函数将被调用
    area = Property(depends_on=['width', 'height']) 

    # 通过cached_property decorator缓存_get_area函数的输出
    @cached_property 
    def _get_area(self):
        """
        area的get函数，注意此函数名和对应的Proerty名的关系
        """
        print('recalculating...')
        return self.width * self.height

rect=Rectangle()
print(rect.area)
# recalculating...
# 2.0
rect.width=10 # 修改width的时候，只是print('recalculating...'); 需要_get_are()的时候才会return
print(rect.area)
print(rect.area)
# recalculating...
# 20.0
# 20.0

rect.edit_traits() # 这个是非阻塞的
rect.edit_traits() # 这个是非阻塞的

t=rect.trait('area') # get type
print(len(t._notifiers(True))) #2, 到目前为止，有两个要被通知的；
rect.configure_traits() # 这个是阻塞的
```

Traits中根据属性名直接决定了它的访问函数，当用户读取area值时，将得到_get_area函数的返回值；而设置area的值时，_set_area函数将被调用。此外，通过关键字参数depends_on，指定当width和height属性变化时自动计算area属性。

对话框中的一个修改Height或者width, 三个界面的三个数据同步更新

![](traits\traits02.png)

### trait notification

如下分为静态notification; 动态notification;

```python
from traits.api import HasTraits, Str, Int


class Child(HasTraits):
    # 当某个trait属性值发生变化时，HasTraits对象会通知所有监听此属性的函数
    name = Str
    age = Int
    doing = Str

    def __str__(self):
        return f'{self.name}<{id(self)}>'

    # 当age属性的值被修改时，下面的函数将被运行
    def _age_changed(self, old, new):
        print(f'{self}.age changed from {old} to {new}\n')

    # _anytrait_changed则是一个特殊的静态监听函数，HasTraits对象的任何trait属性值的改变都会调用此函数
    def _anytrait_changed(self, name, old, new):
        print(f'anytrait changed: {self}.{name} from {old} to {new}')

# log_trait_changed是一个普通函数。通过c1.on_trait_change调用动态地将其与c1的doing属性联系起来，
# 即当c1对象的doing属性改变时，log_trait_changed函数将被调用。
def log_trait_changed(obj, name, old, new):
    print(f'log: {obj}.{name} changed from {old} to {new}')

if __name__ == '__main__':
    c1=Child(name='Grey', age=26)
    c2=Child(name='Moris', age=53)
    c1.on_trait_change(log_trait_changed, name='doing')
    
    c1.age=66
    c1.doing='sleeping'
    c2.doing='eating' # 上面的提示两次，这个因为没有on_trai_change()只提示一次
```

```bash
# output
anytrait changed: Grey<1444128496192>.name from  to Grey
anytrait changed: Grey<1444128496192>.age from 0 to 26
Grey<1444128496192>.age changed from 0 to 26

anytrait changed: Moris<1444128496104>.name from  to Moris
anytrait changed: Moris<1444128496104>.age from 0 to 53
Moris<1444128496104>.age changed from 0 to 53

anytrait changed: Grey<1444128496192>.age from 26 to 66
Grey<1444128496192>.age changed from 26 to 66

anytrait changed: Grey<1444128496192>.doing from  to sleeping
log: Grey<1444128496192>.doing changed from  to sleeping
anytrait changed: Moris<1444128496104>.doing from  to eating
```

![](traits\traits03.png)

## TraitsUI

在开发科学计算程序时，我们希望快速实现一个够用的界面，让用户能够交互式的处理数据，而又不希望在界面制作上花费过多的精力。以traits为基础、以Model-View-Controller为设计思想的TraitUI库就是实现这一理想；因为其他图形库(Tkinter, pyQt..)需要掌握很多api

![](traits\traitsui01.png)

```python
from traits.api import HasTraits, Str, Int

class SimpleEmployee(HasTraits):
    first_name=Str
    last_name=Str
    department=Str
    salary=Int

sam=SimpleEmployee()
sam.configure_traits()
sam.print_traits()
```

```bash
# output
department: 'PKU'
first_name: 'Grey'
last_name:  'Alpha'
salary:     70000
```

![](traits\traitsui02.png)

```python
from traits.api import HasTraits, Str, Int
from traitsui.view import View,Item

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str
    salary = Int

view1 = View(
    Item(name = 'department', label=u"部门", tooltip=u"在哪个部门干活"),
    Item(name = 'last_name', label=u"姓"),
    Item(name = 'first_name', label=u"名"))

sam = SimpleEmployee()
sam.configure_traits(view=view1)
```

首先从其中载入View和Item。View用来生成视图，而Item则用来描述视图中的项目(控件)。程序中，用Item依次创建三个视图项目，都作为参数传递给View，于是所生成的界面中按照参数的顺序显示控件，而不是按照trait属性名排序了

![](traits\traitsui03.png)

```python
from traits.api import HasTraits, Str, Int
from traitsui.view import View, Item, Group

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str

    employee_number = Str
    salary = Int
    bonus = Int

view1 = View(
    Group(
        Item(name = 'employee_number', label=u'编号'),
        Item(name = 'department', label=u"部门", tooltip=u"在哪个部门干活"),
        Item(name = 'last_name', label=u"姓"),
        Item(name = 'first_name', label=u"名"),
        label = u'个人信息',
        show_border = True
    ),
    Group(
        Item(name = 'salary', label=u"工资"),
        Item(name = 'bonus', label=u"奖金"),
        label = u'收入',
        show_border = True
    )
)

sam = SimpleEmployee()
sam.configure_traits(view=view1)
```

![](traits\traitsui04.png)

```python
from traits.api import HasTraits, Str, Int
from traitsui.view import View, Item, Group

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str

    employee_number = Str
    salary = Int
    bonus = Int

view2 = View(Group(
    Group(
        Item(name = 'employee_number', label=u'编号'),
        Item(name = 'department', label=u"部门", tooltip=u"在哪个部门干活"),
        Item(name = 'last_name', label=u"姓"),
        Item(name = 'first_name', label=u"名"),
        label = u'个人信息',
        show_border = True
        ),
    Group(
        Item(name = 'salary', label=u"工资"),
        Item(name = 'bonus', label=u"奖金"),
        label = u'收入',
        show_border = True
        )
))

sam = SimpleEmployee()
sam.configure_traits(view=view2)
```

或者在第一个基础上修改

```python
view2 = View( Group( view1.content ) )
sam.configure_traits(view=view2)
```

![](traits\traitsui05.png)

```python
from traits.api import HasTraits, Str, Int
from traitsui.view import View, Item, Group

class SimpleEmployee(HasTraits):
    first_name = Str
    last_name = Str
    department = Str

    employee_number = Str
    salary = Int
    bonus = Int

view2 = View(Group(
    Group(
        Item(name = 'employee_number', label=u'编号'),
        Item(name = 'department', label=u"部门", tooltip=u"在哪个部门干活"),
        Item(name = 'last_name', label=u"姓"),
        Item(name = 'first_name', label=u"名"),
        label = u'个人信息',
        show_border = True
        ),
    Group(
        Item(name = 'salary', label=u"工资"),
        Item(name = 'bonus', label=u"奖金"),
        label = u'收入',
        show_border = True
        ),
    layout = 'split', orientation = 'horizontal'
))

sam = SimpleEmployee()
sam.configure_traits(view=view2)
```

其他的参见文档: HGroup, HFlow, HSplit, Tabbed, VGroup, VFlow, VFold, VGrid, VSplit

#### view type

通过kind属性可以修改View对象的显示类型：

- 'modal' : 模式窗口, 非即时更新
- 'live' : 非模式窗口，即时更新
- 'livemodal' : 模式窗口，即时更新
- 'nonmodal' : 非模式窗口，非即时更新
- 'wizard' : 向导类型
- 'panel' : 嵌入到其它窗口中的面板，即时更新，非模式
- 'subpanel'

所谓模式窗口，表示此窗口关闭之前，程序中的其它窗口都不能被激活。

而即时更新则是指当窗口中的控件内容改变时，修改会立即反应到窗口所对应的模型数据上，

非即时更新的窗口则会复制模型数据，所有的改变在模型副本上进行，只有当用户确定修改(通常通过OK或者Apply按钮)时，才会修改原始数据。

'wizard'由一系列特定的向导窗口组成，属于模式窗口，并且即时更新数据。

'panel'和'subpanel' 则是嵌入到窗口中的面板，panel可以拥有自己的命令按钮，而subpanel则没有命令按钮。