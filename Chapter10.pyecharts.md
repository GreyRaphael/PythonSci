# pyecharts

[pyecharts github](https://github.com/pyecharts/pyecharts)

example: in python console

```py
from pyecharts.charts import Bar
from pyecharts import options as opts
import os

bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带"])
    .add_yaxis("商家A", [114, 55, 27])
    .add_yaxis("商家B", [57, 134, 137])
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
)

bar.render('test.html')
os.system('test.html') # html containing image in browser
```

example: in jupyter notebook

```py
from pyecharts.charts import Bar
from pyecharts import options as opts

bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带"])
    .add_yaxis("商家A", [114, 55, 27])
    .add_yaxis("商家B", [57, 134, 137])
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
)

bar.render_notebook()
```

```py
bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带"])
    # stack bars
    .add_yaxis("商家A", [114, 55, 27],stack=True)
    .add_yaxis("商家B", [57, 134, 137], stack=True)
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
)
```

```py
bar = (
    Bar()
    .add_xaxis(["衬衫", "毛衣", "领带"])
    .add_yaxis("商家A", [114, 55, 27])
    .add_yaxis("商家B", [57, 134, 137])
    .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
    # reverse axis
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position="right"))
)
```

其他种类图直接参考[gallery](https://pyecharts.org) or 运行[github example](https://github.com/pyecharts/pyecharts/tree/master/example)得到render.html进而浏览器打开即可