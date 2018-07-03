# Python Other plot

- [Python Other plot](#python-other-plot)
    - [Chaco](#chaco)
    - [tvtk](#tvtk)
    - [mayavi](#mayavi)
    - [VPython](#vpython)
    - [OpenCV](#opencv)

## Chaco

Chaco是一个2D的绘图库，类似matplotlib, matlab; 是一个interactive plot

Chaco的优势在于它可以很方便地嵌入到你的应用程序之中，开发出自己独特的绘图应用(比如, 实时数据可视化)

`conda info chaco`，只支持python2

现在的interactive plot一般都是在Jupyter里面用plotly

## tvtk

[VTK](www.vtk.org) 是一套三维的数据可视化工具，它由C++编写，包涵了近千个类帮助我们处理和显示数据.

可以用python调用c++的库，为了更好发挥动态语言的特性；[Enthought](https://www.enthought.com/)包装VTK库成**tvtk**

安装是从`mayavi`中来的

## mayavi

conda里面只显示支持python2

## VPython

[VPython](http://vpython.org/) 是做3D动画的， 只支持python2

## OpenCV

OpenCV是Intel公司开发的开源计算机视觉库。它用C语言高速地实现了许多图像处理和计算机视觉方面的通用算法

OpenCV的库可以分为5个主要组成部分:

- CV : 包括了基本的图像处理和高级的计算机视觉算法，在Python中，opencv.cv模块与之对应
- ML : 机器学习库，包括许多统计分类器，opencv.ml模块与之对应
- HighGUI : 提供各种图像、视频、数据的输入输出和简单的GUI开发，opencv.highgui模块与之对应
- CXCore : 上述三个库都是以CXCore提供的基本数据结构和函数为基础，主模块opencv与之对应
- CvAux : 包括一些实验性的算法

![](res\opencv01.png)

