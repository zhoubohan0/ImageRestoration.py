# 图像复原

## 简介

​	为数据集图片模拟添加激光红点，并通过设计算法，在复杂场景下自动去除光电，完成图像修复。项目由`zhoubohan`和`yangyuxue`合作完成。

## 工作流程

### 数据集准备

​	爬虫获取网站风景图片

### 准备激光点素材

​	对激光点图片进行尺度变换、阈值化处理，获得掩膜

### 添加激光点

​	随机生成尺度和目标位置

### 激光点检测

> [备注]：部分引用自https://github.com/Yuppie898988/Laser_light_dot_detect

1. 光点HSV通道色相提取

2. 闭运算，填补圆形

3. 方差分析

   - 根据二值图像构建边界

   - 对每块边界点找出过边界的最小圆圆心

   - 分析各边界点的圆心距的方差

   - 保留方差小于阈值`minVar`的圆心

### 图像修复

​	**FMM**算法，opencv中cv2.inpaint()函数，flags = **cv2.INPAINT_TELEA**

## 效果展示

| ![13](./test_withspot/13.jpg) | ![13](./test_withoutspot/13.jpg) |
| :---------------------------: | -------------------------------- |
|        image with spot        | image without spot               |
| ![18](./test_withspot/18.jpg) | ![18](./test_withoutspot/18.jpg) |
|        image with spot        | image without spot               |

## 文件结构

`README`：说明

`display`：中间过程展示

`src`：激光点素材图片

`test`：测试图片

`test_withspot`：添加激光点图片（中间输出）

`test_withoutspot`：去除激光点图片（最终输出）

`main`：主程序

`Spider`：简易爬虫（构建数据集）
