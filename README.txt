需要的依赖包：
	- pywavelets

目录结构
	-Project
  	- util package		: （无需更改）工具包
  		- __init__.py	: （无需更改）初始化log
		- aser.py		: （无需更改）计算P,R,F1指标的工具包
  		- g_lfcc_final.py	: （无需更改）提取LFCC特征的包
         		- log.py		: （无需更改）python的log工具
         		- model_handle.py	: （无需更改）模型处理工具包，保存和读取训练好的模型
         		- progressbar.py	: （无需更改）进度栏工具包
         		- util.py		: （无需更改）读取文件列表、生成文件夹列表、写日志等函数集合
    	- __init__.py		: （无需更改）
    	- config.py		: 项目配置文件（包含路径，学习率等等。。。）
    	- gen_feat.py		: 训练集验证集测试集生成文件
    	- LCNN_cl_lfcc.py		: （无需更改）神经网络（可能有其他命名  Waveunet、Lcnn_For_Sig等）
    	- test.py			: 测试文件
    	- train.py			: 模型训练文件

准备工作：按照注释将config文件准备好

训练模型:
	- python train.py
	- python -W ignore train.py : 忽略由提取小波特征造成的警告
	- 若要指定模型，在model/或者best_model/文件夹中找到模型名字，命令行python train.py -m model_name

测试模型：
	- python test,py
	- python -W ignore test.py : 忽略由提取小波特征造成的警告
	- 若要指定模型，在model/或者best_model/文件夹中找到模型名字，命令行python test.py -m model_name