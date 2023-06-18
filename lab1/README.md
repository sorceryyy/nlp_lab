# 文本情感分类
## 环境依赖
* pytorch
* jieba
* torchtext==0.3.1

## 用法
### 训练
请在目录下运行
```bash
python main.py --train-model True
```

### 测试
请在目录下运行，该命令会生成测试集的预测结果到当前目录
```bash
python main.py --train-model True --snapshot snapshot/model_name.pt
```


### 词向量
请从以下网址下载对应词向量:
https://github.com/Embedding/Chinese-Word-Vectors

下载完成后，请将文件放到 pretrained/ 下，并在运行时添加参数：
```bash
--static True --pretrained-name word_vector_file_name
```


### 参数设定
查看更多可指定参数请使用
```bash
python3 main.py -h
```



