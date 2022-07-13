# 多模态情感分析

该仓库存放了多模态情感分析实验的配套代码。

## 目录结构

```
├─baseline                  不使用预训练模型的baseline
│ ├─baseline.py
│ ├─model.py
│ └─data_utils.py
├─data                      训练数据-文本和图像(未上传) 
├─label                     训练数据-标签    
│ ├─test_without_label.txt
│ └─train.txt
├─output                    输出(预测测试集)
├─main.py                   主函数    
├─data_utils.py             数据处理模块
├─model.py                  多模态模型构建
├─figs.ipynb                部分图表
├─README.md
└─requirements.txt
```

## 安装
```bash
pip install -r requirements.txt
```

## 添加数据集
请手动将训练数据中的文本和图像放入data/

## 查看可调整参数
```bash
python ./main.py -h
```

## 训练
多模态融合模型
```bash
python ./main.py
```

消融实验(仅图像)
```bash
python ./main.py --image_only
```

消融实验(仅文本)
```bash
python ./main.py --text_only
```

## 预测测试集
```bash
python ./main.py --do_test
```

## 其他
训练baseline模型
```bash
python ./baseline/baseline.py
```

## 参考

vista-net: https://github.com/PreferredAI/vista-net

TomBERT: https://github.com/jefferyYu/TomBERT

transformers.modeling_bert: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

