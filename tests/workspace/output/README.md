---
tasks:
- sentence-similarity
model_type:
- bert
domain:
- nlp
frameworks:
- pytorch
backbone:
- transformer
metrics:
- accuracy
license: Apache License 2.0
language: 
- cn
tags:
- Alibaba
- sentence-similarity
- 文本相似度
- 文本对匹配
datasets:
  train:
  - damo/QBQTC
  test:
  - damo/QBQTC
finetune-support: True
---

# MaSTS中文预训练-CLUE语义匹配模型介绍

基于此模型在[QBQTC数据集](https://modelscope.cn/datasets/damo/QBQTC/summary)上训练得到[MaSTS中文文本相似度-CLUE语义匹配模型](https://modelscope.cn/models/damo/nlp_masts_sentence-similarity_clue_chinese-large/summary)。

## 模型描述

模型通过对语义匹配任务改进的掩码策略进行无监督预训练。按照BERT文本对分类的方式，在QBQTC数据集上进行微调。

### 期望模型使用方式以及适用范围

模型主要用于在QBQTC数据集上进行微调。

## 如何使用

### 环境安装

请参考ModelScope[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。

### Finetune/训练代码范例

```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config


model_id = 'damo/nlp_masts_backbone_clue_chinese-large'
dataset_id = 'QBQTC'

WORK_DIR = 'workspace'

cfg = read_config(model_id, revision='v1.0.0')
cfg.train.work_dir = WORK_DIR
cfg_file = osp.join(WORK_DIR, 'train_config.json')
cfg.dump(cfg_file)

train_dataset = MsDataset.load(dataset_id, namespace='damo', subset_name='default', split='train', keep_default_na=False)
eval_dataset = MsDataset.load(dataset_id, namespace='damo', subset_name='public', split='test', keep_default_na=False)

kwargs = dict(
    model=model_id,
    model_revision='v1.0.0',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    cfg_file=cfg_file,
)

trainer = build_trainer(default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(cfg.train.max_epochs):
    eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)

print('===============================================================')
print('evaluate success')
print('===============================================================')
```
