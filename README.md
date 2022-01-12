# pytorch_distribute_test

测试不同的Pytorch并行训练加速方案，基于imagenet数据，resnet模型

+ single GPU，使用一个GPU进行深度学习模型训练
+ nn.DataParallel，简单方便的DataParallel，单进程管理
+ torch.distributed + torch.multiprocessing，distributed并行加速，多进程管理
+ apex，apex再加速


### 重要参数
+ batch_size: 400
    + 每个GPU分配400个sample，填满40G显存
+ 节点数： 1
+ 显卡数： 8

### 测试结果
| 测试项 | 每Epoch用时（秒） | GPU利用率 | 显存占用率 |
| --- | ---| --- | --- |
| single GPU | 1786.7849 | 99.5% | 99.8% |
| nn.DataParallel | 984.5840 | 59.8% | 99.8% |
| torch.distributed | 239.3970 | 99.5% | 99.8% |
| hfai.nccl.distributed | 236.7068 | 99.5% | 99.8% |
| apex | 230.9803 | 88.8% | 61.2% |
| apex *4 | 54.5026 | 79.2% | 61.2% |
