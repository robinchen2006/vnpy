# 模型训练
# 依赖：先执行 research_workflow_mlp_feature.py 生成数据集缓存

# 加载模块
import numpy as np

from vnpy.alpha import Segment, AlphaDataset, AlphaModel, AlphaLab
from vnpy.alpha.model.models.mlp_model import MlpModel

# 任务参数
name = "300_mlp"

# 创建数据中心
lab: AlphaLab = AlphaLab("./lab/csi300")

# 从文件缓存加载
dataset: AlphaDataset = lab.load_dataset(name)

# 创建模型对象
kwargs = {
    "input_size": 158,
    "hidden_sizes": (256,),
    "lr": 0.002,
    "optimizer": "adam",
    "n_epochs": 8000,
    "batch_size": 8192,
    "weight_decay": 0.0002,
    "seed": 42
}

model: AlphaModel = MlpModel(**kwargs)

# 使用数据集训练模型
model.fit(dataset)

# 查看模型细节
model.detail()

# 保存模型（供后续预测信号使用）
lab.save_model(name, model)
print(f"模型已保存：{name}")
