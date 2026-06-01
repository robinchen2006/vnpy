# 模型训练
# 依赖：先执行 research_workflow_alpha101_feature.py 生成数据集缓存

# 加载模块
import numpy as np
import os

from vnpy.alpha import Segment, AlphaDataset, AlphaModel, AlphaLab
from vnpy.alpha.model.models.lgb_model import LgbModel

# 任务参数
name = "300_lgb"

# 创建数据中心
current_folder = os.path.dirname(os.path.abspath(__file__))
csi300_root = os.path.join(current_folder, f"./lab/csi300")

lab: AlphaLab = AlphaLab(csi300_root)


# 从文件缓存加载
dataset: AlphaDataset = lab.load_dataset(name)

# 创建模型对象
model: AlphaModel = LgbModel(seed=42)

# 使用数据集训练模型
model.fit(dataset)

# 查看模型细节
model.detail()

# 保存模型（供后续预测信号使用）
lab.save_model(name, model)
print(f"模型已保存：{name}")
