#!/bin/bash

# 安装必要的依赖
pip install torch torchvision torchaudio
pip install transformers
pip install pandas
pip install scikit-learn
pip install tqdm
pip install matplotlib
pip install pyarrow

# 创建必要的目录
mkdir -p outputs/plots

echo "设置完成！"
