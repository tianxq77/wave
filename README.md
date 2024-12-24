# 项目名称
用于基于正常数据的异常检测

## 项目结构
- `data/raw/`: 原始传感器数据（HDF 文件）。
- `data/processed/`: 处理后的小波系数（JSON 文件）。
- `models/autoencoder.py`: 自编码器模型定义。
- `utils/`: 包含小波变换和数据处理工具。
- `train.py`: 自编码器模型训练脚本。
- `detect.py`: 异常检测脚本。
- 
## 数据结构示例
- 小波分解系数
```
{
    "device_1": {
        "level_0": [1.234, 2.345, 3.456, ...],
        "level_1": [0.123, -0.456, 0.789, ...],
        "level_2": [-0.987, 0.654, -0.321, ...]
    },
    "device_2": {
        "level_0": [1.567, 2.678, 3.789, ...],
        "level_1": [0.213, -0.324, 0.435, ...],
        "level_2": [-0.123, 0.456, -0.789, ...]
    }
}
```


## 安装依赖
```
pip install -r requirements.txt
```



数据预处理：
python utils/wavelet_transform.py
训练模型：
python train.py
运行检测：
python detect.py