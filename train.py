import os
import numpy as np
import pywt

from utils.data_processing import preprocess_data_and_save
from utils.wavelet_transform import load_wavelet_transform_from_json
from models.autoencoder import build_autoencoder

# 原始数据路径
raw_data_path = "data/raw/sensor_data.hdf"
processed_data_path = "data/processed/wavelet_transformed.json"


# 数据集名称列表
datasets = ['current', 'voltage']

# 处理数据并保存结果
preprocess_data_and_save(raw_data_path, datasets, processed_data_path, wavelet='db8', level=5)


# 加载小波变换后的数据（分别加载电流和电压的数据）
wavelet_data_current = load_wavelet_transform_from_json('data/processed/wavelet_transformed.json/current_wavelet.json')
wavelet_data_voltage = load_wavelet_transform_from_json('data/processed/wavelet_transformed.json/voltage_wavelet.json')

# 构造训练数据（重构每列的原始信号）
def reconstruct_signals(wavelet_data):
    training_data = []
    for device, coeffs in wavelet_data.items():
        # 将小波系数重构为信号
        signal = pywt.waverec([coeffs[f'level_{j}'] for j in range(len(coeffs))], 'db8')
        training_data.append(signal)
    # 转置为样本 x 维度
    return np.array(training_data).T

training_data_current = reconstruct_signals(wavelet_data_current)
training_data_voltage = reconstruct_signals(wavelet_data_voltage)

# 合并电流和电压的数据作为训练数据
training_data = np.concatenate([training_data_current, training_data_voltage], axis=1)

# 训练自编码器
input_dim = training_data.shape[1]
autoencoder = build_autoencoder(input_dim)
autoencoder.fit(
    training_data,
    training_data,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

# 保存模型为新的 .keras 格式
autoencoder.save("models/autoencoder_model.keras")

