import numpy as np

# 参数设置
sampling_rate = 10000  # 采样频率 (Hz)
sampling_duration = 120  # 每类工况采样时长 (秒)
total_points = sampling_rate * sampling_duration  # 总点数
sample_length = 1024  # 每个样本点数
num_samples = 1000  # 每类工况提取的样本数

# 模拟正常工况和故障工况信号
time = np.linspace(0, sampling_duration, total_points)

# 正常工况信号 (正弦波)
normal_signal = np.sin(2 * np.pi * 50 * time)  # 基频 50 Hz

# 故障工况信号 (添加谐波)
fault_signal = normal_signal + 0.2 * np.sin(2 * np.pi * 150 * time)  # 添加 150 Hz 谐波

# 样本切分函数
def extract_samples(signal, sample_length, num_samples):
    samples = []
    for i in range(num_samples):
        start_idx = i * sample_length
        end_idx = start_idx + sample_length
        if end_idx <= len(signal):
            samples.append(signal[start_idx:end_idx])
    return np.array(samples)

# 提取样本
normal_samples = extract_samples(normal_signal, sample_length, num_samples)
fault_samples = extract_samples(fault_signal, sample_length, num_samples)

# 检查结果
print(f"正常工况样本数量: {len(normal_samples)}, 每个样本长度: {normal_samples.shape[1]}")
print(f"故障工况样本数量: {len(fault_samples)}, 每个样本长度: {fault_samples.shape[1]}")
