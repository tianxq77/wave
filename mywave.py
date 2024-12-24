import pywt
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 示例信号
np.random.seed(42)  # 固定随机种子，便于重复实验
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 256)) + 0.5 * np.random.randn(256)


# 小波分解
def wavelet_decompose(signal, wavelet='db1', level=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs


# 保存分解参数
def save_params(filename, wavelet, coeffs):
    params = {'wavelet': wavelet, 'coeffs': coeffs}
    with open(filename, 'wb') as f:
        pickle.dump(params, f)


# 加载分解参数
def load_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# 小波重构
def wavelet_reconstruct(params):
    wavelet = params['wavelet']
    coeffs = params['coeffs']
    reconstructed_signal = pywt.waverec(coeffs, wavelet)
    return reconstructed_signal


# 绘图
def plot_results(signal, coeffs, reconstructed_signal):
    plt.figure(figsize=(12, 8))

    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(signal, label="Original Signal")
    plt.title("Original Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # 分解系数
    plt.subplot(3, 1, 2)
    for i, coeff in enumerate(coeffs):
        plt.plot(coeff, label=f"Level {i}")
    plt.title("Wavelet Decomposition Coefficients")
    plt.xlabel("Sample Index")
    plt.ylabel("Coefficient Value")
    plt.legend()

    # 重构信号
    plt.subplot(3, 1, 3)
    plt.plot(reconstructed_signal, label="Reconstructed Signal", linestyle="--")
    plt.title("Reconstructed Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 使用示例
wavelet = 'db4'  # 小波基
coeffs = wavelet_decompose(signal, wavelet)

# 保存分解结果
filename = 'wavelet_params.pkl'
save_params(filename, wavelet, coeffs)

# 加载并重构信号
loaded_params = load_params(filename)
reconstructed_signal = wavelet_reconstruct(loaded_params)

# 绘制结果
plot_results(signal, coeffs, reconstructed_signal)
