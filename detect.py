import numpy as np
import tensorflow as tf

# 加载模型（使用 .keras 格式）
autoencoder = tf.keras.models.load_model("models/autoencoder_model.keras")

# 加载和预处理新数据
new_data_path = "data/raw/new_sensor_data.hdf"
new_raw_data = load_hdf_data(new_data_path)
new_processed_data = preprocess_data(new_raw_data)

# 获取重构误差
reconstructed_data = autoencoder.predict(new_processed_data)
errors = np.mean(np.square(new_processed_data - reconstructed_data), axis=1)

# 设置阈值并检测
threshold = np.percentile(errors, 95)  # 假设正常数据的 95% 定为阈值
anomalies = errors > threshold

# 输出结果
print(f"检测到 {np.sum(anomalies)} 个异常样本")
