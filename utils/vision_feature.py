# vision_feature.py
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


class OpenFaceFeatureProcessor:
    def __init__(self, max_length=96, feature_dim=29):
        self.max_length = max_length  # 与音频特征对齐的时间步长
        self.feature_dim = feature_dim  # OpenFace特征维度
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_features(self, video_path):
        """从预处理好的.npy文件加载特征"""
        # 将视频路径转换为特征文件路径
        # 示例：video_path="data/SIMS/Raw/video_001/0001.mp4"
        # → feature_path="data/SIMS/OpenFaceFeatures/video_001/0001.npy"
        path_parts = os.path.normpath(video_path).split(os.sep)
        video_id = path_parts[-2]
        clip_id = os.path.splitext(path_parts[-1])[0]
        feature_path = os.path.join(
            "/".join(path_parts[:-3]),  # 根目录
            "OpenFaceFeatures",  # 新增的特征目录
            video_id,
            f"{clip_id}.npy"
        )
        return np.load(feature_path)  # 形状: [原始帧数, feature_dim]

    def process_features(self, features):
        """处理时序特征：标准化、截断/填充"""
        # 处理空特征的情况
        if len(features) == 0:
            return torch.zeros((self.max_length, self.feature_dim))

        # 标准化
        if not self.is_fitted:
            self.scaler.partial_fit(features)
            self.is_fitted = True
        features = self.scaler.transform(features)

        # 时间维度处理
        seq_len = features.shape[0]
        if seq_len > self.max_length:
            # 均匀采样截断
            indices = np.linspace(0, seq_len - 1, self.max_length, dtype=int)
            features = features[indices]
        elif seq_len < self.max_length:
            # 零填充
            pad = np.zeros((self.max_length - seq_len, self.feature_dim))
            features = np.vstack([features, pad])

        return torch.tensor(features, dtype=torch.float32)

    def create_mask(self, raw_features):
        """生成特征有效掩码"""
        valid_length = min(len(raw_features), self.max_length)
        mask = torch.cat([
            torch.ones(valid_length),
            torch.zeros(self.max_length - valid_length)
        ])
        return mask.long()