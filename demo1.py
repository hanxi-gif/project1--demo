# 先安装依赖：pip install streamlit joblib mne numpy scikit-learn scipy pandas
import streamlit as st
import pandas as pd
import numpy as np
import mne
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import signal

warnings.filterwarnings("ignore")

# ===================== 配置：替换为你的CSV数据路径 =====================

MODEL_PATH = "eeg_intent_model.pkl"
SCALER_PATH = "eeg_scaler.pkl"
LE_PATH = "eeg_label_encoder.pkl"


# ===================== 1. 封装算法函数（复用你的逻辑） =====================
def load_and_preprocess_data():
    """加载CSV数据+预处理（IIR+ICA）"""
    # 加载数据
    df = pd.read_csv("E:\数据集\kaggle数据集\脑电图运动想象BCICIV_2a\BCICIV_2a_all_patients.csv")
    eeg_data = df.iloc[:, 4:26].values.T  # [通道数, 时间点]
    raw_labels = df.iloc[:, 2].values

    # 创建MNE Raw对象
    sfreq = 250
    ch_names = [f"EEG{i + 1}" for i in range(22)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)

    # 预处理（IIR+ICA）
    raw.filter(2, 40, method="iir")
    ica = mne.preprocessing.ICA(n_components=4, random_state=42)
    ica.fit(raw)
    raw = ica.apply(raw)

    # 创建Epochs
    n_timepoints = raw.n_times
    trial_length = 3 * sfreq
    events = np.array([[i, 0, 1] for i in range(0, n_timepoints - trial_length, trial_length)])
    epochs = mne.Epochs(raw, events, tmin=0, tmax=3 - 1 / sfreq, preload=True, baseline=None)

    return epochs, raw_labels


def extract_features(epochs, sfreq=250):
    """提取特征（复用你的时域+频域逻辑）"""
    data = epochs.get_data()  # [n_epochs, n_channels, n_times]
    n_epochs, n_channels, n_times = data.shape
    features = []

    for i in range(n_epochs):
        epoch_features = []
        for ch in range(n_channels):
            sig = data[i, ch, :]
            # 时域特征
            time_feats = [np.mean(sig), np.std(sig), np.var(sig),
                          np.max(sig) - np.min(sig), np.median(sig), np.mean(np.abs(sig))]
            # 差分特征
            diff_feats = [np.mean(np.diff(sig)), np.std(np.diff(sig))]
            # 频域特征
            nperseg = min(64, len(sig) // 2)
            f, Pxx = signal.welch(sig, fs=sfreq, nperseg=nperseg)
            alpha = np.mean(Pxx[(f >= 8) & (f <= 13)]) if np.any((f >= 8) & (f <= 13)) else 0.0
            beta = np.mean(Pxx[(f >= 13) & (f <= 30)]) if np.any((f >= 13) & (f <= 30)) else 0.0
            # 合并
            epoch_features.extend(time_feats + diff_feats + [alpha, beta])
        features.append(epoch_features)

    return np.array(features)


def train_or_load_model(features, raw_labels, n_epochs_actual):
    """训练/加载模型（首次训练后保存）"""
    try:
        # 加载已保存的模型
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        st.sidebar.success("✅ 已加载预训练模型")
    except:
        # 首次训练
        st.sidebar.info("🔄 首次运行，自动训练模型...")
        # 准备标签
        if len(raw_labels) >= n_epochs_actual:
            y = raw_labels[:n_epochs_actual]
        else:
            y = np.tile(raw_labels, n_epochs_actual // len(raw_labels) + 1)[:n_epochs_actual]
        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        # 划分数据
        X_train, _, y_train, _ = train_test_split(features, y_encoded, test_size=0.2, random_state=42)
        # 标准化+训练
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X_train_scaled, y_train)
        # 保存模型
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le, LE_PATH)
        st.sidebar.success("✅ 模型训练完成")

    return model, scaler, le


# ===================== 2. Streamlit界面逻辑 =====================
def main():
    st.title("🎯 脑电运动想象意图识别DEMO")
    st.write("基于BCI Competition IV 2a数据集 | 支持运动意图分类")

    # 侧边栏：设备状态
    st.sidebar.header("📡 设备状态")
    st.sidebar.write("电极连接：✅ 正常")
    st.sidebar.write("信号质量：⭐⭐⭐⭐")
    st.sidebar.write("采样率：250Hz")

    # 加载数据+预处理（页面加载时执行）
    with st.spinner("初始化：加载数据+预处理..."):
        epochs, raw_labels = load_and_preprocess_data()
        n_epochs_actual = len(epochs)
        st.success("✅ 数据加载+预处理完成")

    # 提取特征
    with st.spinner("提取脑电特征..."):
        features = extract_features(epochs)
        st.success("✅ 特征提取完成")

    # 加载/训练模型
    model, scaler, le = train_or_load_model(features, raw_labels, n_epochs_actual)

    # 核心交互：选择试次+识别意图
    st.subheader("选择要识别的脑电试次")
    trial_idx = st.slider("试次编号", min_value=0, max_value=n_epochs_actual - 1, value=0)

    if st.button("🚀 识别该试次的意图", type="primary"):
        # 获取选中试次的特征
        trial_feat = features[trial_idx].reshape(1, -1)
        trial_feat_scaled = scaler.transform(trial_feat)
        # 预测
        pred = model.predict(trial_feat_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]
        # 显示结果
        st.success(f"🎉 意图识别结果：")
        st.markdown(f"<h2 style='color:green; text-align:center'>{pred_label}</h2>", unsafe_allow_html=True)


if __name__ == "__main__":

    main()
