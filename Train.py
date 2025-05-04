import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import tensorflow as tf
from model import Model
import warnings
import os
import joblib
from datetime import datetime
import logging
import psutil
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 创建实验目录
experiment_time = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_dir = os.path.join("experiments", experiment_time)
os.makedirs(experiment_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(experiment_dir, "training.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 设置警告过滤
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 监控内存使用
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")

# 读取数据并过滤
logger.info("开始加载数据...")
start_time = time.time()

try:
    df = pd.read_csv("find.csv", low_memory=False)
    logger.info(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"原始数据形状: {df.shape}")
except Exception as e:
    logger.error(f"数据加载失败: {str(e)}")
    raise

# 数据清洗和预处理
logger.info("开始数据清洗...")
df = df[['loan_amnt', 'int_rate', 'annual_inc', 'loan_status']]

# 数据清洗和预处理
with tqdm(total=4, desc="数据清洗进度") as pbar:
    # 处理无穷值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    pbar.update(1)
    
    # 删除缺失值
    df.dropna(inplace=True)
    pbar.update(1)
    
    # 确保数值列为数值类型
    for col in ['loan_amnt', 'int_rate', 'annual_inc']:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('[\$,\%]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    pbar.update(1)
    
    # 筛选二分类标签
    valid_status = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(valid_status)]
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    pbar.update(1)

logger.info(f"数据清洗完成，剩余样本数: {len(df)}")
log_memory_usage()

# 处理异常值 - 使用更保守的方法，只过滤极端异常值
for col in ['loan_amnt', 'int_rate', 'annual_inc']:
    # 使用百分位数处理异常值，更保守的裁剪
    lower = df[col].quantile(0.001)
    upper = df[col].quantile(0.999)
    df[col] = df[col].clip(lower=lower, upper=upper)

# 打印类别分布
print("类别分布:")
print(df['loan_status'].value_counts(normalize=True))

# 保存基本统计数据
df.describe().to_csv(f"{experiment_dir}/data_stats.csv")

# 特征工程 - 更丰富的特征集
print("特征工程...")

# 对数变换
df['log_annual_inc'] = np.log1p(df['annual_inc'].clip(lower=1))
df['log_loan_amnt'] = np.log1p(df['loan_amnt'].clip(lower=1))

# 比率特征
df['debt_ratio'] = df['loan_amnt'] / (df['annual_inc'].clip(lower=1))
df['interest_burden'] = df['loan_amnt'] * df['int_rate'] / 1000
df['income_per_dollar_borrowed'] = df['annual_inc'] / (df['loan_amnt'].clip(lower=1))
df['interest_to_income'] = df['int_rate'] / (df['log_annual_inc'].clip(lower=1))

# 交互特征
df['loan_int_interaction'] = df['loan_amnt'] * df['int_rate']
df['high_risk_flag'] = ((df['debt_ratio'] > 0.5) & (df['int_rate'] > 15)).astype(int)

# 分箱特征
try:
    df['inc_bin'] = pd.qcut(df['annual_inc'], q=5, labels=False)
    df['loan_bin'] = pd.qcut(df['loan_amnt'], q=5, labels=False)
    df['int_bin'] = pd.qcut(df['int_rate'], q=5, labels=False)
except:
    print("分箱过程中出现错误，跳过分箱特征")

# 最终特征选择
numerical_features = [
    'loan_amnt', 'int_rate', 'annual_inc',
    'log_annual_inc', 'log_loan_amnt',
    'debt_ratio', 'interest_burden',
    'income_per_dollar_borrowed', 'interest_to_income',
    'loan_int_interaction'
]

# 检查并删除有问题的特征
for col in numerical_features:
    if col in df.columns and (df[col].isnull().any() or np.isinf(df[col]).any()):
        print(f"特征 {col} 包含无效值，已移除")
        numerical_features.remove(col)

# 保存特征重要性
pd.DataFrame({'feature': numerical_features}).to_csv(f"{experiment_dir}/features.csv", index=False)

# 准备数据
X = df[numerical_features].values
y = df['loan_status'].values

# 保存原始数据维度信息
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"正样本比例: {np.mean(y):.2%}")

# 使用SMOTE进行过采样
print("使用SMOTE平衡数据集...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 数据转换 - 使用StandardScaler进行标准化
print("特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 保存变换器
joblib.dump(scaler, f"{experiment_dir}/scaler.pkl")

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# 使用交叉验证训练多个模型
logger.info("开始交叉验证训练...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

val_aucs = []
val_accuracies = []
models = []
best_model = None
best_auc = 0

# 创建模型检查点目录
checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    logger.info(f"\n===== 训练折 {fold + 1}/{n_folds} =====")
    fold_start_time = time.time()

    # 分割训练集和验证集
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_fold_train),
        y=y_fold_train
    )
    class_weights = dict(enumerate(class_weights))
    logger.info(f"类别权重: {class_weights}")

    # 初始化模型
    model = Model(
        input_dim=X_fold_train.shape[1],
        complexity='high'
    )

    # 设置检查点回调
    checkpoint_path = os.path.join(checkpoint_dir, f"model_fold_{fold + 1}.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=8,
            mode='max',
            min_lr=1e-6
        )
    ]

    # 训练模型
    history = model.train(
        X_fold_train, y_fold_train,
        X_fold_val, y_fold_val,
        epochs=80,
        batch_size=1024,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 评估验证集性能
    val_metrics = model.evaluate(X_fold_val, y_fold_val)
    val_auc = val_metrics.get('auc', 0)
    val_aucs.append(val_auc)
    val_accuracies.append(val_metrics.get('accuracy', 0))

    # 保存模型
    model_save_path = os.path.join(experiment_dir, f"model_fold_{fold + 1}.weights.h5")
    model.save(model_save_path)
    models.append(model)

    # 更新最佳模型
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = model

    # 记录训练时间
    fold_time = time.time() - fold_start_time
    logger.info(f"折 {fold + 1} 训练完成，耗时: {fold_time:.2f}秒")
    logger.info(f"验证AUC: {val_auc:.4f}, 验证准确率: {val_metrics.get('accuracy', 0):.4f}")

# 保存最佳模型
if best_model:
    best_model_save_path = os.path.join(experiment_dir, "best_model.weights.h5")
    best_model.save(best_model_save_path)
    logger.info(f"最佳模型已保存，验证AUC: {best_auc:.4f}")

# 使用集成方法进行最终预测
logger.info("\n===== 测试集评估(集成) =====")
test_probs = np.zeros(len(y_test))

for i, model in enumerate(models):
    logger.info(f"使用模型 {i+1} 进行预测...")
    _, fold_probs = model.predict(X_test, return_proba=True)
    test_probs += fold_probs.flatten()  # 确保是一维数组

# 平均预测概率
test_probs /= len(models)
test_preds = (test_probs >= 0.5).astype(int)

# 评估最终性能
logger.info("分类报告:")
logger.info(classification_report(y_test, test_preds))

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, test_probs)
roc_auc = auc(fpr, tpr)

# 计算PR曲线
precision, recall, _ = precision_recall_curve(y_test, test_probs)
pr_auc = auc(recall, precision)

# 计算混淆矩阵
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig(f"{experiment_dir}/confusion_matrix.png")
plt.close()

# 计算其他评估指标
metrics = {
    'ROC AUC': roc_auc,
    'PR AUC': pr_auc,
    'F1 Score': f1_score(y_test, test_preds),
    'Precision': precision_score(y_test, test_preds),
    'Recall': recall_score(y_test, test_preds),
    'Average Precision': average_precision_score(y_test, test_probs)
}

# 保存评估指标
pd.DataFrame(metrics.items(), columns=['Metric', 'Value']).to_csv(
    f"{experiment_dir}/evaluation_metrics.csv", index=False
)

# 可视化ROC曲线和PR曲线
plt.figure(figsize=(15, 6))

# ROC曲线
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")

# PR曲线
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR曲线')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig(f"{experiment_dir}/roc_pr_curves.png")
plt.close()

# 记录最终评估结果
logger.info("\n===== 最终评估结果 =====")
for metric, value in metrics.items():
    logger.info(f"{metric}: {value:.4f}")

# 记录总训练时间
total_time = time.time() - start_time
logger.info(f"\n总训练时间: {total_time:.2f}秒")
log_memory_usage()
