import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Add, Input, Multiply, Lambda
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import os
from sklearn.preprocessing import QuantileTransformer


class Model:
    def __init__(self, input_dim, complexity='medium'):
        self.input_dim = input_dim
        self.complexity = complexity
        self.feature_importance = {}
        
        # 使用KerasModel构建模型
        inputs = Input(shape=(input_dim,))
        
        # 特征分组
        n_features = input_dim
        financial_features = inputs[:, :3]  # 前3个特征为金融特征
        risk_features = inputs[:, 3:]  # 剩余为风险特征
        
        # 金融特征注意力
        financial_attention = Dense(3,
                                  activation='sigmoid',
                                  kernel_regularizer=l2(2e-3))(financial_features)
        financial_weighted = Multiply()([financial_features, financial_attention])
        
        # 风险特征注意力
        risk_attention = Dense(n_features-3,
                             activation='sigmoid',
                             kernel_regularizer=l2(2e-3))(risk_features)
        risk_weighted = Multiply()([risk_features, risk_attention])
        
        # 特征融合
        x = tf.keras.layers.Concatenate()([
            financial_weighted,
            risk_weighted
        ])
        
        # 特征交互层
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # 根据复杂度选择网络结构
        if complexity == 'high':
            x = self._residual_block(x, 1024, dropout_rate=0.5)
            x = self._residual_block(x, 512, dropout_rate=0.5)
            x = self._residual_block(x, 256, dropout_rate=0.5)
        elif complexity == 'medium':
            x = self._residual_block(x, 256, dropout_rate=0.4)
            x = self._residual_block(x, 128, dropout_rate=0.4)
        else:
            x = self._residual_block(x, 128, dropout_rate=0.3)
        
        # 输出层
        outputs = Dense(1, activation='sigmoid')(x)
        
        # 构建模型
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                weight_decay=1e-3
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # 添加特征重要性跟踪
        self.feature_importance_history = []

    def _weighted_binary_crossentropy(self, y_true, y_pred):
        """改进的加权二分类交叉熵损失函数"""
        # 计算基础交叉熵
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # 动态调整类别权重
        if hasattr(self, 'class_weights'):
            weights = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)
            indices = tf.cast(y_true, tf.int32)
            weights = tf.gather(weights, indices)
            bce = bce * weights
            
        return tf.reduce_mean(bce)

    def _one_cycle_schedule(self, epoch, steps_per_epoch):
        """自定义OneCycle学习率调度"""
        total_steps = steps_per_epoch * self.epochs
        current_step = epoch * steps_per_epoch
        
        pct = current_step / total_steps
        pct_start = 0.2
        
        if pct < pct_start:
            return self.max_lr * (pct / pct_start)
        elif pct < 0.4:
            return self.max_lr
        else:
            return self.max_lr * (1 - (pct - 0.4) / 0.6)

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, class_weight=None, callbacks=None):
        # 设置类别权重
        if class_weight is not None:
            self.class_weights = class_weight
        
        # OneCycle调度器配置
        self.epochs = epochs
        self.max_lr = 1e-3
        steps_per_epoch = len(x_train) // batch_size
        
        # 调整学习率和早停策略
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=20,  # 增加耐心值
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.2,  # 更激进的学习率衰减
                patience=10,
                mode='max',
                min_lr=1e-6
            )
        ]
        
        # 训练执行
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )
        return history

    def _track_feature_importance(self, epoch, logs):
        """跟踪特征重要性"""
        for layer in self.model.layers:
            if 'attention' in layer.name:
                importance = np.mean(layer.get_weights()[0], axis=1)
                self.feature_importance_history.append(importance)
        
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.feature_importance_history:
            return None
        return np.mean(self.feature_importance_history, axis=0)

    def _residual_block(self, inputs, units, dropout_rate):
        shortcut = inputs
        if inputs.shape[-1] != units:
            shortcut = Dense(units, use_bias=False, 
                           kernel_initializer='he_normal')(shortcut)
        
        x = Dense(units, 
                 kernel_constraint=max_norm(3.0),
                 kernel_regularizer=l2(1e-4),
                 use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    def evaluate(self, X_test, y_test):
        """增强的评估方法"""
        # 预测结果
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # 计算评估指标
        metrics = {
            'accuracy': np.mean(y_pred_binary == y_test),
            'auc': roc_auc_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred_binary),
            'recall': recall_score(y_test, y_pred_binary),
            'f1': f1_score(y_test, y_pred_binary)
        }
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_binary))
        
        return metrics

    def predict(self, X_new, threshold=0.5, return_proba=False):
        """优化预测方法"""
        y_pred = self.model.predict(X_new, verbose=0)
        
        if return_proba:
            return (y_pred >= threshold).astype(int), y_pred
        else:
            return (y_pred >= threshold).astype(int)

    def save(self, filepath):
        """保存模型权重"""
        self.model.save_weights(filepath)
        
    def load(self, filepath):
        """加载模型权重"""
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
        else:
            raise FileNotFoundError(f"模型权重文件不存在: {filepath}")

    def apply_feature_transformation(self, X_resampled):
        """应用特征变换"""
        scaler = QuantileTransformer(output_distribution='normal')
        X_scaled = scaler.fit_transform(X_resampled)
        return X_scaled

    def weighted_ensemble(self, models, X_test, y_test):
        """加权集成"""
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # 可以根据验证集表现调整权重
        test_probs = np.zeros(len(y_test))
        for i, (model, weight) in enumerate(zip(models, weights)):
            _, fold_probs = model.predict(X_test, return_proba=True)
            test_probs += fold_probs.flatten() * weight
        
        return test_probs

