import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class PredictionModel:

    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=1
            ))
        ])
        self.features = [
            'hour', 'minute_interval', 'dayofweek', 'month',
            'O_lat', 'O_lng', 'D_lat', 'D_lng',
            'O_SPEED', 'D_SPEED', 'OD_Dis_km',
            'avg_speed', 'distance_category'
        ]
        self.time_interval = 15
        self.is_trained = False

    def _enhance_features(self, df):
        """更鲁棒的特征工程"""
        # 基础特征
        features = {
            'hour': df['O_time'].dt.hour,
            'minute_interval': (df['O_time'].dt.minute // 15) * 15,
            'dayofweek': df['O_time'].dt.dayofweek,
            'month': df['O_time'].dt.month,
            'O_lat': df['O_lat'],
            'O_lng': df['O_lng'],
            'D_lat': df['D_lat'],
            'D_lng': df['D_lng'],
            'OD_Dis_km': df.get('OD_Dis_km', 0),
            'speed_ratio': df['O_SPEED'] / (df['D_SPEED'] + 1e-5)  # 避免除零
        }

        # 添加方向向量
        features.update({
            'lat_diff': df['D_lat'] - df['O_lat'],
            'lng_diff': df['D_lng'] - df['O_lng']
        })

        # 构建DataFrame
        X = pd.DataFrame(features)

        # 添加距离分类
        X['dist_type'] = pd.cut(
            X['OD_Dis_km'],
            bins=[0, 3, 10, float('inf')],
            labels=['short', 'medium', 'long']
        )
        X = pd.get_dummies(X, columns=['dist_type'], prefix='dist')

        return X

    def fit(self, od_data):
        """改进的训练方法"""
        df = od_data.copy()

        # 检查数据是否包含有效的订单计数
        if 'order_count' not in df.columns:
            # 如果没有order_count列，尝试从O_FLAG/D_FLAG推断
            if 'O_FLAG' in df.columns and 'D_FLAG' in df.columns:
                # 假设O_FLAG=1表示订单开始，D_FLAG=1表示订单结束
                df['order_count'] = df['O_FLAG']  # 或使用其他逻辑
            else:
                # 如果无法推断，则不能进行有监督学习
                raise ValueError(
                    "数据必须包含order_count列或O_FLAG/D_FLAG列"
                    "用于确定订单数量"
                )

        # 验证目标变量
        if df['order_count'].nunique() == 1:
            # 如果所有值相同，改为使用行程距离或时间作为目标变量
            print("警告: order_count无变化，改为预测OD_TIME_s")
            df['order_count'] = df['OD_TIME_s']  # 或其他有变化的变量

        # 特征工程
        X = self._enhance_features(df)
        y = df['order_count']

        # 模型训练
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_names_ = X.columns.tolist()

    def predict_od_orders(self, start_date, od_data):
        """改进的预测方法"""
        if not self.is_trained:
            return {"error": "模型未训练"}

        try:
            df = od_data.copy()
            df['O_time'] = pd.to_datetime(start_date)

            # 确保包含必要的列
            if 'OD_TIME_s' not in df.columns:
                df['OD_TIME_s'] = 0  # 默认值

            X_predict = self._enhance_features(df)
            X_predict = X_predict[self.feature_names_]

            predictions = self.model.predict(X_predict)

            # 根据训练目标返回适当结果
            if hasattr(self, 'predict_order_count'):
                return {
                    "time": start_date.strftime("%Y-%m-%d"),
                    "predictions": [max(0, round(p)) for p in predictions],
                    "units": "order_count"
                }
            else:
                return {
                    "time": start_date.strftime("%Y-%m-%d"),
                    "predictions": predictions.tolist(),
                    "units": "OD_TIME_s (seconds)"
                }
        except Exception as e:
            return {"error": str(e)}