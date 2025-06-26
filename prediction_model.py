import pandas as pd

class PredictionModel:
    def __init__(self):
        pass

    def predict_demand(self, historical_data: pd.DataFrame, time_period: str):
        """
        预测特定时间和地点的乘客需求。
        historical_data: 包含历史订单数据，至少包括时间、上车地点等信息。
        time_period: 预测的时间段，例如 'hourly', 'daily'。
        """
        print(f"正在预测 {time_period} 的乘客需求...")

        if historical_data.empty or 'O_time' not in historical_data.columns:
            print("历史数据为空或缺少'O_time'列，无法进行需求预测。")
            return pd.DataFrame(columns=['time_unit', 'demand'])

        # 确保 'O_time' 是 datetime 类型
        historical_data['O_time'] = pd.to_datetime(historical_data['O_time'])

        if time_period == 'hourly':
            historical_data['time_unit'] = historical_data['O_time'].dt.hour
            demand_by_unit = historical_data.groupby('time_unit').size().reset_index(name='demand')
            # 填充所有小时，确保0-23小时都有数据，没有数据的填充0
            all_hours = pd.DataFrame({'time_unit': range(24)})
            demand_by_unit = pd.merge(all_hours, demand_by_unit, on='time_unit', how='left').fillna(0)
            demand_by_unit['demand'] = demand_by_unit['demand'].astype(int)
            print("按小时预测需求完成。")
            return demand_by_unit
        elif time_period == 'daily':
            historical_data['time_unit'] = historical_data['O_time'].dt.date
            demand_by_unit = historical_data.groupby('time_unit').size().reset_index(name='demand')
            print("按天预测需求完成。")
            return demand_by_unit
        else:
            print(f"不支持的时间粒度: {time_period}。目前只支持 'hourly' 和 'daily'。")
            return pd.DataFrame(columns=['time_unit', 'demand'])