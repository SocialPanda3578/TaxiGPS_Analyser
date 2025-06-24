import pandas as pd
import os
from datetime import datetime, timedelta
import math
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point


class DataAnalyzer:
    def __init__(self):
        pass

    def extract_od_data(self, df):
        """提取OD数据 (Origin-Destination)"""
        # 按车辆ID分组
        grouped = df.groupby('id')
        od_pairs = []

        for taxi_id, group in grouped:
            # 按时间排序
            group = group.sort_values('time')

            # 寻找状态从0变为1的点(上客点)和从1变为0的点(下客点)
            status_changes = group['status'].diff().fillna(0)
            pickup_points = group[status_changes == 1]  # 上客点
            dropoff_points = group[status_changes == -1]  # 下客点

            # 确保上客点和下客点数量匹配
            min_points = min(len(pickup_points), len(dropoff_points))

            for i in range(min_points):
                pickup = pickup_points.iloc[i]
                dropoff = dropoff_points.iloc[i]

                # 计算OD对的时间(秒)和距离(千米)
                time_diff = (dropoff['time'] - pickup['time']).total_seconds()

                # 确保时间差为正，否则跳过此OD对
                if time_diff <= 0:
                    continue

                # 使用哈弗辛公式计算两点间距离
                distance = self.haversine(pickup['long'], pickup['lati'],
                                          dropoff['long'], dropoff['lati'])

                # 确保距离为正，否则跳过此OD对
                if distance <= 0:
                    continue

                od_pair = {
                    'O_COMMADDR': taxi_id,
                    'O_time': pickup['time'],
                    'O_lat': pickup['lati'],
                    'O_lng': pickup['long'],
                    'O_HEAD': pickup.get('head', 0),  # 如果存在
                    'O_SPEED': pickup['speed'],
                    'O_FLAG': pickup['status'],
                    'D_time': dropoff['time'],
                    'D_lat': dropoff['lati'],
                    'D_lng': dropoff['long'],
                    'D_HEAD': dropoff.get('head', 0),  # 如果存在
                    'D_SPEED': dropoff['speed'],
                    'D_FLAG': dropoff['status'],
                    'OD_TIME_s': time_diff,
                    'OD_Dis_km': distance
                }

                od_pairs.append(od_pair)

        return pd.DataFrame(od_pairs)

    def haversine(self, lon1, lat1, lon2, lat2):
        """计算两点间的距离(千米)"""
        # 将经纬度转换为弧度
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # 哈弗辛公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # 地球半径(千米)
        return c * r

    def cluster_pickup_points(self, od_data, eps=0.01, min_samples=10):
        """对上客点进行密度聚类"""
        # 提取上客点坐标
        pickup_coords = od_data[['O_lng', 'O_lat']].values

        # 使用DBSCAN进行聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pickup_coords)
        labels = db.labels_

        # 添加聚类标签到数据中
        od_data.loc[:, 'cluster'] = labels

        # 计算聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 计算每个簇的中心点和热力值
        hotspots = []

        # 按时间段分组
        od_data.loc[:, 'hour'] = od_data['O_time'].dt.hour
        time_groups = od_data.groupby('hour')
        for hour, hour_group in time_groups:
            # 对每个时间段内的簇进行分析
            for cluster_id in set(hour_group['cluster']):
                if cluster_id != -1:  # 排除噪声点
                    cluster_points = hour_group[hour_group['cluster'] == cluster_id]

                    # 计算簇的中心点(平均经纬度)
                    center_lng = cluster_points['O_lng'].mean()
                    center_lat = cluster_points['O_lat'].mean()

                    # 计算热力值(簇中点的数量)
                    count = len(cluster_points)

                    hotspots.append({
                        'lng': center_lng,
                        'lat': center_lat,
                        'count': count,
                        'time': f"{hour:02d}:00"
                    })

        return pd.DataFrame(hotspots), n_clusters

    def analyze_time_distribution(self, od_data, interval='15min'):
        """分析乘客打车的时间分布"""
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(od_data['O_time']):
            od_data.loc[:, 'O_time'] = pd.to_datetime(od_data['O_time'])

        # 设置时间索引
        od_time_indexed = od_data.set_index('O_time')

        # 按指定时间间隔重采样并计数
        time_distribution = od_time_indexed.resample(interval).size().reset_index()
        time_distribution.columns = ['O_time', 'count']

        return time_distribution

    def calculate_average_speed(self, od_data):
        """计算订单的平均速度"""
        # 计算平均速度 (km/h)
        od_data.loc[:, 'avg_speed'] = od_data['OD_Dis_km'] / (od_data['OD_TIME_s'] / 3600)

        # 过滤掉不合理的速度值
        od_data = od_data[od_data['avg_speed'] < 120]  # 假设最高速度限制为120km/h

        # 按小时分组计算平均速度
        od_data.loc[:, 'hour'] = od_data['O_time'].dt.hour
        hourly_speed = od_data.groupby('hour')['avg_speed'].mean().reset_index()
        hourly_speed.columns = ['O_time', 'sudu']

        return hourly_speed

    def count_occupied_taxis(self, od_data, freq='1min'):
        """统计每分钟载客出租车数量（支持多车辆）"""
        import datetime
        today = datetime.datetime.now().date()
        od_data['O_time'] = od_data['O_time'].apply(
            lambda x: datetime.datetime.combine(today, x.time())
        )
        od_data['D_time'] = od_data['D_time'].apply(
            lambda x: datetime.datetime.combine(today, x.time())
        )

        # 2. 动态生成时间范围（覆盖所有订单）
        min_time = od_data['O_time'].min().floor(freq)  # 向下取整到分钟
        max_time = od_data['D_time'].max().ceil(freq)   # 向上取整到分钟
        time_range = pd.date_range(start=min_time, end=max_time, freq=freq)

        # 3. 初始化计数DataFrame
        occupied_count = pd.DataFrame(index=time_range, data={'number': 0})

        # 4. 按车辆分组处理载客时段
        for taxi_id, group in od_data.groupby('O_COMMADDR'):
            # 合并同一车辆的连续/重叠载客时段
            intervals = []
            for _, row in group.sort_values('O_time').iterrows():
                start, end = row['O_time'], row['D_time']
                if intervals and start <= intervals[-1][1]:
                    # 重叠时段，合并（延长结束时间）
                    intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
                else:
                    intervals.append((start, end))

            # 标记载客时段（累加计数）
            for start, end in intervals:
                mask = (occupied_count.index >= start) & (occupied_count.index < end)
                occupied_count.loc[mask, 'number'] += 1

        # 5. 重置索引并返回
        return occupied_count.reset_index().rename(columns={'index': 'TIME'})

    def analyze_trip_distance(self, od_data):
        """分析出行距离分布"""
        # 定义距离类别
        od_data.loc[:, 'distance_category'] = pd.cut(
            od_data['OD_Dis_km'],
            bins=[0, 4, 8, float('inf')],
            labels=['near', 'middle', 'far']
        )

        # 按日期分组统计各类别数量
        od_data.loc[:, 'day'] = od_data['O_time'].dt.day
        distance_stats = od_data.groupby('day')['distance_category'].value_counts().unstack().fillna(0)

        # 转换为整数
        distance_stats = distance_stats.astype(int)

        # 重置索引
        distance_stats = distance_stats.reset_index()

        return distance_stats

    def get_region(self, lng, lat, region_data):
        point = Point(lng, lat)
        # print(region_data.columns)
        for index, row in region_data.iterrows():
            if row['geometry'].contains(point):
                return row['qh']  # 区号
        return None

    def analyze_order_features(self, od_data):
        """分析订单特征，统计两个区域之间的订单数量"""
        sz_path = sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        region_data = gpd.read_file(sz_path, encoding='utf8')

        od_data['O_region'] = od_data.apply(lambda row: self.get_region(row['O_lng'], row['O_lat'], region_data),
                                            axis=1)
        od_data['D_region'] = od_data.apply(lambda row: self.get_region(row['D_lng'], row['D_lat'], region_data),
                                            axis=1)

        order_count = od_data.groupby(['O_region', 'D_region']).size().reset_index(name='count')
        return order_count

    def predict_orders(self, od_data):
        """分小时预测订单需求"""
        od_data['hour'] = od_data['O_time'].dt.hour
        hourly_orders = od_data.groupby(['hour', 'O_region', 'D_region']).size().reset_index(name='count')

        predictions = {}
        for hour in hourly_orders['hour'].unique():
            hour_orders = hourly_orders[hourly_orders['hour'] == hour]
            total_demand = hour_orders['count'].sum()
            hour_predictions = []
            for _, row in hour_orders.iterrows():
                hour_predictions.append({
                    'origin': row['O_region'],
                    'destination': row['D_region'],
                    'demand': row['count']
                })
            predictions[hour] = {
                'total_demand': total_demand,
                'orders': hour_predictions
            }
        return predictions