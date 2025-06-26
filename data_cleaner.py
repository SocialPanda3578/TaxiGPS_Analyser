import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os

class DataCleaner:
    def load_data(self, file_path):
        if 'TaxiData1e6.csv' in file_path:
            df = pd.read_csv(file_path, header=1)
            df = df.iloc[:, 1:]
            df.columns = ['id', 'time', 'long', 'lati', 'status', 'speed']
        else:
            df = pd.read_csv(file_path, header=None, names=['id', 'time', 'long', 'lati', 'status', 'speed'])
        return df

    def filter_by_shenzhen_boundary(self, df):
        """使用深圳市行政区划边界过滤数据点"""
        # 读取深圳市行政区划边界数据
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf8')
        
        # 创建GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['long'], df['lati'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # 空间查询：判断点是否在深圳市边界内
        # 确保坐标系统一致
        gdf.crs = sz.crs
        
        # 使用空间连接查找在边界内的点
        joined = gpd.sjoin(gdf, sz, how='inner', predicate='within')
        
        # 返回在边界内的原始数据
        filtered_df = df.loc[joined.index]
        print(f"空间过滤：从{len(df)}行数据中保留{len(filtered_df)}行在深圳市边界内的数据")
        return filtered_df
        
    def filter_by_grid(self, df, grid_size=0.01):
        """使用栅格图过滤数据点
        
        参数:
            df: 数据框
            grid_size: 栅格大小（经纬度单位）
        """
        if len(df) == 0:
            return df
            
        # 读取深圳市行政区划边界数据
        sz_path = os.path.join(os.path.dirname(__file__), 'sz', 'sz.shp')
        sz = gpd.read_file(sz_path, encoding='utf8')
        
        # 获取深圳市边界的范围
        minx, miny, maxx, maxy = sz.total_bounds
        
        # 创建栅格
        x_grid = np.arange(minx, maxx, grid_size)
        y_grid = np.arange(miny, maxy, grid_size)
        
        # 将数据点分配到栅格中
        df['grid_x'] = pd.cut(df['long'], bins=x_grid, labels=False)
        df['grid_y'] = pd.cut(df['lati'], bins=y_grid, labels=False)
        
        # 找出有效栅格（与深圳市边界相交的栅格）
        valid_grids = set()
        
        # 创建栅格多边形并检查是否与深圳市边界相交
        from shapely.geometry import box
        
        for i in range(len(x_grid)-1):
            for j in range(len(y_grid)-1):
                grid_box = box(x_grid[i], y_grid[j], x_grid[i+1], y_grid[j+1])
                for geom in sz.geometry:
                    if grid_box.intersects(geom):
                        valid_grids.add((i, j))
                        break
        
        # 过滤数据点，只保留在有效栅格内的点
        valid_mask = df.apply(lambda row: (row['grid_x'], row['grid_y']) in valid_grids 
                             if not pd.isna(row['grid_x']) and not pd.isna(row['grid_y']) else False, axis=1)
        
        filtered_df = df[valid_mask].copy()
        # 删除辅助列
        filtered_df = filtered_df.drop(columns=['grid_x', 'grid_y'])
        
        print(f"栅格过滤：从{len(df)}行数据中保留{len(filtered_df)}行在有效栅格内的数据")
        return filtered_df
    
    def clean_data(self, df, min_long=113.75, max_long=114.65, min_lati=22.4, max_lati=22.85, max_speed=120):
        df_cleaned = df.copy()
        
        # 确保按id和time排序，这是后续处理的基础
        df_cleaned['time'] = pd.to_datetime(df_cleaned['time'], format='%H:%M:%S')
        df_cleaned = df_cleaned.sort_values(by=['id', 'time']).reset_index(drop=True)

        # 找到id和time都相同的重复记录
        duplicates_mask = df_cleaned.duplicated(subset=['id', 'time'], keep=False)

        # 使用深圳市行政区划边界过滤数据
        df_cleaned = self.filter_by_shenzhen_boundary(df_cleaned)
        
        # 使用栅格图过滤数据
        # df_cleaned = self.filter_by_grid(df_cleaned, grid_size=0.01)
        
        # 经纬度范围过滤（作为备用，主要依靠上面的空间过滤）
        df_cleaned = df_cleaned[(df_cleaned['long'] > min_long) & (df_cleaned['long'] < max_long)]
        df_cleaned = df_cleaned[(df_cleaned['lati'] > min_lati) & (df_cleaned['lati'] < max_lati)]
        # 速度异常过滤
        df_cleaned = df_cleaned[df_cleaned['speed'] < max_speed]
        df_cleaned = df_cleaned.dropna(subset=['time', 'long', 'lati', 'status'])
        # 处理重复记录
        processed_df_list = []
        for (id_val, time_val), group in df_cleaned[duplicates_mask].groupby(['id', 'time']):
            unique_statuses = group['status'].nunique()
            if unique_statuses == 1:
                # status相同，保留第一条
                processed_df_list.append(group.iloc[[0]])
            else:
                # status不同，需要根据状态变化决定保留哪条
                if 1 in group['status'].values:
                    processed_df_list.append(group[group['status'] == 1].iloc[[0]])
                else:
                    processed_df_list.append(group.iloc[[0]])
        
        # 将处理过的重复记录和非重复记录合并
        df_non_duplicates = df_cleaned[~duplicates_mask]
        if processed_df_list:
            df_processed_duplicates = pd.concat(processed_df_list)
            df_cleaned = pd.concat([df_non_duplicates, df_processed_duplicates]).sort_values(by=['id', 'time']).reset_index(drop=True)
        else:
            df_cleaned = df_non_duplicates.sort_values(by=['id', 'time']).reset_index(drop=True)

        # 为每个id创建一个组，并在组内进行状态比较
        df_cleaned['prev_status'] = df_cleaned.groupby('id')['status'].shift(1)
        df_cleaned['next_status'] = df_cleaned.groupby('id')['status'].shift(-1)
        # 排除NaN值（即序列的开头和结尾）
        abnormal_status_mask = (
            (df_cleaned['status'] != df_cleaned['prev_status']) & 
            (df_cleaned['status'] != df_cleaned['next_status']) & 
            (df_cleaned['prev_status'].notna()) & # 确保不是序列的第一个
            (df_cleaned['next_status'].notna())    # 确保不是序列的最后一个
        )
        # 移除异常状态的记录
        df_cleaned = df_cleaned[~abnormal_status_mask].reset_index(drop=True)
        # 最终确保按id和time排序
        df_cleaned = df_cleaned.sort_values(by=['id', 'time']).reset_index(drop=True)
        # 移除辅助列
        df_cleaned = df_cleaned.drop(columns=['prev_status', 'next_status'])
        return df_cleaned