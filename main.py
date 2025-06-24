from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import os

# 导入自定义模块
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer
from prediction_model import PredictionModel

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TaxiGPSAnalyzer:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer()
        self.predictor = PredictionModel()

    def process_file(self, file, min_long, max_long, min_lati, max_lati, max_speed):
        # 创建临时文件夹存储图表
        temp_dir = "temp_plots"
        os.makedirs(temp_dir, exist_ok=True)

        # 读取CSV文件
        df = self.cleaner.load_data(file.name)
        print(f"成功读取文件，共{len(df)}行数据")

        # 数据清洗
        df_cleaned = self.cleaner.clean_data(df, min_long, max_long, min_lati, max_lati, max_speed)
        print(f"数据清洗完成，保留{len(df_cleaned)}行数据")

        # 提取OD数据
        od_data = self.analyzer.extract_od_data(df_cleaned)
        print(f"OD数据提取完成，共{len(od_data)}对OD数据")

        # 热点聚类分析
        hotspots, n_clusters = self.analyzer.cluster_pickup_points(od_data)
        print(f"热点聚类分析完成，发现{n_clusters}个簇")

        # 时间分布分析
        time_dist = self.analyzer.analyze_time_distribution(od_data)
        print("时间分布分析完成")

        # 速度分析
        speed_data = self.analyzer.calculate_average_speed(od_data)
        print("速度分析完成")

        # 载客数量分析
        occupied_data = self.analyzer.count_occupied_taxis(od_data)
        print("载客数量分析完成")

        # 距离分析
        distance_data = self.analyzer.analyze_trip_distance(od_data)
        print("距离分析完成")

        # 乘客需求预测
        #print(od_data.head().to_dict())
        self.predictor.fit(od_data)
        predict_data = self.predictor.predict_od_orders(datetime(1900, 1, 1) , od_data)
        # print(predict_data)
        print("乘客需求预测完成")

        # 订单特征分析
        order_count = self.analyzer.analyze_order_features(od_data)
        print("订单特征分析完成")

        # 订单预测
        order_predictions = self.analyzer.predict_orders(od_data)
        print("订单特征分析完成")


        # 生成可视化结果
        gps_plot_path = os.path.join(temp_dir, "gps_plot.png")
        self.visualizer.plot_gps_points(df_cleaned, gps_plot_path)

        hotspots_plot_path = os.path.join(temp_dir, "hotspots_plot.png")
        self.visualizer.plot_hotspots(hotspots, hotspots_plot_path)

        time_plot_path = os.path.join(temp_dir, "time_plot.png")
        self.visualizer.plot_time_distribution(time_dist, time_plot_path)

        speed_plot_path = os.path.join(temp_dir, "speed_plot.png")
        self.visualizer.plot_speed_by_hour(speed_data, speed_plot_path)

        occupied_plot_path = os.path.join(temp_dir, "occupied_plot.png")
        self.visualizer.plot_occupied_taxis(occupied_data, occupied_plot_path)

        distance_plot_path = os.path.join(temp_dir, "distance_plot.png")
        self.visualizer.plot_distance_distribution(distance_data, distance_plot_path)

        #demand_plot_path = os.path.join(temp_dir, "demand_plot.png")
        #self.visualizer.plot_demand_prediction(demand_prediction, demand_plot_path)

        prediction_heatmap_path = os.path.join(temp_dir, "prediction_heatmap.png")
        self.visualizer.plot_order_prediction_summary(order_predictions, prediction_heatmap_path)

        order_heatmap_path = os.path.join(temp_dir, "order_heatmap.png")
        self.visualizer.plot_order_count_heatmap(order_count, order_heatmap_path)


        # 生成分析摘要
        summary = {
            "数据量": len(df),
            "清洗后数据量": len(df_cleaned),
            "OD对数量": len(od_data),
            "热点簇数量": n_clusters,
            "平均行程距离": f"{od_data['OD_Dis_km'].mean():.2f} km",
            "平均行程时间": f"{od_data['OD_TIME_s'].mean() / 60:.2f} 分钟",
            "平均行驶速度": f"{od_data['OD_Dis_km'].sum() / (od_data['OD_TIME_s'].sum() / 3600):.2f} km/h",
        }

        # 确保所有图像都存在
        for path in [
            gps_plot_path, hotspots_plot_path, time_plot_path,
            speed_plot_path, occupied_plot_path, distance_plot_path,
            '''demand_plot_path''', order_heatmap_path, prediction_heatmap_path
        ]:
            if not os.path.exists(path):
                print(f"警告: 文件不存在: {path}")

        return summary, gps_plot_path, hotspots_plot_path, time_plot_path, speed_plot_path, occupied_plot_path, distance_plot_path, '''demand_plot_path''', order_heatmap_path, prediction_heatmap_path

def create_interface():  # Gradio
    analyzer = TaxiGPSAnalyzer()

    with gr.Blocks(title="出租车GPS数据分析系统") as iface:
        gr.Markdown("# 出租车GPS数据时空特征提取及可视化系统")
        gr.Markdown("上传包含出租车GPS数据的CSV文件，系统将进行数据清洗、分析及可视化。")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="上传CSV文件")

                with gr.Accordion("清洗参数设置", open=False):
                    min_long_input = gr.Number(label="最小经度", value=113.75)
                    max_long_input = gr.Number(label="最大经度", value=114.6)
                    min_lati_input = gr.Number(label="最小纬度", value=22.4)
                    max_lati_input = gr.Number(label="最大纬度", value=22.85)
                    max_speed_input = gr.Number(label="最大速度 (km/h)", value=120)

                submit_btn = gr.Button("开始分析")
                summary_output = gr.JSON(label="分析摘要")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("GPS点分布"):
                        gps_plot = gr.Image(label="GPS点分布图")

                    with gr.TabItem("热门上客点"):
                        hotspots_plot = gr.Image(label="热门上客点热力图")

                    with gr.TabItem("时间分布"):
                        time_plot = gr.Image(label="乘客打车时间分布")

                    with gr.TabItem("道路速度"):
                        speed_plot = gr.Image(label="城市道路平均速度变化")

                    with gr.TabItem("载客数量"):
                        occupied_plot = gr.Image(label="载客出租车数量变化")

                    with gr.TabItem("出行距离"):
                        distance_plot = gr.Image(label="出行距离分布")

                    with gr.TabItem("需求预测"):
                        demand_plot = gr.Image(label="乘客需求预测")

                    with gr.TabItem("订单热力图"):
                        order_heatmap = gr.Image(label="订单热力图")

                    with gr.TabItem("预测热力图"):
                        prediction_heatmap = gr.Image(label="预测热力图")

        submit_btn.click(
            fn=analyzer.process_file,
            inputs=[file_input, min_long_input, max_long_input, min_lati_input, max_lati_input, max_speed_input],
            outputs=[summary_output, gps_plot, hotspots_plot, time_plot, speed_plot, occupied_plot, distance_plot,
                     demand_plot, order_heatmap, prediction_heatmap]
        )

    return iface


if __name__ == "__main__":
    iface = create_interface()
    iface.launch()