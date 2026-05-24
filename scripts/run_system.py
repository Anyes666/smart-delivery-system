# run_system.py
"""
智能配送系统启动脚本
此脚本提供了一个简单的接口来运行整个配送优化系统，
可以通过命令行参数或交互式菜单选择不同的配置。
 """
import argparse
import sys  
import os
from pathlib import Path # 导入 Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ → 项目根目录
sys.path.append(project_root)

# 导入主系统类
from main import DeliverySystem

def parse_arguments():
                        """解析命令行参数"""
                        parser = argparse.ArgumentParser(description='智能配送系统')
                        parser.add_argument(
                            '--data-file', '-d',
                            type=str,
                            default='data/my_points1.csv', # 默认使用这个文件
                            help='输入数据 CSV 文件路径 (相对于项目根目录), 默认: data/points.csv'
                        )
                        parser.add_argument(
                            '--num-vehicles', '-v',
                            type=int,
                            default=3,
                            help='车辆数量, 默认: 3'
                        )
                        parser.add_argument(
                            '--algorithm', '-a',
                            type=str,
                            choices=['ortools', 'greedy', 'cluster'],
                            default='ortools',
                            help='使用的算法: ortools, greedy, cluster. 默认: ortools'
                        )
                        parser.add_argument(
                            '--output-dir', '-o',
                            type=str,
                            default='output',
                            help='输出目录, 默认: output'
                        )
                        parser.add_argument(
                            '--depot-index', '-dp',
                            type=int,
                            default=0, # 假设配送中心是第一个点
                            help='配送中心在 CSV 文件中的索引 (从0开始), 默认: 0'
                        )
                        parser.add_argument(
                            '--road-network', '-rn',
                            action='store_true',
                            default=False,
                            help='使用 OSM 真实路网规划路径 (需要网络下载路网数据)'
                        )
                        parser.add_argument(
                            '--place-name', '-pn',
                            type=str,
                            default='Shanghai, China',
                            help='路网下载区域名称, 默认: Shanghai, China (需 --road-network 启用)'
                        )
                        parser.add_argument(
                            '--dark-theme',
                            action='store_true',
                            default=False,
                            help='使用暗色科技风地图主题 (需 --road-network 启用)'
                        )
                        parser.add_argument(
                            '--simulate-congestion',
                            action='store_true',
                            default=True,
                            help='模拟交通拥堵 (默认启用, 需 --road-network)'
                        )
                        parser.add_argument(
                            '--accidents',
                            type=int,
                            default=2,
                            help='随机交通事故数量, 默认: 2'
                        )
                        parser.add_argument(
                            '--simulation-time',
                            type=str,
                            default='10:00',
                            help='模拟时间 HH:MM, 默认: 10:00'
                        )
                        parser.add_argument(
                            '--amap-key', '-ak',
                            type=str,
                            default=None,
                            help='高德 (Amap) Web API Key (自动启用高德路网+瓦片)'
                        )
                        parser.add_argument(
                            '--real-time', '-rt',
                            action='store_true',
                            default=False,
                            help='使用当前真实时间作为模拟时间 (覆盖 --simulation-time)'
                        )
                        parser.add_argument(
                            '--simulated-collection',
                            action='store_true',
                            default=False,
                            help='标记本次采集为非真实时段 (供数据采集器使用)'
                        )
                        return parser.parse_args()

def run_with_args(args):
                        """使用命令行参数运行系统"""
                        from config import settings

                        # Handle Amap key first (affects provider)
                        if args.amap_key:
                            settings.ROAD_NETWORK_PROVIDER = 'amap'
                            settings.AMAP_CONFIG['api_key'] = args.amap_key
                            settings.TILE_PROVIDER = 'amap'

                        use_road_network = args.road_network or (args.amap_key is not None)
                        # Auto-enable if Amap key is already in settings
                        if not use_road_network:
                            ak = settings.AMAP_CONFIG.get('api_key', '')
                            if ak and ak != 'YOUR_AMAP_KEY':
                                use_road_network = True

                        # Parse simulation time
                        if args.real_time:
                            from datetime import datetime
                            now = datetime.now()
                            hh, mm = now.hour, now.minute
                            settings.DEFAULT_SIMULATION_TIME = hh * 3600 + mm * 60
                            args.simulation_time = f"{hh:02d}:{mm:02d}"
                            print(f"[INFO] --real-time: 使用当前时间 {args.simulation_time}")
                        else:
                            try:
                                parts = args.simulation_time.split(':')
                                hh, mm = int(parts[0]), int(parts[1])
                                settings.DEFAULT_SIMULATION_TIME = hh * 3600 + mm * 60
                            except (ValueError, IndexError):
                                pass

                        # Phase 2+3 config
                        settings.VISUALIZATION_CONFIG['use_dark_theme'] = args.dark_theme
                        if not args.simulate_congestion:
                            settings.CONGESTION_CONFIG['n_random_accidents'] = 0
                        else:
                            settings.CONGESTION_CONFIG['n_random_accidents'] = args.accidents

                        # Simulated collection flag (for data collector)
                        settings.SIMULATED_COLLECTION = args.simulated_collection

                        # Display config
                        provider = settings.ROAD_NETWORK_PROVIDER
                        print("=" * 60)
                        print("  Smart Delivery System")
                        print("=" * 60)
                        original_path = Path(args.data_file)
                        filename_only = original_path.name
                        print(f"  Data:        {filename_only}")
                        print(f"  Vehicles:    {args.num_vehicles}")
                        print(f"  Algorithm:   {args.algorithm}")
                        print(f"  Output:      {args.output_dir}")
                        print(f"  Road net:    {'Amap API' if provider == 'amap' else 'OSMnx' if use_road_network else 'Euclidean'}")
                        if provider == 'amap':
                            ak = settings.AMAP_CONFIG.get('api_key', '')
                            print(f"  Amap key:    {'***configured***' if ak and ak != 'YOUR_AMAP_KEY' else 'NOT SET'}")
                        print(f"  Map tiles:   {'Amap' if settings.TILE_PROVIDER == 'amap' else 'CartoDB'}")
                        print(f"  Sim time:    {args.simulation_time}")
                        if args.simulate_congestion:
                            print(f"  Incidents:   up to {args.accidents} (~10% probability each)")
                        else:
                            print(f"  Incidents:   disabled")
                        print("=" * 60)

                        system = DeliverySystem(use_road_network=use_road_network)

                        success = system.run_full_pipeline(
                            filepath=filename_only,
                            num_vehicles=args.num_vehicles,
                            algorithm=args.algorithm,
                            output_dir=args.output_dir,
                        )

                        if success:
                            print("\nDone.")
                        else:
                            print("\nFailed.")

def interactive_menu():
                        """Interactive menu for the Smart Delivery System."""
                        print("=" * 60)
                        print("    Smart Delivery System - Interactive Mode")
                        print("=" * 60)

                        from src.utils.data_processing import list_available_data_files
                        from config import settings

                        # ── 1. Data file ──
                        try:
                            available_files = list_available_data_files()
                            if not available_files:
                                print("[ERROR] No CSV files found in data/ directory.")
                                return
                            print(f"\nAvailable data files: {available_files}")
                        except Exception as e:
                            print(f"[ERROR] Cannot list data files: {e}")
                            return

                        while True:
                            try:
                                choice = input("Choose CSV file: ").strip()
                                if choice in available_files:
                                    filename_to_pass = choice
                                    print(f"  -> {filename_to_pass}")
                                    break
                                else:
                                    print(f"  File '{choice}' not found. Choose from the list.")
                            except KeyboardInterrupt:
                                print("\nCancelled.")
                                return

                        # ── 2. Max vehicles available ──
                        print(f"\n车队规模 (单车容量={settings.VEHICLE_CAPACITY}, "
                              f"启用成本={settings.VEHICLE_FIXED_COST/1000:.0f}km等效)")
                        print("  算法会自动决定实际使用几辆, 仅在省下的距离>启用成本时才多派车")
                        try:
                            num_vehicles = int(input("最大可用车辆数 (default 5): ") or "5")
                        except ValueError:
                            num_vehicles = 5
                            print("  Using default: 5")

                        # ── 3. Algorithm ──
                        print("\nAlgorithm:")
                        print("  1. OR-Tools [recommended]")
                        print("  2. Greedy")
                        print("  3. Cluster-first")
                        algo_choice = input("Choose (1/2/3, default 1): ").strip()
                        algo_map = {'1': 'ortools', '2': 'greedy', '3': 'cluster'}
                        algorithm = algo_map.get(algo_choice, 'ortools')

                        # ── 4. Output dir ──
                        output_dir = input("\nOutput directory (default 'output'): ").strip() or "output"

                        # ── 5. Road network (auto from settings) ──
                        provider = getattr(settings, 'ROAD_NETWORK_PROVIDER', 'osmnx')
                        use_road_network = settings.USE_REAL_ROAD_NETWORK
                        if provider == 'amap':
                            key = settings.AMAP_CONFIG.get('api_key', '')
                            has_key = key and key != 'YOUR_AMAP_KEY'
                            if has_key:
                                use_road_network = True
                                print(f"\nRoad network: Amap (API key configured) [auto]")
                            else:
                                print(f"\nRoad network: Amap (no API key -- Euclidean fallback)")
                                use_road_network = False
                        else:
                            choice = input("\nUse OSM road network? (y/N): ").lower() == 'y'
                            use_road_network = choice
                            if choice:
                                place = input("Place name (default 'Shanghai, China'): ").strip() or "Shanghai, China"
                                settings.ROAD_NETWORK_CONFIG['place_name'] = place

                        # ── 6. Simulation time ──
                        choice = input("\nSimulation time: 输入 HH:MM 或 直接回车=当前真实时间: ").strip()
                        if choice:
                            try:
                                parts = choice.split(':')
                                hh, mm = int(parts[0]), int(parts[1])
                                settings.DEFAULT_SIMULATION_TIME = hh * 3600 + mm * 60
                                sim_time = choice
                            except (ValueError, IndexError):
                                settings.DEFAULT_SIMULATION_TIME = 36000
                                sim_time = "10:00"
                        else:
                            from datetime import datetime
                            now = datetime.now()
                            hh, mm = now.hour, now.minute
                            settings.DEFAULT_SIMULATION_TIME = hh * 3600 + mm * 60
                            sim_time = f"{hh:02d}:{mm:02d}"
                            print(f"  使用当前真实时间: {sim_time}")

                        # ── 7. Map tiles ──
                        tile_choice = input("Use Amap tiles? (Y/n, default Y): ").strip().lower()
                        settings.VISUALIZATION_CONFIG['use_dark_theme'] = (tile_choice != 'n')

                        # ── 8. Confirm ──
                        print("\n" + "=" * 60)
                        print("Configuration:")
                        print(f"  Data:        {filename_to_pass}")
                        print(f"  Vehicles:    {num_vehicles}")
                        print(f"  Algorithm:   {algorithm}")
                        print(f"  Output:      {output_dir}")
                        print(f"  Road net:    {'Enabled' if use_road_network else 'Euclidean'}")
                        print(f"  Sim time:    {sim_time}")
                        print("=" * 60)

                        confirm = input("\nStart? (y/N): ").lower()
                        if confirm != 'y':
                            print("Cancelled.")
                            return

                        system = DeliverySystem(use_road_network=use_road_network)
                        success = system.run_full_pipeline(
                            filepath=filename_to_pass,
                            num_vehicles=num_vehicles,
                            algorithm=algorithm,
                            output_dir=output_dir,
                        )

                        if success:
                            print(f"\nDone! Output files in '{output_dir}/'")
                        else:
                            print(f"\nFailed. Check errors above.")

def main():
                        """主函数"""
                        try:
                            # 如果用户传入了命令行参数（除脚本名外），直接运行；否则进入交互模式
                            if len(sys.argv) > 1:
                                args = parse_arguments()
                                run_with_args(args)
                            else:
                                print("Tip: Use command-line flags, e.g.:")
                                print("     python run_system.py -d shanghai_20_houses.csv -v 3")
                                print("     python run_system.py -d provincial_capitals.csv -v 5 -a ortools")
                                print("     python run_system.py --amap-key YOUR_KEY")
                                print("\nOr use interactive menu:")
                                interactive_menu()
                        except SystemExit:
                            # argparse 抛出 SystemExit 时（如 --help），直接退出
                            pass
                        except Exception as e:
                            print(f"[ERROR] {e}")
                            import traceback
                            traceback.print_exc()

if __name__ == "__main__":
    main()