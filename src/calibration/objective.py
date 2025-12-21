import pandas as pd
import numpy as np
import sys
import os

# 确保可以导入 scripts 目录下的 common_data
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts'))
from common_data import load_sim_data, load_route_stop_dist, build_sim_trajectory, load_real_link_speeds

def calculate_l1_rmse(sim_xml_path, real_links_csv, route_stop_dist_csv, route='68X', bound='I'):
    """
    计算站点级行程时间的 RMSE (J1 指标)
    """
    # 1. 加载数据
    sim_raw = load_sim_data(sim_xml_path)
    if sim_raw.empty:
        return 1e6 # 如果没有仿真数据，返回一个极大的惩罚值
        
    dist_df = load_route_stop_dist(route_stop_dist_csv)
    # 转换方向标识以匹配
    bound_map = {'I': 'inbound', 'O': 'outbound'}
    target_bound = bound_map.get(bound, bound)
    
    dist_df = dist_df[(dist_df['route'] == route) & (dist_df['bound'] == target_bound)]
    if dist_df.empty:
        raise ValueError(f"No stops found for route {route} and bound {bound} in {route_stop_dist_csv}")

    real_links = load_real_link_speeds(real_links_csv)
    real_links = real_links[(real_links['route'] == route) & (real_links['bound'] == target_bound)]
    
    # 2. 计算真实值 (累积行程时间)
    real_link_stats = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean().reset_index()
    real_link_stats = real_link_stats.sort_values('from_seq')
    real_cum_time = {1: 0}
    for _, row in real_link_stats.iterrows():
        f, t = row['from_seq'], row['to_seq']
        if f in real_cum_time:
            real_cum_time[t] = real_cum_time[f] + row['travel_time_s']
    
    real_time_df = pd.DataFrame(list(real_cum_time.items()), columns=['seq', 'real_time_s'])

    # 3. 处理仿真数据 (累积行程时间)
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    if sim_traj.empty:
        return 1e6
        
    sim_trips = []
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        min_seq = group['seq'].min()
        start_time = group.loc[group['seq'] == min_seq, 'arrival_time'].values[0]
        group['rel_time_s'] = group['arrival_time'] - start_time
        sim_trips.append(group[['seq', 'rel_time_s']])
    
    if not sim_trips:
        return 1e6

    sim_all = pd.concat(sim_trips)
    sim_stats = sim_all.groupby('seq')['rel_time_s'].mean().reset_index().rename(columns={'rel_time_s': 'sim_time_s'})
    
    # 4. 合并并计算 RMSE
    comparison = pd.merge(real_time_df, sim_stats, on='seq', how='inner')
    
    # 我们只关注序列大于 1 的点（seq=1 时 error 恒为 0）
    valid_comp = comparison[comparison['seq'] > 1]
    if valid_comp.empty:
        return 1e6
        
    rmse = np.sqrt(((valid_comp['sim_time_s'] - valid_comp['real_time_s']) ** 2).mean())
    return rmse


# =============================================================================
# 鲁棒性目标函数升级 (Week 3)
# =============================================================================

from scipy.stats import ks_2samp, wasserstein_distance as scipy_wasserstein
from typing import Dict, Tuple, Optional


def calculate_ks_distance(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    """
    计算两组样本的 K-S (Kolmogorov-Smirnov) 统计量
    
    K-S 统计量衡量两个经验分布函数的最大垂直距离，
    值越小表示分布越相似。
    
    Args:
        sim_values: 仿真值数组
        real_values: 真实值数组
        
    Returns:
        ks_stat: K-S 统计量 [0, 1]，越小越好
    """
    if len(sim_values) < 2 or len(real_values) < 2:
        return 1.0  # 样本不足时返回最大值
    
    stat, _ = ks_2samp(sim_values, real_values)
    return stat


def calculate_wasserstein_distance(sim_values: np.ndarray, real_values: np.ndarray) -> float:
    """
    计算 Wasserstein-1 距离 (Earth Mover's Distance)
    
    Wasserstein 距离衡量将一个分布"搬运"到另一个分布所需的最小代价，
    对分布形状和位置都敏感。
    
    Args:
        sim_values: 仿真值数组
        real_values: 真实值数组
        
    Returns:
        wasserstein: Wasserstein-1 距离，越小越好
    """
    if len(sim_values) < 1 or len(real_values) < 1:
        return 1e6
    
    return scipy_wasserstein(sim_values, real_values)


def robust_loss(errors: np.ndarray, lambda_std: float = 0.5) -> float:
    """
    鲁棒综合损失：J = mean(E) + λ * std(E)
    
    通过惩罚误差标准差，鼓励参数在不同时段/场景下表现稳定，
    而非仅优化平均值。
    
    Args:
        errors: 误差数组（绝对误差）
        lambda_std: 标准差惩罚系数，典型值 0.3-0.7
        
    Returns:
        loss: 鲁棒损失值
    """
    if len(errors) == 0:
        return 1e6
    
    mean_e = np.mean(np.abs(errors))
    std_e = np.std(np.abs(errors))
    return mean_e + lambda_std * std_e


def quantile_loss(errors: np.ndarray, quantile: float = 0.9) -> float:
    """
    分位数损失（如 P90）
    
    关注误差分布的尾部（最差情况），
    确保参数在极端情况下也有可接受的表现。
    
    Args:
        errors: 误差数组（绝对误差）
        quantile: 分位数，如 0.9 表示 P90
        
    Returns:
        loss: 分位数损失值
    """
    if len(errors) == 0:
        return 1e6
    
    return np.quantile(np.abs(errors), quantile)


def calculate_l1_robust_objective(
    sim_xml_path: str,
    real_links_csv: str,
    route_stop_dist_csv: str,
    route: str = '68X',
    bound: str = 'I',
    use_ks: bool = True,
    use_robust: bool = True,
    lambda_std: float = 0.5,
    ks_weight: float = 50.0,
    quantile: float = 0.9
) -> Dict[str, float]:
    """
    综合鲁棒性 L1 目标函数
    
    整合 RMSE、K-S 分布距离、鲁棒损失等多个指标，
    用于多目标优化或加权单目标优化。
    
    Args:
        sim_xml_path: SUMO 仿真输出 XML 路径
        real_links_csv: 真实链路速度 CSV 路径
        route_stop_dist_csv: 路线站点距离 CSV 路径
        route: 线路名称
        bound: 方向 ('I' or 'O')
        use_ks: 是否计算 K-S 统计量
        use_robust: 是否使用 mean+λ*std 鲁棒损失
        lambda_std: 鲁棒损失中的 λ 参数
        ks_weight: K-S 项的权重系数
        quantile: 分位数损失的分位点
        
    Returns:
        dict: 包含各项指标的字典
            - rmse: 传统 RMSE
            - ks_stat: K-S 统计量（站间时间分布）
            - wasserstein: Wasserstein 距离
            - robust_loss: mean + λ*std
            - quantile_loss: P90 分位损失
            - combined: 加权综合目标
    """
    # 1. 加载数据
    sim_raw = load_sim_data(sim_xml_path)
    if sim_raw.empty:
        return _empty_metrics()
    
    dist_df = load_route_stop_dist(route_stop_dist_csv)
    bound_map = {'I': 'inbound', 'O': 'outbound'}
    target_bound = bound_map.get(bound, bound)
    
    dist_df = dist_df[(dist_df['route'] == route) & (dist_df['bound'] == target_bound)]
    if dist_df.empty:
        return _empty_metrics()
    
    real_links = load_real_link_speeds(real_links_csv)
    real_links = real_links[(real_links['route'] == route) & (real_links['bound'] == target_bound)]
    
    # 2. 计算真实链路时间分布
    real_link_times = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].apply(list).to_dict()
    real_link_means = real_links.groupby(['from_seq', 'to_seq'])['travel_time_s'].mean()
    
    # 3. 处理仿真数据
    sim_traj = build_sim_trajectory(sim_raw, dist_df)
    if sim_traj.empty:
        return _empty_metrics()
    
    # 计算仿真链路时间
    sim_link_times = {}
    for vid, group in sim_traj.groupby('vehicle_id'):
        group = group.sort_values('seq')
        for i in range(len(group) - 1):
            row1 = group.iloc[i]
            row2 = group.iloc[i + 1]
            key = (int(row1['seq']), int(row2['seq']))
            travel_time = row2['arrival_time'] - row1['departure_time']
            if travel_time > 0:
                sim_link_times.setdefault(key, []).append(travel_time)
    
    # 4. 计算各项指标
    all_real_times = []
    all_sim_times = []
    errors = []
    
    for key, sim_times in sim_link_times.items():
        if key in real_link_times:
            real_times = real_link_times[key]
            all_real_times.extend(real_times)
            all_sim_times.extend(sim_times)
            
            # 误差：仿真均值 vs 真实均值
            sim_mean = np.mean(sim_times)
            real_mean = np.mean(real_times)
            errors.append(sim_mean - real_mean)
    
    if not errors:
        return _empty_metrics()
    
    errors = np.array(errors)
    all_real_times = np.array(all_real_times)
    all_sim_times = np.array(all_sim_times)
    
    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # K-S 统计量
    ks_stat = calculate_ks_distance(all_sim_times, all_real_times) if use_ks else 0.0
    
    # Wasserstein 距离
    wasserstein = calculate_wasserstein_distance(all_sim_times, all_real_times)
    
    # 鲁棒损失
    robust = robust_loss(errors, lambda_std) if use_robust else rmse
    
    # 分位数损失
    q_loss = quantile_loss(errors, quantile)
    
    # 综合目标（加权组合）
    combined = robust + ks_weight * ks_stat if use_ks else robust
    
    return {
        'rmse': rmse,
        'ks_stat': ks_stat,
        'wasserstein': wasserstein,
        'robust_loss': robust,
        'quantile_loss': q_loss,
        'combined': combined
    }


def _empty_metrics() -> Dict[str, float]:
    """返回空/错误情况下的默认指标"""
    return {
        'rmse': 1e6,
        'ks_stat': 1.0,
        'wasserstein': 1e6,
        'robust_loss': 1e6,
        'quantile_loss': 1e6,
        'combined': 1e6
    }


if __name__ == "__main__":
    # 简单的冒烟测试逻辑
    print("=== Objective Functions Smoke Test ===")
    
    # 测试基础损失函数
    test_errors = np.array([10, 15, 8, 12, 20, 5])
    print(f"测试误差: {test_errors}")
    print(f"  RMSE: {np.sqrt(np.mean(test_errors**2)):.2f}")
    print(f"  Robust (λ=0.5): {robust_loss(test_errors, 0.5):.2f}")
    print(f"  P90: {quantile_loss(test_errors, 0.9):.2f}")
    
    # 测试分布距离
    sim_vals = np.random.normal(100, 15, 50)
    real_vals = np.random.normal(105, 12, 50)
    print(f"\n分布距离测试:")
    print(f"  K-S: {calculate_ks_distance(sim_vals, real_vals):.4f}")
    print(f"  Wasserstein: {calculate_wasserstein_distance(sim_vals, real_vals):.2f}")
    
    print("\n✓ 目标函数测试通过")

