"""
双代理融合模型实现
==================
基于反比方差加权 (Inverse-Variance Weighting) 的 Kriging + RBF 融合策略。

融合公式：
    y_fusion(U) = (σ²_RBF * y_Kriging + σ²_Kriging * y_RBF) / (σ²_Kriging + σ²_RBF)

原理：
- Kriging 方差大（不确定性高）时，融合结果更依赖 RBF
- Kriging 方差小（置信度高）时，融合结果更依赖 Kriging
- 实现全局趋势（Kriging）与局部拟合（RBF）的自适应平衡
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional

# 支持作为模块导入和直接运行两种方式
try:
    from .surrogate import KrigingSurrogate
    from .rbf_surrogate import RBFSurrogate
except ImportError:
    from surrogate import KrigingSurrogate
    from rbf_surrogate import RBFSurrogate


class DualSurrogate:
    """
    双代理融合模型
    
    结合 Kriging 的不确定性量化能力和 RBF 的局部拟合能力，
    通过反比方差加权实现自适应融合。
    
    适用场景：
    - 数据稀疏区域：Kriging 方差大，RBF 主导
    - 数据密集区域：Kriging 方差小，Kriging 主导
    """
    
    def __init__(self, 
                 kriging_random_state: int = 42,
                 rbf_kernel: str = 'gaussian',
                 rbf_smoothing: float = 0.1,
                 min_variance: float = 1e-8):
        """
        初始化双代理模型
        
        Args:
            kriging_random_state: Kriging 模型随机种子
            rbf_kernel: RBF 核函数类型
            rbf_smoothing: RBF 平滑因子
            min_variance: 方差下限，防止除零
        """
        self.kriging = KrigingSurrogate(random_state=kriging_random_state)
        self.rbf = RBFSurrogate(kernel=rbf_kernel, smoothing=rbf_smoothing)
        self.min_variance = min_variance
        self.is_fitted = False
        
        # 存储训练信息
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练双代理模型
        
        同时训练 Kriging 和 RBF，为后续融合做准备。
        
        Args:
            X: 训练样本特征 (n_samples, n_features)
            y: 训练样本目标值 (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        
        self._X_train = X.copy()
        self._y_train = y.copy()
        
        # 训练两个基模型
        self.kriging.fit(X, y)
        self.rbf.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        反比方差加权融合预测
        
        融合公式：
            y_fusion = (σ²_RBF * y_Kriging + σ²_Kriging * y_RBF) / (σ²_Kriging + σ²_RBF)
        
        融合方差（基于独立假设）：
            σ²_fusion = 1 / (1/σ²_Kriging + 1/σ²_RBF)
        
        Args:
            X: 预测点 (n_samples, n_features)
            
        Returns:
            mu_fusion: 融合预测均值 (n_samples,)
            sigma_fusion: 融合预测标准差 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = np.atleast_2d(X)
        
        # 获取两个模型的预测
        mu_kriging, sigma_kriging = self.kriging.predict(X)
        mu_rbf, sigma_rbf = self.rbf.predict(X)
        
        # 计算方差（确保非负且有下限）
        var_kriging = np.maximum(sigma_kriging ** 2, self.min_variance)
        var_rbf = np.maximum(sigma_rbf ** 2, self.min_variance)
        
        # 反比方差加权融合
        # y_fusion = (σ²_RBF * y_Kriging + σ²_Kriging * y_RBF) / (σ²_Kriging + σ²_RBF)
        total_var = var_kriging + var_rbf
        w_kriging = var_rbf / total_var  # Kriging 权重 = RBF 方差 / 总方差
        w_rbf = var_kriging / total_var  # RBF 权重 = Kriging 方差 / 总方差
        
        mu_fusion = w_kriging * mu_kriging + w_rbf * mu_rbf
        
        # 融合方差：1 / (1/σ²_K + 1/σ²_R) = σ²_K * σ²_R / (σ²_K + σ²_R)
        var_fusion = (var_kriging * var_rbf) / total_var
        sigma_fusion = np.sqrt(var_fusion)
        
        return mu_fusion, sigma_fusion
    
    def expected_improvement(self, X_candidates: np.ndarray, best_y: float, 
                            xi: float = 0.1) -> np.ndarray:
        """
        基于融合模型计算期望改进量 (EI)
        
        使用融合后的均值和标准差计算 EI，
        保留 Kriging 的不确定性引导能力。
        
        Args:
            X_candidates: 候选点集 (n_candidates, n_features)
            best_y: 当前已知最小目标值
            xi: 探索因子
            
        Returns:
            ei: 期望改进量 (n_candidates,)
        """
        mu_fusion, sigma_fusion = self.predict(X_candidates)
        
        # EI 计算（最小化目标）
        with np.errstate(divide='warn'):
            imp = best_y - mu_fusion - xi
            Z = imp / sigma_fusion
            ei = imp * norm.cdf(Z) + sigma_fusion * norm.pdf(Z)
            ei[sigma_fusion <= 0.0] = 0.0
        
        return ei
    
    def get_model_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取各模型在预测点的权重（用于可视化和诊断）
        
        Args:
            X: 预测点 (n_samples, n_features)
            
        Returns:
            w_kriging: Kriging 权重 (n_samples,)
            w_rbf: RBF 权重 (n_samples,)
        """
        X = np.atleast_2d(X)
        
        _, sigma_kriging = self.kriging.predict(X)
        _, sigma_rbf = self.rbf.predict(X)
        
        var_kriging = np.maximum(sigma_kriging ** 2, self.min_variance)
        var_rbf = np.maximum(sigma_rbf ** 2, self.min_variance)
        
        total_var = var_kriging + var_rbf
        w_kriging = var_rbf / total_var
        w_rbf = var_kriging / total_var
        
        return w_kriging, w_rbf
    
    def get_individual_predictions(self, X: np.ndarray) -> dict:
        """
        获取各模型的独立预测结果（用于诊断和对比）
        
        Args:
            X: 预测点 (n_samples, n_features)
            
        Returns:
            dict: 包含各模型预测的字典
        """
        X = np.atleast_2d(X)
        
        mu_kriging, sigma_kriging = self.kriging.predict(X)
        mu_rbf, sigma_rbf = self.rbf.predict(X)
        mu_fusion, sigma_fusion = self.predict(X)
        w_kriging, w_rbf = self.get_model_weights(X)
        
        return {
            'kriging': {'mu': mu_kriging, 'sigma': sigma_kriging},
            'rbf': {'mu': mu_rbf, 'sigma': sigma_rbf},
            'fusion': {'mu': mu_fusion, 'sigma': sigma_fusion},
            'weights': {'kriging': w_kriging, 'rbf': w_rbf}
        }


if __name__ == "__main__":
    # 冒烟测试
    print("=== Dual Surrogate (Inverse-Variance Weighting) Smoke Test ===\n")
    
    # 生成测试数据：2D 函数
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (15, 2))
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.1 * np.random.randn(15)
    
    # 训练双代理模型
    model = DualSurrogate()
    model.fit(X_train, y_train)
    
    # 预测点：包含数据密集和稀疏区域
    X_test = np.array([
        [0.0, 0.0],   # 数据密集区域
        [1.5, 1.5],   # 中等区域
        [-1.8, 1.8],  # 边缘区域
    ])
    
    print("预测结果对比：")
    print("-" * 70)
    results = model.get_individual_predictions(X_test)
    
    for i, x in enumerate(X_test):
        print(f"\n点 [{x[0]:.1f}, {x[1]:.1f}]:")
        print(f"  Kriging: μ={results['kriging']['mu'][i]:.3f}, σ={results['kriging']['sigma'][i]:.3f}")
        print(f"  RBF:     μ={results['rbf']['mu'][i]:.3f}, σ={results['rbf']['sigma'][i]:.3f}")
        print(f"  Fusion:  μ={results['fusion']['mu'][i]:.3f}, σ={results['fusion']['sigma'][i]:.3f}")
        print(f"  权重:    w_K={results['weights']['kriging'][i]:.2%}, w_R={results['weights']['rbf'][i]:.2%}")
    
    # 测试 EI 计算
    print("\n\nEI 计算测试：")
    best_y = np.min(y_train)
    ei = model.expected_improvement(X_test, best_y)
    for i, (x, e) in enumerate(zip(X_test, ei)):
        print(f"  点 {x}: EI={e:.4f}")
    
    print("\n✓ Dual Surrogate 测试通过")
