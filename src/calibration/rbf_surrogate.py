"""
RBF 代理模型实现
================
用于贝叶斯优化的 Radial Basis Function 插值模型。
支持交叉验证方差估计，以便与 Kriging 模型进行反比方差加权融合。
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import LeaveOneOut
from typing import Tuple, Optional


class RBFSurrogate:
    """
    RBF (Radial Basis Function) 代理模型
    
    特点:
    - 基于 scipy.interpolate.RBFInterpolator 实现
    - 支持多种核函数: thin_plate_spline, multiquadric, gaussian 等
    - 通过 LOO-CV (Leave-One-Out Cross-Validation) 估计预测方差
    
    接口设计与 KrigingSurrogate 保持一致，方便切换和融合。
    """
    
    def __init__(self, kernel: str = 'gaussian', smoothing: float = 0.0,
                 epsilon: Optional[float] = None):
        """
        初始化 RBF 代理模型
        
        Args:
            kernel: RBF 核函数类型，可选:
                - 'thin_plate_spline': r^2 * log(r)，适合平滑插值
                - 'multiquadric': sqrt(1 + (r/epsilon)^2)，常用默认
                - 'gaussian': exp(-(r/epsilon)^2)，局部性强
                - 'cubic': r^3
                - 'linear': r
            smoothing: 平滑因子，0 表示精确插值，>0 允许一定误差以换取平滑性
            epsilon: 核函数的 epsilon 参数（仅 multiquadric/gaussian 需要）
        """
        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.is_fitted = False
        
        # 存储训练数据用于方差估计
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._model: Optional[RBFInterpolator] = None
        
        # LOO-CV 误差用于方差估计
        self._loo_mse: float = 1.0  # 默认方差
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 RBF 代理模型
        
        Args:
            X: 训练样本特征 (n_samples, n_features)
            y: 训练样本目标值 (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        
        self._X_train = X.copy()
        self._y_train = y.copy()
        
        # 对于需要 epsilon 的核函数（gaussian, multiquadric, inverse_multiquadric, inverse_quadratic）
        # 如果未指定，则自动根据数据计算合理的默认值
        epsilon = self.epsilon
        if epsilon is None and self.kernel in ['gaussian', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic']:
            # 使用训练点之间的平均距离作为 epsilon 的基准
            if len(X) > 1:
                dists = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                epsilon = np.mean(np.min(dists, axis=1))
            else:
                epsilon = 1.0
            self._auto_epsilon = epsilon  # 保存自动计算的值
        else:
            self._auto_epsilon = epsilon
        
        # 构建 RBF 插值器
        if self._auto_epsilon is not None:
            self._model = RBFInterpolator(
                X, y,
                kernel=self.kernel,
                smoothing=self.smoothing,
                epsilon=self._auto_epsilon
            )
        else:
            self._model = RBFInterpolator(
                X, y,
                kernel=self.kernel,
                smoothing=self.smoothing
            )
        
        # 计算 LOO-CV MSE 作为方差估计的基础
        self._compute_loo_variance()
        self.is_fitted = True
    
    def _compute_loo_variance(self) -> None:
        """
        通过 Leave-One-Out 交叉验证计算预测方差估计
        
        原理：LOO-CV 的均方误差可作为模型不确定性的代理
        """
        if self._X_train is None or len(self._X_train) < 3:
            self._loo_mse = 1.0
            return
        
        loo = LeaveOneOut()
        loo_errors = []
        
        for train_idx, test_idx in loo.split(self._X_train):
            X_tr, X_te = self._X_train[train_idx], self._X_train[test_idx]
            y_tr, y_te = self._y_train[train_idx], self._y_train[test_idx]
            
            try:
                if self._auto_epsilon is not None:
                    temp_model = RBFInterpolator(
                        X_tr, y_tr,
                        kernel=self.kernel,
                        smoothing=self.smoothing,
                        epsilon=self._auto_epsilon
                    )
                else:
                    temp_model = RBFInterpolator(
                        X_tr, y_tr,
                        kernel=self.kernel,
                        smoothing=self.smoothing
                    )
                pred = temp_model(X_te)
                loo_errors.append((pred[0] - y_te[0]) ** 2)
            except Exception:
                # 如果某次 LOO 失败，跳过
                continue
        
        if loo_errors:
            self._loo_mse = np.mean(loo_errors)
        else:
            self._loo_mse = 1.0
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测均值和标准差
        
        Args:
            X: 预测点 (n_samples, n_features)
            
        Returns:
            mu: 预测均值 (n_samples,)
            sigma: 预测标准差 (n_samples,)  -- 基于距离加权的 LOO 方差
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = np.atleast_2d(X)
        
        # 预测均值
        mu = self._model(X)
        
        # 预测方差估计：基于距离的自适应方差
        # 距离训练点越远，不确定性越大
        sigma = self._estimate_variance(X)
        
        return mu, sigma
    
    def _estimate_variance(self, X: np.ndarray) -> np.ndarray:
        """
        估计预测点的方差
        
        策略：结合 LOO-MSE 和到最近训练点的距离
        σ²(x) = LOO_MSE * (1 + min_dist(x) / mean_dist)
        
        物理意义：
        - 远离训练点时，不确定性增大
        - LOO_MSE 提供基础不确定性水平
        """
        if self._X_train is None or len(self._X_train) == 0:
            return np.ones(len(X)) * np.sqrt(self._loo_mse)
        
        # 计算每个预测点到所有训练点的距离
        # 使用广播计算欧氏距离
        dists = np.linalg.norm(X[:, np.newaxis, :] - self._X_train[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        
        # 计算训练点之间的平均距离作为参考尺度
        if len(self._X_train) > 1:
            train_dists = np.linalg.norm(
                self._X_train[:, np.newaxis, :] - self._X_train[np.newaxis, :, :], 
                axis=2
            )
            np.fill_diagonal(train_dists, np.inf)
            mean_train_dist = np.mean(np.min(train_dists, axis=1))
        else:
            mean_train_dist = 1.0
        
        # 距离越远，方差越大
        variance_scale = 1.0 + min_dists / (mean_train_dist + 1e-8)
        variance = self._loo_mse * variance_scale
        
        # 返回标准差
        return np.sqrt(np.maximum(variance, 1e-8))
    
    def get_variance(self, X: np.ndarray) -> np.ndarray:
        """
        获取预测方差（用于反比方差加权融合）
        
        Args:
            X: 预测点 (n_samples, n_features)
            
        Returns:
            variance: 预测方差 (n_samples,)
        """
        _, sigma = self.predict(X)
        return sigma ** 2


if __name__ == "__main__":
    # 冒烟测试
    print("=== RBF Surrogate Smoke Test ===")
    
    # 生成测试数据：2D Rosenbrock-like 函数
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (20, 2))
    y_train = (1 - X_train[:, 0])**2 + 100 * (X_train[:, 1] - X_train[:, 0]**2)**2
    
    # 训练模型
    model = RBFSurrogate(kernel='thin_plate_spline', smoothing=0.1)
    model.fit(X_train, y_train)
    
    # 预测
    X_test = np.array([[0.0, 0.0], [1.0, 1.0], [-1.5, 2.0]])
    mu, sigma = model.predict(X_test)
    
    print(f"训练样本数: {len(X_train)}")
    print(f"LOO-MSE: {model._loo_mse:.4f}")
    print(f"\n预测结果:")
    for i, (x, m, s) in enumerate(zip(X_test, mu, sigma)):
        print(f"  点 {x}: μ={m:.2f}, σ={s:.2f}")
    
    print("\n✓ RBF Surrogate 测试通过")
