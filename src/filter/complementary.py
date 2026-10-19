"""互补滤波（Complementary Filter）：陀螺仪 + 加速度计的经典融合。

两个传感器各有一个致命缺陷，而且**互补**：

| | 短时间（高频） | 长时间（低频） |
|---|---|---|
| 陀螺仪 ω → 积分得角度 | **准** | **漂移**（零偏积分成线性漂移） |
| 加速度计 → 重力方向算角度 | **噪声大** | **不漂**（重力是绝对参考） |

所以让陀螺管高频、加速度计管低频，各取所长：

    angle[k] = α·(angle[k-1] + (gyro[k] - bias)·dt) + (1-α)·angle_acc[k]

拆成两步看更清楚：

    ① 预测（陀螺，短时间可信）：pred = angle[k-1] + (gyro - bias)·dt
    ② 修正（加速度计，长期可信）：angle[k] = α·pred + (1-α)·angle_acc[k]

**这就是一个增益固定的"观测器"**：用预测值往前推，再用测量值按比例 K = (1-α) 拉回来。
卡尔曼滤波做的事完全一样，只是把固定的 (1-α) 换成一个随协方差自动变化的 K。
所以互补滤波是理解 KF 的最好跳板——它俩是同一个结构，只差一个"增益怎么定"。

⚠️ 注意 α 的含义和 Day 3 的一阶低通**相反**：

- Day 3：`y[k] = α·x[k] + (1-α)·y[k-1]`，α 是**新样本**（输入）的权重；
- Day 5：α 是**陀螺（历史预测）**的权重，`1-α` 才是**加速度计（新测量）**的权重。

因为这里的"新样本"是加速度计角，所以加速度计这一路走的是一个系数为 `1-α` 的一阶低通，
其时间常数与截止频率可直接套 Day 3 的公式（令 α' = 1-α）：

    τ  = dt·(1-α')/α' = dt·α/(1-α)
    cos(2π·fc·dt) = (2 - 2α' - α'²) / (2(1-α'))         ← 与 Day 3 完全同一个式子

举例：dt = 10 ms、α = 0.98 → α' = 0.02 → τ = 0.49 s → fc ≈ 0.32 Hz
（这个数很关键：α 只有在给定采样率时才有物理意义。200 Hz 下同样的 0.98，τ 只剩 0.245 s。）
"""

import math


class ComplementaryFilter:
    """互补滤波（陀螺 + 加速度计融合角度）。

    Parameters
    ----------
    alpha : float
        陀螺（预测）的权重，取 (0, 1)。常见 0.98。
        α 越大 → 越信陀螺 → 平滑，但抗漂移弱；α 越小 → 越信加速度计 → 跟手，但噪声大。
    dt : float
        采样周期（秒）。α 必须和 dt 一起看才有意义。
    angle0 : float
        初始角度（度）。
    bias : float
        陀螺零偏（度/秒）。这里只能**手工标定后填入**；
        互补滤波本身不会估计零偏（那是 Kalman 的活）。
    """

    def __init__(self, alpha=0.98, dt=0.01, angle0=0.0, bias=0.0):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha 必须落在 (0, 1) 内")
        if dt <= 0:
            raise ValueError("dt 必须为正的采样周期（秒）")
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.bias = float(bias)
        self.angle0 = float(angle0)
        self.reset()

    # ---------- 运行 ----------
    def reset(self, angle0=None):
        self._angle = self.angle0 if angle0 is None else float(angle0)

    def update(self, gyro_rate, accel_angle):
        """输入一拍陀螺角速度（度/秒）和加速度计角度（度），返回融合后的角度。

        ① 预测：用陀螺把上一拍的角度往前推一步
        ② 修正：用加速度计按 (1-alpha) 的权重拉回来
        """
        pred = self._angle + (float(gyro_rate) - self.bias) * self.dt
        self._angle = self.alpha * pred + (1.0 - self.alpha) * float(accel_angle)
        return self._angle

    # ---------- 体检用的属性 ----------
    @property
    def accel_weight(self):
        """加速度计那一路的一阶低通系数 α' = 1-α（Day 3 的写法）。"""
        return 1.0 - self.alpha

    @property
    def tau(self):
        """等效时间常数（秒），τ = dt·α/(1-α)。"""
        return self.dt * self.alpha / self.accel_weight

    @property
    def cutoff_hz(self):
        """-3dB 截止频率（Hz），离散域精确解，与 Day 3 的 from_cutoff 同一个式子。

        交叉点的物理含义：低于 fc 的成分主要信加速度计，高于 fc 主要信陀螺。
        """
        a = self.accel_weight
        if a <= 0.0:
            return 0.0
        c = (2.0 - 2.0 * a - a * a) / (2.0 * (1.0 - a))
        c = max(-1.0, min(1.0, c))
        return math.acos(c) / (2.0 * math.pi * self.dt)

    @property
    def angle(self):
        return self._angle

    def __repr__(self):
        return (f"ComplementaryFilter(alpha={self.alpha}, dt={self.dt}, "
                f"tau={self.tau:.3f}s, fc={self.cutoff_hz:.2f}Hz)")
