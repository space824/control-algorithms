"""二阶低通滤波器（Biquad）：双线性变换 + 预畸变，Q = 1/√2 时即巴特沃斯。

连续域的标准二阶低通：

    H(s) = ωn² / (s² + (ωn/Q)·s + ωn²)

- ωn：自然频率；Q：品质因数。
- Q = 1/√2 ≈ 0.7071 时是 **巴特沃斯（Butterworth）**：通带最平坦、无谐振峰，
  并且幅度正好在 ω = ωn 处降到 -3dB——"截止频率"在二阶里最自然的含义。
- Q > 0.7071：通带鼓起一个谐振峰，截止附近会"翘"；
- Q < 0.7071：更保守，过渡带更缓，但更不容易振。

用双线性变换（Tustin）+ 预畸变化成 biquad 差分方程：

    y[k] = b0·x[k] + b1·x[k-1] + b2·x[k-2]
                   − a1·y[k-1] − a2·y[k-2]

系数（已归一化 a0 = 1，w0 = 2π·fc·dt）：

    alpha = sin(w0) / (2Q)
    b0 = (1 − cos w0) / 2 / (1 + alpha)
    b1 = (1 − cos w0)     / (1 + alpha)
    b2 = b0
    a1 = −2·cos(w0)       / (1 + alpha)
    a2 = (1 − alpha)      / (1 + alpha)

这套闭式就是 RBJ Audio EQ Cookbook 的写法，等价于"双线性变换 + 预畸变"：
采样率远高于 fc 时与连续域巴特沃斯几乎重合，接近奈奎斯特频率也不会跑偏。

与一阶低通（`src/filter/lowpass.py`）的区别——这就是 Day 4 要对比的：

| | 一阶 | 二阶 |
|---|---|---|
| 高频衰减斜率 | -20 dB/十倍频 | **-40 dB/十倍频** |
| 状态个数 | 1（y[k-1]） | 4（x[k-1], x[k-2], y[k-1], y[k-2]） |
| 同 fc 下相位滞后 | 小（3.98 ms @3Hz） | **约 1.6 倍**（6.29 ms @3Hz） |
| 同 fc 下**白噪声** | σ×0.334 | σ×0.281（**只少 1.5 dB**） |
| 对 fc 以上的**窄带干扰** | −20 dB/十倍频 | **−40 dB/十倍频**，离 fc 越远优势越大 |

一句话：二阶不是"更高级的一阶"，是拿**更多的相位滞后**去换**更陡的滚降**。

⚠️ 两个容易想错的地方（Day 4 实测数据）：

1. **白噪声下，同样的延迟就换来同样的噪声**——滤波器的等效噪声带宽只与低频群延迟成反比，
   与阶数无关。所以"升级到二阶"在白噪声上几乎打平（本课参数下只差 1.5 dB）。
2. 二阶真正的价值在**集中在高频的窄带干扰**（PWM 纹波、结构谐振、电源工频）：
   衰减斜率翻倍，离 fc 越远差距越大。用错场景会觉得"二阶没用"，那是对手的问题，不是滤波器的问题。
"""

import math

# 巴特沃斯的品质因数 Q = 1/√2 ≈ 0.7071
SQRT1_2 = 1.0 / math.sqrt(2.0)


class SecondOrderLowPass:
    """二阶低通（biquad，直接 I 型实现）。

    Parameters
    ----------
    b0, b1, b2, a1, a2 : float
        已归一化（a0 = 1）的差分方程系数。一般不要手填，
        用 `from_cutoff()` 由截止频率生成。

    直接 I 型的递推式：

        y[k] = b0·x[k] + b1·x[k-1] + b2·x[k-2] − a1·y[k-1] − a2·y[k-2]
    """

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0 = float(b0)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.reset()

    # ---------- 构造：由截止频率 → 系数 ----------
    @classmethod
    def from_cutoff(cls, cutoff_hz, dt, q=SQRT1_2):
        """由 -3dB 截止频率和采样周期生成二阶低通系数（默认巴特沃斯）。

        参数
        ----
        cutoff_hz : float
            -3dB 截止频率（Hz），必须落在 (0, fs/2) 内。
        dt : float
            采样周期（秒）。
        q : float
            品质因数，默认 1/√2（巴特沃斯）。
        """
        if dt <= 0:
            raise ValueError("dt 必须为正的采样周期（秒）")
        fs = 1.0 / dt
        if not (0.0 < cutoff_hz < fs / 2.0):
            raise ValueError(
                f"cutoff_hz={cutoff_hz} Hz 必须落在 (0, fs/2) = (0, {fs / 2:.1f}) Hz 内；"
                f"超过奈奎斯特频率时双线性变换会折叠，响应不再可信"
            )
        if q <= 0:
            raise ValueError("q 必须为正")

        w0 = 2.0 * math.pi * cutoff_hz * dt
        c = math.cos(w0)
        s = math.sin(w0)

        alpha = s / (2.0 * q)
        a0 = 1.0 + alpha

        b0 = (1.0 - c) / 2.0 / a0
        b1 = (1.0 - c) / a0
        b2 = b0
        a1 = -2.0 * c / a0
        a2 = (1.0 - alpha) / a0

        return cls(b0, b1, b2, a1, a2)

    # ---------- 运行 ----------
    def reset(self, y0=0.0):
        """把状态复位到稳态 y0（输入也当作一直是 y0）。"""
        y0 = float(y0)
        self._x1 = y0      # x[k-1]
        self._x2 = y0      # x[k-2]
        self._y1 = y0      # y[k-1]
        self._y2 = y0      # y[k-2]

    def update(self, x):
        """输入一个新样本 x[k]，返回滤波输出 y[k]。"""
        x = float(x)
        y = (self.b0 * x
             + self.b1 * self._x1
             + self.b2 * self._x2
             - self.a1 * self._y1
             - self.a2 * self._y2)

        # 移位：为下一拍准备历史
        self._x2 = self._x1
        self._x1 = x
        self._y2 = self._y1
        self._y1 = y
        return y

    # ---------- 体检用的属性 ----------
    @property
    def coefficients(self):
        """(b0, b1, b2, a1, a2)，方便外部画频率响应。"""
        return (self.b0, self.b1, self.b2, self.a1, self.a2)

    @property
    def dc_gain(self):
        """直流增益，(b0+b1+b2)/(1+a1+a2)。低通滤波器应当 ≈ 1。"""
        return (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)

    def __repr__(self):
        return (f"SecondOrderLowPass(b0={self.b0:.6g}, b1={self.b1:.6g}, "
                f"b2={self.b2:.6g}, a1={self.a1:.6g}, a2={self.a2:.6g})")
