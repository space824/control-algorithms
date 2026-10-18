"""增量式（速度形式）PID 控制器。

位置式 PID 直接输出绝对控制量：
    u[k] = Kp * e[k] + Ki * I[k] + Kd * D[k]
    其中 I[k] = I[k-1] + e[k] * dt，D[k] = (e[k] - e[k-1]) / dt

增量式 PID 输出的是"控制量的增量" Δu[k]，再叠加到上一次输出上：
    u[k] = u[k-1] + Δu[k]

对位置式公式做一阶差分，消去积分累加项，得到 Δu 的递推式。
注意：本文件里的 kp / ki / kd 与 PositionPID **同名同义**（都是连续域增益），
所以差分之后必须把 dt 的因子上补齐：

    Δu[k] = Kp * (e[k] - e[k-1])
          + Ki * e[k] * dt
          + Kd * (e[k] - 2*e[k-1] + e[k-2]) / dt

很多教材写成下面这样，把两个 dt 因子省掉了：

    Δu[k] = Kp * Δe + Ki * e[k] + Kd * Δ²e

那是因为教材里的 Ki、Kd 已经不是连续域增益，而是提前吸收了采样周期的系数
（Ki = Kp*T/Ti，Kd = Kp*Td/T）。这两种写法的数字**不能混用**：若拿位置式那组
kp/ki/kd 直接套进省掉 dt 的式子，积分增益会被放大 1/dt 倍（本例 dt=0.01，
即放大 100 倍），闭环极点会跑出单位圆，仿真直接发散。
本文件采用补齐 dt 的写法，保证与 PositionPID 共用同一组增益参数、结果可对照。

与位置式真正的区别只有一条：
- 天然抗积分饱和：增量式没有单独的积分累加器，唯一的"记忆"是上一次输出 u[k-1]，
  而 u 会被输出限幅夹住，所以不会像位置式那样在饱和期间无限累积积分。
顺带的好处：输出是增量，做无扰动切换（手动/自动切换）时不容易跳变。

代价：输出是相对的，若执行器需要绝对量则需额外累加；且要保留前两个误差
值 e[k-1]、e[k-2]。

一个常见误解：增量式并**不能**消除"微分冲击（derivative kick）"。
增益配平之后，增量式与位置式在代数上完全等价，设定值阶跃时两者的 u 会在同一拍
跳出同一个尖峰（Day 2 的例子都是 33.02）。微分冲击的根源是"对误差求导"，
真正的解法是**微分先行**——让 D 项作用在测量值上而不是误差上
（见 控制项目实现/P1-直流电机速度闭环 里的实现）。
"""


class IncrementalPID:
    """增量式 PID。

    Parameters
    ----------
    kp, ki, kd : float
        比例、积分、微分增益（与位置式 PID 同名同义，都是连续域增益）。
    dt : float
        控制周期（秒）。ki 项要乘它、kd 项要除它，见模块开头的说明。
    output_limits : (lo, hi) or None
        输出限幅，None 表示该方向不限幅。
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01,
                 output_limits=(None, None)):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.lo, self.hi = output_limits
        self.reset()

    def reset(self):
        self._e_prev = 0.0    # e[k-1]
        self._e_prev2 = 0.0   # e[k-2]
        self._last_output = 0.0

    def update(self, setpoint, measurement):
        """根据设定值和测量值计算控制量增量并叠加。

        Returns
        -------
        float
            本次控制量 u。
        """
        e = setpoint - measurement

        # 增量式 PID 的核心递推式（dt 因子必须补齐，否则与位置式不等价）
        du = (self.kp * (e - self._e_prev)
              + self.ki * e * self.dt
              + self.kd * (e - 2.0 * self._e_prev + self._e_prev2) / self.dt)

        u = self._last_output + du

        # 输出限幅（夹住 u 也就夹住了"记忆"，所以天然抗积分饱和）
        if self.hi is not None and u > self.hi:
            u = self.hi
        elif self.lo is not None and u < self.lo:
            u = self.lo

        # 移位更新
        self._e_prev2 = self._e_prev
        self._e_prev = e
        self._last_output = u
        return u
