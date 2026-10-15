"""增量式（速度形式）PID 控制器。

位置式 PID 直接输出绝对控制量：
    u[k] = Kp * e[k] + Ki * I[k] + Kd * D[k]

增量式 PID 输出的是"控制量的增量" Δu[k]，再叠加到上一次输出上：
    u[k] = u[k-1] + Δu[k]

对位置式公式做一阶差分，消去积分累加项，得到 Δu 的递推式：
    Δu[k] = Kp * (e[k] - e[k-1])
          + Ki * e[k]
          + Kd * (e[k] - 2*e[k-1] + e[k-2])

与位置式的关键区别（也是本课要对比的两点）：
- 天然抗积分饱和：增量式没有单独的积分累加器，唯一的"记忆"是上一次输出
  u[k-1]，而 u 会被输出限幅夹住，所以不会像位置式那样无限累积积分。
- 对设定值突变更平滑：微分项作用在误差的二阶差分上、量级有限；且输出是
  逐步叠加的，不会像位置式那样在设定值阶跃时产生"微分冲击（derivative kick）"。

代价：输出是相对的，若执行器需要绝对量则需额外累加；且要保留前两个误差
值 e[k-1]、e[k-2]。
"""


class IncrementalPID:
    """增量式 PID。

    Parameters
    ----------
    kp, ki, kd : float
        比例、积分、微分增益（与位置式 PID 同名同义）。
    dt : float
        控制周期（秒）。增量式公式里 dt 会被约掉，这里保留仅为接口统一。
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

        # 增量式 PID 的核心递推式
        du = (self.kp * (e - self._e_prev)
              + self.ki * e
              + self.kd * (e - 2.0 * self._e_prev + self._e_prev2))

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
