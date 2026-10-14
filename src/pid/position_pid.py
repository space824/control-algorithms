"""位置式（并联形式）PID 控制器。

连续形式：
    u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt

离散化（积分用累加和，微分用后向差分）：
    e[k]  = r[k] - y[k]
    I[k]  = I[k-1] + e[k] * dt
    D[k]  = (e[k] - e[k-1]) / dt
    u[k]  = Kp * e[k] + Ki * I[k] + Kd * D[k]

要点：
- 位置式 PID 直接输出"绝对控制量" u，适合需要绝对量的场合（如阀门开度、占空比）。
- 积分项会无限累积，输出一旦饱和就会造成"积分饱和（integral windup）"，
  这里用"条件积分"做简化抗饱和：只有当输出没有被限幅时才真正累积积分。
  更完善的方案（Day 2 的增量式、以及积分反算/钳位）会进一步讨论。
"""


class PositionPID:
    """位置式 PID。

    Parameters
    ----------
    kp, ki, kd : float
        比例、积分、微分增益。
    dt : float
        控制周期（秒），与采样保持一致。
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
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_output = 0.0

    def update(self, setpoint, measurement):
        """根据设定值和测量值计算控制量。

        Returns
        -------
        float
            本次控制量 u。
        """
        error = setpoint - measurement

        # 先算一个"积分候选值"，若输出被限幅则不真正累积（抗积分饱和）
        integral_candidate = self._integral + error * self.dt
        derivative = (error - self._prev_error) / self.dt

        u = (self.kp * error
             + self.ki * integral_candidate
             + self.kd * derivative)

        # 输出限幅
        if self.hi is not None and u > self.hi:
            u = self.hi
        elif self.lo is not None and u < self.lo:
            u = self.lo
        else:
            # 未饱和时才更新积分
            self._integral = integral_candidate

        self._prev_error = error
        self._last_output = u
        return u
