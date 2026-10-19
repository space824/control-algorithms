# MATLAB 实现

把 `control-algorithms` 里的算法（位置式 PID、增量式 PID、一阶低通、二阶低通、互补滤波）用 **MATLAB** 重新实现一遍，并逐段解释「代码为什么这么写」。

> 这套代码对应仓库里 `examples/` 下的 Python 脚本，生成的图形应和 `docs/images/` 里的 PNG 一致。用 MATLAB 重新写一遍的目的：一是熟悉 MATLAB 这个控制工程的标准工具，二是把「算法」和「画图」分清楚——算法核心是那个递推，画图只是最后那几句 `plot`。

---

## 目录结构

```
matlab实现/
├── README.md                  ← 你在这里
├── 代码/
│   ├── day1_position_pid.m          ← 位置式 PID 阶跃响应
│   ├── day2_incremental_pid.m       ← 增量式 vs 位置式（积分饱和；微分冲击辨析）
│   ├── day3_lowpass_filter.m        ← 一阶低通滤波（含幅频响应）
│   ├── day4_second_order_lowpass.m  ← 二阶巴特沃斯 biquad（含与一阶的对比）
│   └── day5_complementary_filter.m  ← 互补滤波（陀螺 + 加速度计融合）
└── 解析/
    ├── Day1_位置式PID解析.md
    ├── Day2_增量式PID解析.md
    ├── Day3_低通滤波解析.md
    ├── Day4_二阶低通滤波解析.md
    └── Day5_互补滤波解析.md
```

---

## 怎么运行

1. 打开 MATLAB（需要 R2016b 或更新版本，因为代码用到了脚本末尾的 local function）。
2. 把「代码」文件夹设为当前目录，或在 MATLAB 里双击某个 `.m` 文件。
3. 直接按 **F5**（或点「运行」按钮），会自动弹窗显示图形。

五个脚本都是独立的，跑哪个都行：

| 脚本 | 结果对应 |
|---|---|
| `day1_position_pid.m` | `docs/images/day1_position_pid.png` |
| `day2_incremental_pid.m` | `docs/images/day2_incremental_pid.png` |
| `day3_lowpass_filter.m` | `docs/images/day3_lowpass_filter.png` |
| `day4_second_order_lowpass.m` | `docs/images/day4_second_order_lowpass.png` |
| `day5_complementary_filter.m` | `docs/images/day5_complementary_filter.png` |

> 已在 R2026a 上逐个跑过，`day4` 与 Python 版的结构性结论完全一致（b/a 系数、4.32% 超调、3.98/6.29 ms 延迟、−10.7/−21.9 dB 干扰衰减）。RMSE 会有小数点后的差异，因为 MATLAB 的 `rng(0)` 与 numpy 的 `default_rng(0)` 生成的是不同随机序列——对齐的是结构和结论，不是某一次抽样。

---

## 对照速查：matplotlib → MATLAB

| Python (matplotlib) | MATLAB | 说明 |
|---|---|---|
| `plt.subplots(2,1)` | `subplot(2,1,1)` / `subplot(2,1,2)` | 分块画布 |
| `ax.plot(t, y, label='x')` | `plot(t, y, 'DisplayName','x')` | 画曲线并命名 |
| `ax.set_ylabel('y')` | `ylabel('y')` | 坐标轴标签 |
| `ax.set_title('t')` | `title('t')` | 标题 |
| `ax.grid(True)` | `grid on` | 网格 |
| `ax.legend()` | `legend('show')` | 图例 |
| `plt.axhline(-3, ls='--')` | `yline(-3, '--')` | 水平线（R2018b+） |
| `plt.axvline(fc)` | `xline(fc)` | 垂直线（R2018b+） |
| `ax.set_yscale('log')` | `semilogy(...)` | 对数 y 轴 |
| `np.arange(n)*dt` | `t = 0:dt:duration` | 时间轴 |
| `ax.semilogx(...)` | `semilogx(...)` | 对数 x 轴（同名） |
| `np.cumsum(x)*dt` | `cumsum(x)*dt` | 累积求和（积分） |
| `fig.savefig(out, dpi=150)` | `exportgraphics(gcf,'out.png','Resolution',150)` | 导出图片 |
| `np.random.default_rng(0)` | `rng(0)` | 固定随机种子（序列不同） |
| `matplotlib.use('Agg')` | **不需要** | MATLAB 默认直接弹窗显示 |

> 最重要的一个差异：MATLAB 里**同一个 `subplot` 要叠画多条曲线，必须先 `hold on`**，否则第二条 `plot` 会把第一条擦掉。Python 的 `ax.plot` 天生能叠画，没有这一步。

---

## 更进一步：MATLAB 的原生工具箱

上面 `.m` 文件是**逐行翻译 Python 的手写循环**，目的是看清算法每一步。但 MATLAB 真正的优势是 Control System Toolbox 和 Signal Processing Toolbox，同样的事能压到几行：

```matlab
G = tf(0.3, [1 1]);          % 一阶惯性环节 0.3/(s+1)
C = pid(4, 2, 0.3);          % PID 控制器
T = feedback(C*G, 1);        % 单位负反馈闭环
step(T); grid on;            % 一步画阶跃响应
```

```matlab
[b, a] = butter(2, fc/(Fs/2));   % 二阶巴特沃斯，两行拿到 biquad 系数
[H, f] = freqz(b, a, 1024, Fs);  % 频率响应
y = filter(b, a, x);             % 直接跑差分方程
```

`tf`、`pid`、`feedback`、`step`、`lsim`、`bode`、`butter`、`freqz` 都是现成的。以后学状态空间（`ss`）、LQR（`lqr`）、观测器（`place`）时，工具箱是行业标准，会比纯 Python 舒服得多。

> Day 4 的 `.m` 里仍手算了 biquad 系数（没用 `butter`），理由是：和 Python 版一一对应、能互相验算，而且手推一遍才知道 Q 是从 `alpha = sin(w0)/(2Q)` 这一步进去的。等熟了再用 `butter` 提速。

---

## 和 Simulink 的关系

上面这些递推做的事，在 Simulink 里就是**拖方块连线**：

```
[Step] ──> [⊕] ──> [PID Controller] ──> [Transfer Function] ──> [Scope]
            ↑                                                    │
            └────────────────── 反馈线 ────────────────────────────┘
```

- `setpoint` → `Step` 块
- `PositionPID` 类 → `PID Controller` 块（Kp/Ki/Kd 直接填对话框）
- `FirstOrderPlant` → `Transfer Function` 块（填 `0.3/(s+1)`）
- `plt.plot` 曲线 → `Scope` 示波器

手写代码和 Simulink 是**同一件事的两种表达**：代码适合理解原理，Simulink 适合搭复杂系统、和实物联动（配合 Embedded Coder 还能自动生成 STM32 的 C 代码）。
