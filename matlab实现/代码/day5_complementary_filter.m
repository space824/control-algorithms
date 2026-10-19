% day5_complementary_filter.m
% 互补滤波：陀螺仪 + 加速度计融合角度。
%
%     angle[k] = α·(angle[k-1] + gyro[k]·dt) + (1-α)·angle_acc[k]
%
% 陀螺积分：短时间准、长时间漂移（零偏积分）；加速度计：长期不漂、短期噪声大。
% 互补滤波让两者各管一段频带：低频信加速度计，高频信陀螺。
%
% 本脚本做四件事（四张子图）：
%     (a) 时域：真值 / 纯陀螺积分 / 纯加速度计 / 互补
%     (b) 误差曲线：陀螺误差线性发散
%     (c) alpha 扫描：RMSE 随 alpha 怎么变，最优点在哪
%     (d) 「互补」的含义：加速度计路径低通 + 陀螺路径高通 = 1
%
% 运行：在 MATLAB 里直接按 F5，或在命令行输入 day5_complementary_filter

clear; clc; close all;

%% ---------- 1. 仿真基本参数 ----------
dt       = 0.01;       % 100 Hz，MPU6050 的典型采样率
duration = 30.0;       % 30 秒，够让陀螺漂移到肉眼可见
alpha    = 0.98;       % 课本经典值
t        = 0:dt:duration-dt;
n        = numel(t);

f_tilt     = 0.05;     % 真实倾斜频率 0.05 Hz
amp        = 20.0;     % 幅值 20°
gyro_bias  = 0.5;      % 陀螺零偏 0.5 °/s
gyro_noise = 0.1;      % 陀螺噪声 0.1 °/s
accel_noise = 1.5;     % 加速度计噪声 1.5°

%% ---------- 2. 造「真值 + 两个脏传感器」 ----------
angle_true = amp*sin(2*pi*f_tilt*t);                          % 真实角度
rate_true  = amp*2*pi*f_tilt*cos(2*pi*f_tilt*t);              % 解析求导，不用差分
rng(0);                                                       % 固定随机种子
gyro  = rate_true + gyro_bias + gyro_noise*randn(1, n);       % 零偏 + 白噪声
accel = angle_true + accel_noise*randn(1, n);                 % 同一个角度，噪声大得多

%% ---------- 3. 三种估计 ----------
% ① 纯陀螺积分（不补偿零偏）：cumsum 就是累加，一行代替循环
ang_gyro = [0, cumsum(gyro(2:end))*dt];

% ② 纯加速度计
ang_acc = accel;

% ③ 互补滤波
ang_cf = cf_run(gyro, accel, alpha, dt, 0);

rmse_ = @(x, y) sqrt(mean((x - y).^2));
r_gyro = rmse_(ang_gyro, angle_true);
r_acc  = rmse_(ang_acc,  angle_true);
r_cf   = rmse_(ang_cf,   angle_true);
drift  = ang_gyro(end) - angle_true(end);
[tau, fc] = cf_consts(alpha, dt);

fprintf('dt = %.0f ms (%.0f Hz) | 时长 %.0f s | alpha = %.2f\n', dt*1000, 1/dt, duration, alpha);
fprintf('互补滤波：tau = %.3f s，fc = %.2f Hz\n', tau, fc);
fprintf('%s\n', repmat('-', 1, 56));
fprintf('%-16s%14s%12s\n', '估计方法', 'RMSE(对真值)', '末态误差');
fprintf('%-16s%14.4f%11.2f°\n', '纯陀螺积分', r_gyro, drift);
fprintf('%-16s%14.4f%11.2f°\n', '纯加速度计', r_acc, 0);
fprintf('%-16s%14.4f%11.2f°\n', '互补滤波',   r_cf,  ang_cf(end) - angle_true(end));
fprintf('%s\n', repmat('-', 1, 56));

%% ---------- 4. alpha 扫描 ----------
alphas = linspace(0.50, 0.995, 60);
scan   = zeros(size(alphas));
for i = 1:numel(alphas)
    y = cf_run(gyro, accel, alphas(i), dt, 0);
    scan(i) = rmse_(y, angle_true);
end
[best_rmse, best_i] = min(scan);
j = find(abs(alphas - 0.98) < 0.005, 1);      % 离 0.98 最近的那个采样点
fprintf('alpha 扫描：最优 alpha = %.4f，RMSE = %.4f\n', alphas(best_i), best_rmse);
fprintf('  课本值 alpha=0.98 的 RMSE = %.4f（离最优差 %.1f%%）\n', ...
        scan(j), (scan(j)/best_rmse - 1)*100);

%% ---------- 5. 出图 ----------
figure('Position', [100 100 900 760]);

% (a) 时域对比
subplot(2,2,1); hold on; grid on;
plot(t, angle_true, 'k--', 'LineWidth', 2, 'DisplayName', 'true angle');
plot(t, ang_acc,  'Color', [0.7 0.7 0.7], 'LineWidth', 0.6, ...
     'DisplayName', sprintf('accel only (RMSE %.2f°)', r_acc));
plot(t, ang_gyro, 'r', 'LineWidth', 1.2, ...
     'DisplayName', sprintf('gyro integrate (RMSE %.2f°)', r_gyro));
plot(t, ang_cf,   'b', 'LineWidth', 2.0, ...
     'DisplayName', sprintf('complementary (RMSE %.2f°)', r_cf));
title(sprintf('(a) Fusion at alpha=%.2f, dt=%.0fms, fc=%.2fHz', alpha, dt*1000, fc));
xlabel('time [s]'); ylabel('angle [deg]');
legend('show', 'Location', 'northwest');

% (b) 误差曲线
subplot(2,2,2); hold on; grid on;
plot(t, ang_gyro - angle_true, 'r', 'LineWidth', 1.4, 'DisplayName', 'gyro integrate');
plot(t, ang_acc  - angle_true, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6, ...
     'DisplayName', 'accel only');
plot(t, ang_cf   - angle_true, 'b', 'LineWidth', 1.8, 'DisplayName', 'complementary');
yline(0, 'k', 'HandleVisibility', 'off');
title('(b) Error vs truth — gyro drifts linearly');
xlabel('time [s]'); ylabel('error [deg]');
legend('show', 'Location', 'northwest');

% (c) alpha 扫描
subplot(2,2,3); hold on; grid on;
semilogy(alphas, scan, 'b', 'LineWidth', 1.8, 'DisplayName', 'RMSE(\alpha)');
plot(alphas(best_i), scan(best_i), 'o', 'Color', [0.85 0.33 0.10], ...
     'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.33 0.10], 'DisplayName', 'best');
j = find(abs(alphas - 0.98) < 0.005, 1);
plot(alphas(j), scan(j), 's', 'Color', [0.90 0.50 0.05], ...
     'MarkerSize', 7, 'MarkerFaceColor', [0.90 0.50 0.05], 'DisplayName', '\alpha=0.98 (book)');
title('(c) RMSE vs \alpha — too small trusts accel, too big trusts gyro');
xlabel('\alpha (gyro weight)'); ylabel('RMSE [deg]');
legend('show', 'Location', 'northwest');

% (d) 「互补」：低通 + 高通
subplot(2,2,4); hold on; grid on;
f  = logspace(-2, log10(0.5/dt), 600);
w  = 2*pi*f*dt;
Hlp = (1 - alpha) ./ (1 - alpha*exp(-1j*w));
Hhp = 1 - Hlp;
semilogx(f, abs(Hlp), 'b', 'LineWidth', 1.8, ...
         'DisplayName', 'accel path |H| (low-pass)');
semilogx(f, abs(Hhp), 'Color', [0.90 0.50 0.05], 'LineWidth', 1.8, ...
         'DisplayName', 'gyro path |H| (high-pass)');
set(gca, 'XScale', 'log');        % hold on 之后显式切对数轴，否则 semilogx 可能被保留成线性
xline(fc, ':', 'HandleVisibility', 'off');
yline(0.5, '--', 'HandleVisibility', 'off');
title('(d) Why complementary: low-pass + high-pass = 1');
xlabel('frequency [Hz]'); ylabel('weight');
ylim([0 1.05]); legend('show', 'Location', 'west');

%% ================= 局部函数 =================
function ang = cf_run(gyro, accel, alpha, dt, a0)
% 互补滤波主循环：预测（陀螺）+ 修正（加速度计）
    n = numel(gyro);
    ang = zeros(1, n);
    ang(1) = a0;
    for k = 2:n
        pred     = ang(k-1) + gyro(k)*dt;
        ang(k)   = alpha*pred + (1 - alpha)*accel(k);
    end
end

function [tau, fc] = cf_consts(alpha, dt)
% 由 alpha 反推等效时间常数 τ 与 -3dB 截止频率 fc（与 Day 3 的一阶低通同一套公式）
    a   = 1 - alpha;                  % 加速度计那一路的低通系数 α'
    tau = dt*alpha/a;                 % τ = dt·α/(1-α)
    c   = (2 - 2*a - a^2) / (2*(1 - a));
    c   = max(-1, min(1, c));
    fc  = acos(c) / (2*pi*dt);
end
