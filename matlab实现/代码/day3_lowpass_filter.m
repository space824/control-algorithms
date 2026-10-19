% day3_lowpass_filter.m
% 一阶低通滤波：含噪正弦滤波 + α 与截止频率的关系。
%
% 一阶 RC 低通的离散形式（指数移动平均）：
%     y[k] = α·x[k] + (1-α)·y[k-1]
%
% 本脚本做四件事（四张子图）：
%     (a) 滤波前后对比      (b) 不同 α 对比
%     (c) 幅频响应          (d) 截止频率 vs α
%
% 运行：在 MATLAB 里直接按 F5，或在命令行输入 day3_lowpass_filter

clear; clc; close all;

%% ---------- 1. 仿真基本参数 ----------
dt       = 0.001;      % 采样周期 1ms（1kHz 采样率）
duration = 1.0;        % 1 秒（3 个周期）
t        = 0:dt:duration;
n        = length(t);
Fs       = 1/dt;       % 采样率

%% ---------- 2. 生成含噪声的正弦信号 ----------
f_sig   = 3.0;                      % 正弦频率 3Hz
x_clean = sin(2*pi*f_sig*t);        % 干净正弦
rng(0);                             % 固定随机种子，结果可复现
noise   = 0.4*randn(1, n);          % 高斯噪声，标准差 0.4
x_noisy = x_clean + noise;          % 含噪信号

alphas = [0.05 0.2 0.6];            % 三档平滑系数：低 / 中 / 高

figure('Position', [100 100 850 720]);

%% ---------- (a) 滤波前后对比（取 α=0.2） ----------
subplot(2,2,1); hold on; grid on;
y = filter(0.2, [1 -0.8], x_noisy);      % b=[α], a=[1 -(1-α)]
plot(t, x_clean, 'k--', 'LineWidth', 2, 'DisplayName', 'clean sine');
plot(t, x_noisy, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6, 'DisplayName', 'noisy input');
plot(t, y, 'r', 'DisplayName', 'filtered (\alpha=0.2)');
title('(a) Noisy sine \rightarrow first-order low-pass');
ylabel('amplitude'); legend('show');

%% ---------- (b) 不同 α 滤波同一段含噪信号 ----------
subplot(2,2,2); hold on; grid on;
plot(t, x_clean, 'k--', 'LineWidth', 2, 'DisplayName', 'clean sine');
plot(t, x_noisy, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6, 'DisplayName', 'noisy input');
for a = alphas
    yb = filter(a, [1 -(1-a)], x_noisy);
    plot(t, yb, 'DisplayName', sprintf('\\alpha=%.2f', a));
end
title('(b) Same noisy input, different \alpha');
ylabel('amplitude'); legend('show');

%% ---------- (c) 幅频响应（freqz 直接给出 -3dB 截止频率） ----------
subplot(2,2,3); hold on; grid on;
for a = alphas
    [H, f] = freqz(a, [1 -(1-a)], 1024, Fs);
    semilogx(f, 20*log10(abs(H)), 'DisplayName', sprintf('\\alpha=%.2f', a));
end
set(gca, 'XScale', 'log');   % hold on 之后必须显式切对数轴
yline(-3, '--', 'DisplayName', '-3 dB');
title('(c) Magnitude response |H(f)|');
xlabel('frequency [Hz]'); ylabel('magnitude [dB]');
legend('show'); ylim([-40 1]);

%% ---------- (d) 截止频率 fc 与 α 的关系 ----------
subplot(2,2,4); hold on; grid on;
a_axis = linspace(0.001, 0.8, 1000);   % α>0.828 无 -3dB 点，画到 0.8
fc = acos((2 - 2*a_axis - a_axis.^2) ./ (2*(1 - a_axis))) / (2*pi*dt);
plot(a_axis, fc, 'b', 'DisplayName', 'fc(\alpha)');
xlabel('\alpha (smoothing factor)');
ylabel('-3dB cutoff frequency [Hz]');
title('(d) Cutoff frequency vs \alpha'); legend('show');
