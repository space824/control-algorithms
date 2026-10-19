% day4_second_order_lowpass.m
% 二阶低通滤波（巴特沃斯 biquad）—— 与一阶正面对比。
%
% 一阶：y[k] = α·x[k] + (1-α)·y[k-1]                 → 高频 -20 dB/十倍频
% 二阶：y[k] = b0·x[k] + b1·x[k-1] + b2·x[k-2]
%                       − a1·y[k-1] − a2·y[k-2]      → 高频 -40 dB/十倍频
%
% 本脚本做四件事（四张子图）：
%     (a) 同一段含噪信号：一阶 vs 二阶     (b) 阶跃响应：超调 0% vs 4.32%
%     (c) 幅频响应：滚降斜率差一倍          (d) 延迟 vs 残余噪声：白噪声 vs 窄带干扰
%
% 运行：在 MATLAB 里直接按 F5，或在命令行输入 day4_second_order_lowpass

clear; clc; close all;

%% ---------- 1. 仿真基本参数 ----------
dt       = 0.001;      % 采样周期 1ms（1kHz 采样率），与 Day 3 一致
duration = 1.0;
t        = 0:dt:duration-dt;
n        = numel(t);
Fs       = 1/dt;

f_sig   = 3.0;         % 正弦信号频率 3Hz
f_int   = 120.0;       % 窄带干扰频率（位于 fc 上方约 3.4 倍）
fc      = 35.7;        % 截止频率：正好是 Day 3 里 α=0.2 对应的 fc
sigma   = 0.4;         % 高斯噪声标准差
Q       = 1/sqrt(2);   % 巴特沃斯：Q = 0.7071

%% ---------- 2. 生成含噪声的正弦信号 ----------
x_clean = sin(2*pi*f_sig*t);        % 干净正弦
rng(0);                             % 固定随机种子，结果可复现
x_noisy = x_clean + sigma*randn(1, n);

%% ---------- 3. 设计两个滤波器（同一个 fc，公平对比） ----------
al = design1(fc, dt);               % 一阶的 α（离散域精确解）
[b, a] = design2(fc, dt, Q);        % 二阶的 b、a 系数

fprintf('fc = %.1f Hz | 采样 1/%d s | 信号 %.0f Hz | 干扰 %.0f Hz\n', fc, Fs, f_sig, f_int);
fprintf('一阶 alpha = %.6f\n', al);
fprintf('二阶 b = [%.6f %.6f %.6f]\n', b);
fprintf('二阶 a = [%.6f %.6f %.6f]\n', a);
fprintf('二阶直流增益 = %.8f\n', sum(b)/sum(a));

y1 = filter(al,  [1 -(1-al)], x_noisy);     % 一阶
y2 = filter(b,   a,           x_noisy);     % 二阶

%% ---------- 4. 量化指标（与 Python 版口径一致） ----------
z_sig  = exp(-1j*2*pi*f_sig*dt);
z_int  = exp(-1j*2*pi*f_int*dt);

H1 = (al) / (1 - (1-al)*z_sig);                                      % 一阶在 f_sig 处
H2 = (b(1) + b(2)*z_sig + b(3)*z_sig^2) / ...
     (a(1) + a(2)*z_sig + a(3)*z_sig^2);                             % 二阶在 f_sig 处
d1 = -angle(H1)/(2*pi*f_sig)*1000;      % 相位延迟 ms
d2 = -angle(H2)/(2*pi*f_sig)*1000;

G1 = (al) / (1 - (1-al)*z_int);
G2 = (b(1) + b(2)*z_int + b(3)*z_int^2) / ...
     (a(1) + a(2)*z_int + a(3)*z_int^2);

ng1 = noisegain([al 0 0], [1 -(1-al) 0]);
ng2 = noisegain(b, a);

rmse = @(x, y) sqrt(mean((x - y).^2));
fprintf('\n%-16s%9s%10s%11s%12s\n', '', 'RMSE', '白噪放大', '延迟@3Hz', '120Hz衰减');
fprintf('%-16s%9.4f%10.4f%10.2fms%11.1fdB\n', '含噪输入', rmse(x_noisy, x_clean), 1, 0, 0);
fprintf('%-16s%9.4f%10.4f%10.2fms%11.1fdB\n', '一阶 low-pass', rmse(y1, x_clean), ng1, d1, 20*log10(abs(G1)));
fprintf('%-16s%9.4f%10.4f%10.2fms%11.1fdB\n', '二阶 low-pass', rmse(y2, x_clean), ng2, d2, 20*log10(abs(G2)));

%% ---------- 5. 阶跃响应（用更细的步长看清超调） ----------
dt_s = 0.0002;  n_s = 400;
al_s = design1(fc, dt_s);
[b_s, a_s] = design2(fc, dt_s, Q);
ys1 = filter(al_s, [1 -(1-al_s)], ones(1, n_s));
ys2 = filter(b_s,  a_s,            ones(1, n_s));
ts  = (0:n_s-1)*dt_s;

fprintf('\n%-16s%9s%10s%10s\n', '阶跃', '超调%', '上升ms', '调节ms');
names = {'一阶', '二阶'};
resps = {ys1, ys2};
for k = 1:2
    yy   = resps{k};
    ov   = (max(yy) - 1)*100;
    i10  = find(yy >= 0.1, 1);  i90 = find(yy >= 0.9, 1);
    tr   = (i90 - i10)*dt_s*1000;
    out  = find(abs(yy - 1) > 0.02, 1, 'last');
    fprintf('%-16s%9.2f%10.2f%10.2f\n', names{k}, ov, tr, out*dt_s*1000);
end

%% ---------- 6. 出图 ----------
figure('Position', [100 100 900 760]);

% (a) 同一段含噪信号：一阶 vs 二阶
subplot(2,2,1); hold on; grid on;
plot(t, x_clean, 'k--', 'LineWidth', 2, 'DisplayName', 'clean sine');
plot(t, x_noisy, 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6, 'DisplayName', 'noisy input');
plot(t, y1, 'Color', [0.90 0.50 0.05], 'LineWidth', 1.4, ...
     'DisplayName', sprintf('1st order (delay %.1f ms)', d1));
plot(t, y2, 'Color', [0.00 0.45 0.74], 'LineWidth', 1.8, ...
     'DisplayName', sprintf('2nd order (delay %.1f ms)', d2));
title(sprintf('(a) Same noise, same fc=%.1f Hz', fc));
xlabel('time [s]'); ylabel('amplitude'); legend('show', 'Location', 'northeast');

% (b) 阶跃响应
subplot(2,2,2); hold on; grid on;
plot(ts*1000, ys1, 'Color', [0.90 0.50 0.05], 'LineWidth', 1.6, ...
     'DisplayName', '1st order');
plot(ts*1000, ys2, 'Color', [0.00 0.45 0.74], 'LineWidth', 2.0, ...
     'DisplayName', '2nd order');
yline(1, ':', 'HandleVisibility', 'off');
title('(b) Step response — 2nd order rings slightly');
xlabel('time [ms]'); ylabel('output');
ylim([-0.05 1.25]); legend('show', 'Location', 'southeast');

% (c) 幅频响应
subplot(2,2,3); hold on; grid on;
[H1f, f1] = freqz(al, [1 -(1-al)], 1024, Fs);
[H2f, f2] = freqz(b,  a,           1024, Fs);
semilogx(f1, 20*log10(abs(H1f)), 'Color', [0.90 0.50 0.05], 'LineWidth', 1.8, ...
         'DisplayName', '1st order: -20 dB/dec');
semilogx(f2, 20*log10(abs(H2f)), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.8, ...
         'DisplayName', '2nd order: -40 dB/dec');
set(gca, 'XScale', 'log');    % hold on 之后显式切对数轴
yline(-3, '--', 'HandleVisibility', 'off');
xline(fc, ':', 'HandleVisibility', 'off');
xline(f_int, ':', 'Color', [0.85 0.33 0.10], 'HandleVisibility', 'off');
title('(c) Magnitude response — same -3dB, steeper roll-off');
xlabel('frequency [Hz]'); ylabel('magnitude [dB]');
ylim([-60 2]); legend('show', 'Location', 'northeast');

% (d) 延迟 vs 残余噪声：白噪声 / 窄带干扰
subplot(2,2,4); hold on; grid on;
fcs = logspace(log10(4), log10(300), 80);
d1s = zeros(size(fcs)); d2s = d1s; wn1 = d1s; wn2 = d1s; nb1 = d1s; nb2 = d1s;
for i = 1:numel(fcs)
    al_i = design1(fcs(i), dt);
    [b_i, a_i] = design2(fcs(i), dt, Q);
    d1s(i) = -angle(  al_i/(1-(1-al_i)*z_sig) )/(2*pi*f_sig)*1000;
    d2s(i) = -angle((b_i(1)+b_i(2)*z_sig+b_i(3)*z_sig^2)/ ...
                    (a_i(1)+a_i(2)*z_sig+a_i(3)*z_sig^2))/(2*pi*f_sig)*1000;
    wn1(i) = 20*log10(noisegain([al_i 0 0], [1 -(1-al_i) 0]));
    wn2(i) = 20*log10(noisegain(b_i, a_i));
    g1 = abs(  al_i/(1-(1-al_i)*z_int) );
    g2 = abs((b_i(1)+b_i(2)*z_int+b_i(3)*z_int^2)/ ...
             (a_i(1)+a_i(2)*z_int+a_i(3)*z_int^2));
    nb1(i) = 20*log10(max(g1, 1e-12));
    nb2(i) = 20*log10(max(g2, 1e-12));
end
plot(d1s, wn1, '--', 'Color', [0.90 0.50 0.05], 'LineWidth', 1.4, 'DisplayName', '1st, white noise');
plot(d2s, wn2, '--', 'Color', [0.00 0.45 0.74], 'LineWidth', 1.4, 'DisplayName', '2nd, white noise');
plot(d1s, nb1, '-',  'Color', [0.90 0.50 0.05], 'LineWidth', 1.8, ...
     'DisplayName', sprintf('1st, %.0fHz interference', f_int));
plot(d2s, nb2, '-',  'Color', [0.00 0.45 0.74], 'LineWidth', 1.8, ...
     'DisplayName', sprintf('2nd, %.0fHz interference', f_int));
plot(d1, 20*log10(ng1), 'o', 'Color', [0.90 0.50 0.05], 'MarkerSize', 7, 'HandleVisibility', 'off');
plot(d2, 20*log10(ng2), 'o', 'Color', [0.00 0.45 0.74], 'MarkerSize', 7, 'HandleVisibility', 'off');
title(sprintf('(d) Delay vs residual noise (fc=%.1fHz marked)', fc));
xlabel('phase delay at 3 Hz [ms]'); ylabel('residual level [dB re input]');
legend('show', 'Location', 'southwest');

%% ================= 局部函数 =================
function [b, a] = design2(fc, dt, Q)
% 二阶低通（双线性变换 + 预畸变，RBJ 闭式）。Q = 1/√2 时即巴特沃斯。
    w0    = 2*pi*fc*dt;
    c     = cos(w0);
    s     = sin(w0);
    alpha = s/(2*Q);
    a0    = 1 + alpha;
    b = [(1-c)/2/a0, (1-c)/a0, (1-c)/2/a0];   % b0 b1 b2
    a = [1, -2*c/a0, (1-alpha)/a0];           % a0(=1) a1 a2
end

function al = design1(fc, dt)
% 一阶低通的 α（离散域精确解，与 Python 版 from_cutoff 一致）
    w0 = 2*pi*fc*dt;
    k  = 1 - cos(w0);
    al = -k + sqrt(k*(k+2));
end

function ng = noisegain(b, a)
% 白噪声放大倍数 = sqrt(mean(|H|^2))，ω 在 [0, π] 上均匀取样
    w   = linspace(0, pi, 4096);
    ejw = exp(-1j*w);
    num = b(1) + b(2)*ejw + b(3)*ejw.^2;
    den = a(1) + a(2)*ejw + a(3)*ejw.^2;
    ng  = sqrt(mean(abs(num./den).^2));
end
