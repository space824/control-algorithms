% day1_position_pid.m
% 位置式 PID 控制一阶惯性环节，阶跃响应仿真。
%
% 被控对象（一阶惯性环节）：
%     dx/dt = -a*x + b*u
% 前向欧拉离散化：
%     x[k+1] = x[k] + dt * (-a*x[k] + b*u[k])
%
% 本脚本对比三组参数，直观感受 P / I / D 的作用。
%
% 运行：在 MATLAB 里直接按 F5，或在命令行输入 day1_position_pid

clear; clc; close all;

%% ---------- 1. 仿真基本参数 ----------
dt       = 0.001;     % 控制周期 1ms，越小仿真越精确（但更慢）
duration = 10.0;      % 仿真 10 秒
t        = 0:dt:duration;   % 时间轴：0, 0.001, 0.002, ...
n        = length(t);       % 总步数
setpoint = 1.0;             % 阶跃设定值

% 被控对象参数：dx/dt = -a*x + b*u
a = 1.0;
b = 0.3;

%% ---------- 2. 三组 PID 参数，对比 P / I / D ----------
% 每一行是 [Kp, Ki, Kd]，和 Python 版 day1_position_pid.py 完全一致
P     = [3.0  2.0  0.0;   % P only（带一点积分）
         3.0  3.0  0.0;   % PI
         4.0  2.0  2.0];  % PID
names = {'P only', 'PI', 'PID'};

%% ---------- 3. 画布：上面画输出 y，下面画控制量 u ----------
figure('Position', [100 100 700 600]);
subplot(2,1,1); hold on; grid on;
title('Position PID: step response of first-order plant');
ylabel('output y');
subplot(2,1,2); hold on; grid on;
xlabel('time [s]'); ylabel('control u');

%% ---------- 4. 对每组参数各跑一次仿真 ----------
for c = 1:3
    kp = P(c,1);  ki = P(c,2);  kd = P(c,3);

    % 状态初始化：每次换参数都要清零，否则会串
    x        = 0;   % 被控对象输出初值
    integral = 0;   % 积分累加器
    e_prev   = 0;   % 上一个误差（供微分用）

    y = zeros(1, n);   % 输出 y[k]
    u = zeros(1, n);   % 控制量 u[k]

    for k = 1:n
        y(k) = x;                              % 记录当前输出
        e    = setpoint - y(k);                % 误差
        integral  = integral + e*dt;           % 积分：累加求和
        derivative = (e - e_prev)/dt;          % 微分：后向差分
        u(k) = kp*e + ki*integral + kd*derivative;   % PID 公式

        x = x + dt*(-a*x + b*u(k));            % 被控对象前向欧拉
        e_prev = e;                            % 更新误差记忆
    end

    subplot(2,1,1); plot(t, y, 'DisplayName', names{c});
    subplot(2,1,2); plot(t, u, 'DisplayName', names{c});
end

%% ---------- 5. 补画设定值虚线 + 图例 ----------
subplot(2,1,1); yline(setpoint, 'k--', 'DisplayName', 'setpoint'); legend('show');
subplot(2,1,2); legend('show');
