% day2_incremental_pid.m
% 增量式 PID —— 对比位置式与增量式的两个关键区别。
%
% 现象 1：微分冲击（derivative kick）
%   设定值阶跃时，位置式的微分项产生瞬时尖峰，增量式平滑过渡。
% 现象 2：积分饱和（integral windup）
%   输出限幅时，位置式（无抗饱和）无限累积积分，松开后超调；
%   增量式没有积分累加器，天然无此问题。
%
% 运行：在 MATLAB 里直接按 F5，或在命令行输入 day2_incremental_pid

clear; clc; close all;

dt    = 0.01;              % 控制周期 10ms
kp    = 3.0;  ki = 2.0;  kd = 0.3;   % 两组对比用同一组增益

figure('Position', [100 100 1100 800]);

%% ---------- 面板 (a)(b)：突变响应（微分冲击） ----------
% 二阶欠阻尼系统 x'' + 2*zeta*wn*x' + wn^2*x = b*u
zeta = 0.3;  wn = 2.0;  b = 4.0;
duration = 6.0;  setpoint_seq = [0.0, 1.0];   % t=0 从 0 阶跃到 1

% --- 位置式 PID（带条件积分抗饱和） ---
[t1, r1, y_pos, u_pos] = sim_position(setpoint_seq, duration, dt, ...
                                      kp, ki, kd, zeta, wn, b);

% --- 增量式 PID ---
[t1, ~, y_inc, u_inc] = sim_incremental(setpoint_seq, duration, dt, ...
                                        kp, ki, kd, zeta, wn, b);

subplot(2,2,1); hold on; grid on;
plot(t1, r1, 'k--', 'DisplayName', 'setpoint');
plot(t1, y_pos, 'DisplayName', 'positional');
plot(t1, y_inc, 'DisplayName', 'incremental');
title('(a) Step response: second-order plant (output y)');
ylabel('output y'); legend('show');

subplot(2,2,2); hold on; grid on;
plot(t1, u_pos, 'DisplayName', 'positional');
plot(t1, u_inc, 'DisplayName', 'incremental');
title('(b) Control u: positional PID has derivative kick');
ylabel('control u'); legend('show');

%% ---------- 面板 (c)(d)：积分饱和 ----------
% 一阶系统，输出限幅 (0,1)，设定值先要 3（迫使饱和）再降到 0.5
duration = 10.0;  setpoint_seq = [0.0, 3.0; 5.0, 0.5];   % (时刻, 设定值)
limits = [0.0, 1.0];

% --- 无抗饱和的位置式 PID（故意演示积分饱和） ---
[t2, r2, y_naive, u_naive] = sim_position(setpoint_seq, duration, dt, ...
                                           kp, ki, kd, 1.0, 1.0, 1.0, limits);

% --- 增量式 PID（同样限幅，天然抗饱和） ---
[t2, ~, y_inc2, u_inc2] = sim_incremental(setpoint_seq, duration, dt, ...
                                          kp, ki, kd, 1.0, 1.0, 1.0, limits);

subplot(2,2,3); hold on; grid on;
plot(t2, r2, 'k--', 'DisplayName', 'setpoint');
plot(t2, y_naive, 'DisplayName', 'positional (no anti-windup)');
plot(t2, y_inc2, 'DisplayName', 'incremental');
title('(c) Integral windup: overshoot after limit released (y)');
xlabel('time [s]'); ylabel('output y'); legend('show');

subplot(2,2,4); hold on; grid on;
plot(t2, u_naive, 'DisplayName', 'positional (no anti-windup)');
plot(t2, u_inc2, 'DisplayName', 'incremental');
title('(d) Control u: positional integral keeps growing');
xlabel('time [s]'); ylabel('control u'); legend('show');


%% ================= 以下是本脚本用到的函数 =================

function sp = current_setpoint(t, setpoint_seq)
% 返回当前时刻 t 对应的设定值。setpoint_seq 是 (开始时刻, 设定值) 矩阵。
sp = setpoint_seq(1, 2);
for i = 1:size(setpoint_seq, 1)
    if t >= setpoint_seq(i, 1)
        sp = setpoint_seq(i, 2);
    end
end
end


function [t, r, y, u] = sim_position(setpoint_seq, duration, dt, ...
                                     kp, ki, kd, zeta, wn, b, limits)
% 位置式 PID 仿真。二阶系统参数为 (zeta, wn, b)；传 zeta=1,wn=1,b=1 时
% 走一阶系统分支（见下方 plant_step）。limits=[lo,hi] 为空时表示不限幅。
if nargin < 11
    limits = [];       % 默认不限幅
end
n = floor(duration / dt) + 1;
t = (0:n-1)*dt;  r = zeros(1,n);  y = zeros(1,n);  u = zeros(1,n);

integral = 0;  e_prev = 0;   % PID 状态
x = 0;  v = 0;               % 被控对象状态

for k = 1:n
    t(k) = (k-1)*dt;
    r(k) = current_setpoint(t(k), setpoint_seq);
    y(k) = x;

    e = r(k) - y(k);
    integral_candidate = integral + e*dt;          % 积分候选值
    derivative = (e - e_prev)/dt;
    u(k) = kp*e + ki*integral_candidate + kd*derivative;

    % 输出限幅 + 条件积分抗饱和
    if ~isempty(limits)
        if u(k) > limits(2)
            u(k) = limits(2);
        elseif u(k) < limits(1)
            u(k) = limits(1);
        else
            integral = integral_candidate;         % 未饱和才累积积分
        end
    else
        integral = integral_candidate;
    end

    [x, v] = plant_step(u(k), x, v, dt, zeta, wn, b);
    e_prev = e;
end
end


function [t, r, y, u] = sim_incremental(setpoint_seq, duration, dt, ...
                                        kp, ki, kd, zeta, wn, b, limits)
% 增量式 PID 仿真。与 sim_position 只差 PID 递推式不同。
if nargin < 11
    limits = [];
end
n = floor(duration / dt) + 1;
t = (0:n-1)*dt;  r = zeros(1,n);  y = zeros(1,n);  u = zeros(1,n);

e_prev  = 0;   % e[k-1]
e_prev2 = 0;   % e[k-2]
u_last  = 0;   % 上一次输出
x = 0;  v = 0;

for k = 1:n
    t(k) = (k-1)*dt;
    r(k) = current_setpoint(t(k), setpoint_seq);
    y(k) = x;

    e = r(k) - y(k);
    % 增量式核心递推式（对位置式做一阶差分消去积分累加项）
    du = kp*(e - e_prev) + ki*e + kd*(e - 2*e_prev + e_prev2);
    u(k) = u_last + du;

    % 输出限幅（夹住 u 也就夹住了"记忆"，天然抗积分饱和）
    if ~isempty(limits)
        if u(k) > limits(2)
            u(k) = limits(2);
        elseif u(k) < limits(1)
            u(k) = limits(1);
        end
    end

    [x, v] = plant_step(u(k), x, v, dt, zeta, wn, b);

    % 移位更新
    e_prev2 = e_prev;
    e_prev  = e;
    u_last  = u(k);
end
end


function [x, v] = plant_step(u, x, v, dt, zeta, wn, b)
% 被控对象一步递推。
% 二阶系统：x'' + 2*zeta*wn*x' + wn^2*x = b*u，写成两个一阶状态。
% 特例：zeta=1, wn=1, b=1 时退化成一阶惯性环节 dx/dt = -x + u（前向欧拉）。
v = v + dt*(-2*zeta*wn*v - wn^2*x + b*u);
x = x + dt*v;
end
