%% 主脚本部分
clear;  % 清除所有变量
clc;    % 清除命令窗口的输出

% 真实参数 - 这些参数用于定义模拟数据生成函数的行为
true_params = [1.4, 2.2, 3.3, 4.1, 5.7];

% 设置生成模拟数据的参数
num_samples = 100;   % 生成数据的样本数量
noise_std = 0.1;     % 噪声的标准差，用于生成带噪声的输出数据

% 生成模拟数据
[x_samples, y_samples_noisy] = generate_data(true_params, num_samples, noise_std);


% 初始参数猜测
initial_params = [0.5, 1.5, 2.5, 3.5, 4.5];

%% 粒子群优化
disp('Running Particle Swarm Optimization...');
lb = [-10, -10, -10, -10, -10]; % 参数下界
ub = [10, 10, 10, 10, 10]; % 参数上界
options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 50, 'MaxIterations', 100);
[best_params_pso, best_value_pso] = particleswarm(@(p) sum((objective_function(p, x_samples, y_samples_noisy)).^2), length(initial_params), lb, ub, options);

disp('Best Parameters (PSO):');
disp(best_params_pso);
disp('Best Value (PSO):');
disp(best_value_pso);

% 可视化显示原始数据和拟合数据
y_fitted_pso = arrayfun(@(i) multi_param_function(x_samples(i, :), best_params_pso), 1:num_samples);

figure;
scatter(1:num_samples, y_samples_noisy,50, 'b', 'DisplayName', 'Noisy Data');
hold on;
plot(1:num_samples, y_fitted_pso, 'm-.', 'LineWidth', 1.5, 'DisplayName', 'PSO Fit');
scatter(1:num_samples, y_fitted_pso,25, 'g', 'DisplayName', 'Predicted Data');
legend;
xlabel('Sample Index');
ylabel('Function Value');
title('Data Fitting Comparison');
grid on;

%% 函数定义部分

% 定义多参数函数，用于根据输入和参数计算模型的输出
function y = multi_param_function(x, params)
    % 从参数数组中提取各参数
    a1 = params(1);
    a2 = params(2);
    a3 = params(3);
    a4 = params(4);
    a5 = params(5);
    % 根据模型定义计算输出
    y = a1 * sin(x(1)) + a2 * cos(x(2)) + a3 * x(3)^2 + a4 * x(4) + a5;
end

% 生成模拟数据的函数
function [x_samples, y_samples_noisy] = generate_data(params, num_samples, noise_std)
    % 生成输入样本，每个样本4个特征，特征值在[-100, 100]之间均匀分布
    x_samples = -100 + 200 * rand(num_samples, 4); % 每行表示一个样本
    % 计算每个样本的理想输出
    y_samples = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:num_samples).';
    % 生成正态分布噪声，并添加到理想输出上，生成带噪声的输出数据
    noise = noise_std * randn(num_samples, 1);
    y_samples_noisy = y_samples + noise;
end

% 目标函数（误差平方和，用于最小化）
function F = objective_function(params, x_samples, y_samples)
    predicted = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:size(x_samples, 1)).';
    F = predicted - y_samples;
end












































