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

%% 自定义粒子群优化
disp('Running Custom Particle Swarm Optimization...');
lb = [-10, -10, -10, -10, -10]; % 参数下界
ub = [10, 10, 10, 10, 10]; % 参数上界
swarm_size = 50; % 粒子数量
max_iterations = 100; % 最大迭代次数
w = 0.7; % 惯性权重
c1 = 1.8; % 认知系数
c2 = 1.8; % 社会系数

[best_params_pso, best_value_pso] = custom_particleswarm(@(p) sum((objective_function(p, x_samples, y_samples_noisy)).^2), length(initial_params), lb, ub, swarm_size, max_iterations, w, c1, c2);

disp('Best Parameters (Custom PSO):');
disp(best_params_pso);
disp('Best Value (Custom PSO):');
disp(best_value_pso);

% 可视化显示原始数据和拟合数据
y_fitted_pso = arrayfun(@(i) multi_param_function(x_samples(i, :), best_params_pso), 1:num_samples);

figure;
scatter(1:num_samples, y_samples_noisy,50, 'b', 'DisplayName', 'Noisy Data');
hold on;
plot(1:num_samples, y_fitted_pso, 'm-.', 'LineWidth', 1.5, 'DisplayName', 'Custom PSO Fit');
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

% 自定义粒子群优化函数
function [best_position, best_cost] = custom_particleswarm(costfcn, dim, lb, ub, pop_size, max_iter, w, c1, c2)
% my_pso 使用经典粒子群算法求解目标函数最小值
%
% 输入:
%   costfcn:          目标函数，输入为一个粒子（1×dim），输出为标量目标函数值
%   dim:              粒子维度（与待优化参数个数相同）
%   lb:               参数下界 (1×dim)
%   ub:               参数上界 (1×dim)
%   pop_size:         粒子群规模
%   max_iter:         最大迭代次数
%   w:                惯性权重
%   c1:               认知系数
%   c2:               社会系数
%
% 输出:
%   best_position:    PSO 优化得到的最优参数
%   best_cost:        对应的最优目标函数值

    % 初始化粒子位置和速度
    positions = lb + (ub - lb) .* rand(pop_size, dim); % x = xmin+rand*(xmax-xmin)
    velocities = zeros(pop_size, dim); %v = 0
    
    % 初始化个体最优位置和个人最优代价
    personal_best_positions = positions; %pbext_x = x
    personal_best_costs = arrayfun(@(i) costfcn(personal_best_positions(i, :)), 1:pop_size);%pbest_value = projective_func(x)
    
    % 初始化全局最优位置和全局最优代价
    [~, idx] = min(personal_best_costs);%寻找全局最优索引
    global_best_position = personal_best_positions(idx, :);%设置gbest
    global_best_cost = personal_best_costs(idx);%设置gbest——value
    
    for iter = 1:max_iter %迭代次数
        for i = 1:pop_size %粒子规模
            % 更新速度
            r1 = rand(dim, 1);%随机r1
            r2 = rand(dim, 1);%随机r2
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * r1' .* (personal_best_positions(i, :) - positions(i, :)) + ...
                c2 * r2' .* (global_best_position - positions(i, :)); %设置速度
            %v = 权重*速度+个体系数*个体随机值*（个体最优x-当前x）+全局系数*全局随机值*（全局最优-个体x）
            
            % 更新位置
            positions(i, :) = positions(i, :) + velocities(i, :);
            %x = x+v
            % 边界检查
            positions(i, :) = max(min(positions(i, :), ub), lb);
            %x=max(min(x,边界下限)，边界上限)
            % 计算新位置的成本
            new_cost = costfcn(positions(i, :));
            %当前适应度值 = 适度函数（x）;

            % 更新个人最优位置和个人最优代价
            if new_cost < personal_best_costs(i)%如果当前适应度值小于个体最优适应度值
                personal_best_positions(i, :) = positions(i, :); %个人最优x则等于当前x
                personal_best_costs(i) = new_cost;%个人最优适应值则等于当前适应度值
                
                % 更新全局最优位置和全局最优代价
                if new_cost < global_best_cost %如果当前适应度值小于群体最优适应度值
                    global_best_position = positions(i, :);%群体最优x则等于当前x
                    global_best_cost = new_cost;%群体最优适应值则等于当前适应度值
                end
            end
        end
        
        disp(['Iteration ', num2str(iter), ': Best Cost = ', num2str(global_best_cost)]);
    end
    
    best_position = global_best_position;
    best_cost = global_best_cost;
end



