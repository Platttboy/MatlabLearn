%% 主脚本部分
clear;  % 清除所有变量
clc;    % 清除命令窗口的输出

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

% 目标函数（误差值，用于最小化）
function F = objective_function(params, x_samples, y_samples)
    predicted = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:size(x_samples, 1)).';
    F = predicted - y_samples;  % 返回的是残差
end

function [best_params, best_value] = pso(objective_func,dim,pop_size,max_iter,lb,ub,w,c1,c2)
% my_pso 使用经典粒子群算法求解目标函数最小值
%
% 输入:
%   objective_func:   目标函数句柄，输入为一个粒子（1×dim），输出为标量目标函数值
%   dim:              粒子维度（与待优化参数个数相同）
%   pop_size:         粒子群规模
%   max_iter:         最大迭代次数
%   lb:               参数下界 (1×dim)
%   ub:               参数上界 (1×dim)
%   w :权重参数
%   c1: 个体系数
%   c2: 社会系数
%
% 输出:
%   best_params:      PSO 优化得到的最优参数
%   best_value:       对应的最优目标函数值
    X = repmat(lb,pop_size,1) + rand(pop_size,dim).*(repmat(ub-lb,pop_size,1));
    V = zeros(pop_size,dim);

    %初始化适应度函数

    fitness = arrayfun(@(idx) objective_func(X(idx,:)),1:pop_size)';

    pbest = X;
    pbest_value = fitness;

    [gbest_value,best_idx] = min(fitness);
    gbest = pbest(best_idx,:);

    %迭代
    for iter = 1:max_iter
        for n = 1:pop_size
            V(n,:) = w * V(n,:)+c1*rand(1,dim).*(pbest(n,:) - X(n,:))+c2*rand(1,dim).*(gbest-X(n,:));
            X(n,:) = X(n,:) + V(n,:);

            X(n,:) = min(max(X(n,:),lb),ub);

            current_fitness = objective_func(X(n,:));

            if current_fitness<pbest_value(n)
                pbest_value(n) = current_fitness;
                pbest(n,:) = X(n,:);
            end

        end

        [current_best_value,best_idx] = min(pbest_value);

        if current_best_value < gbest_value
            gbest_value = current_best_value;
            gbest = pbest(best_idx,:);
        end

        disp(['Iteration :' num2str(iter)  '  Best_Value :' num2str(gbest_value)]);
    end

    best_params = gbest;
    best_value = gbest_value;
end

% 真实参数 - 这些参数用于定义模拟数据生成函数的行为
true_params = [1.4, 2.2, 3.3, 4.1, 5.7];

% 设置生成模拟数据的参数
num_samples = 100;   % 生成数据的样本数量
noise_std = 0.1;     % 噪声的标准差，用于生成带噪声的输出数据

% 生成模拟数据
[x_samples, y_samples_noisy] = generate_data(true_params, num_samples, noise_std);

% 初始参数猜测（仅用于后续查看差距，可选）
initial_params = [0.5, 1.5, 2.5, 3.5, 4.5];

%% 使用自编写 PSO 进行优化
disp('Running Particle Swarm Optimization (custom-coded)...');

% 定义粒子群算法的参数
pop_size = 50;
dim = 5;
max_iterations = 200;
w = 0.7;
c1 = 1.5;
c2 = 1.5;

% 定义上下界
lb = [-10,-10,-10,-10,-10];
up = [10,10,10,10,10];

% 调用自编写的 PSO 函数
[best_params_pso,best_value_pso] = pso(@(p) sum((objective_function(p,x_samples, y_samples_noisy)).^2),dim,pop_size,max_iterations,lb,up,w,c1,c2);

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








