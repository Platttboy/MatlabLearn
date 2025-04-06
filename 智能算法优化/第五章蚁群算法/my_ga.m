%% 主脚本部分
clear;  % 清除所有工作空间中的变量
clc;    % 清除命令窗口中的输出

% 真实参数 - 用于生成模拟数据
true_params = [1.4, 2.2, 3.3, 4.1, 5.7];

% 设置生成模拟数据的参数
num_samples = 100;   % 生成数据的样本数量
noise_std = 0.1;     % 噪声的标准差

% 生成模拟数据
[x_samples, y_samples_noisy] = generate_data(true_params, num_samples, noise_std);

% 初始参数猜测（可选，用于查看差距）
initial_params = [0.5, 1.5, 2.5, 3.5, 4.5];

%% 使用自编写的模拟退火算法进行优化
disp('Running Simulated Annealing...');

% 定义模拟退火的主要参数
dim = 5;               % 参数维度
lb = [-10, -10, -10, -10, -10];  % 参数下界
ub = [10, 10, 10, 10, 10];       % 参数上界

T0 = 1000.0;        % 初始温度
alpha = 0.95;    % 温度衰减系数
max_iterations = 2000;  % 最大迭代次数

% 定义目标函数（损失函数），这里与PSO一样
obj_func = @(p) sum((objective_function(p, x_samples, y_samples_noisy)).^2);

% 调用自编写的模拟退火函数
[best_params_sa, best_value_sa] = simulated_annealing(obj_func, dim, lb, ub, T0, alpha, max_iterations, initial_params);

disp('Best Parameters (SA):');
disp(best_params_sa);
disp('Best Value (SA):');
disp(best_value_sa);

% 可视化显示原始数据和拟合数据
y_fitted_sa = arrayfun(@(i) multi_param_function(x_samples(i, :), best_params_sa), 1:num_samples);

figure;
scatter(1:num_samples, y_samples_noisy, 50, 'b', 'DisplayName', 'Noisy Data');
hold on;
plot(1:num_samples, y_fitted_sa, 'm-.', 'LineWidth', 1.5, 'DisplayName', 'SA Fit');
scatter(1:num_samples, y_fitted_sa, 25, 'g', 'DisplayName', 'Predicted Data');
legend;
xlabel('Sample Index');
ylabel('Function Value');
title('Data Fitting Comparison (Simulated Annealing)');
grid on;


function y = multi_param_function(x, params)
    % multi_param_function - 根据输入 x 和模型参数 params，计算函数输出
    % x: 1×4 的输入特征向量
    % params: 1×5 的参数向量 [a1, a2, a3, a4, a5]

    a1 = params(1);
    a2 = params(2);
    a3 = params(3);
    a4 = params(4);
    a5 = params(5);

    y = a1 * sin(x(1)) + a2 * cos(x(2)) + a3 * x(3)^2 + a4 * x(4) + a5;
end


function [x_samples, y_samples_noisy] = generate_data(params, num_samples, noise_std)
    % generate_data - 生成模拟数据
    % params: 真实参数向量
    % num_samples: 样本数量
    % noise_std: 噪声标准差
    
    % 生成每个样本的输入特征（4维）
    x_samples = -100 + 200 * rand(num_samples, 4);
    
    % 计算每个样本的理想输出
    y_samples = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:num_samples).';
    
    % 生成噪声并添加到理想输出
    noise = noise_std * randn(num_samples, 1);
    y_samples_noisy = y_samples + noise;
end


function F = objective_function(params, x_samples, y_samples)
    % objective_function - 计算预测输出和真实输出之间的残差
    % params: 参数向量
    % x_samples: 每个样本的输入特征
    % y_samples: 带噪声的真实输出

    predicted = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:size(x_samples, 1)).';
    F = predicted - y_samples;  % 返回残差向量
end




function [best_params, best_value] = simulated_annealing(objective_func, dim, lb, ub, T0, alpha, max_iter, initial_params)
% simulated_annealing - 使用模拟退火算法求解最小化问题
%
% 输入参数：
%   objective_func:   目标函数句柄，输入为参数 (1×dim)，输出为标量损失
%   dim:              参数维度
%   lb:               参数下界 (1×dim)
%   ub:               参数上界 (1×dim)
%   T0:               初始温度
%   alpha:            温度衰减系数 (0 < alpha < 1)
%   max_iter:         最大迭代次数
%   initial_params:   初始猜测的参数 (1×dim)
%
% 输出参数：
%   best_params:      模拟退火最终得到的最优参数
%   best_value:       对应的最优目标函数值

    % 若用户未提供初始参数，可随机在上下界内生成
    if nargin < 8 || isempty(initial_params)
        current_solution = lb + (ub - lb) .* rand(1, dim);
    else
        current_solution = initial_params;
    end

    % 计算当前解的目标函数值
    current_cost = objective_func(current_solution);

    % 初始化全局最优解
    best_params = current_solution;
    best_value = current_cost;

    % 初始温度
    T = T0;

    % 模拟退火迭代
    for iter = 1:max_iter
        % 在邻域生成新解 (此处用随机扰动示例)
        new_solution = generate_neighbor(current_solution, lb, ub);

        % 计算新解的目标函数值
        new_cost = objective_func(new_solution);

        % 判断是否接受新解
        if new_cost < current_cost
            % 如果新解更优，直接接受
            current_solution = new_solution;
            current_cost = new_cost;

            % 如果当前解优于全局最优，则更新
            if new_cost < best_value
                best_value = new_cost;
                best_params = new_solution;
            end
        else
            % 如果新解更差，以一定概率接受
            delta = new_cost - current_cost;
            % 退火接受概率
            acceptance_prob = exp(-delta / T);
            if rand() < acceptance_prob
                current_solution = new_solution;
                current_cost = new_cost;
            end
        end

        % 温度衰减
        T = alpha * T;

        % 打印进度
        if mod(iter, 50) == 0
            disp(['Iteration: ' num2str(iter) ', Current Best: ' num2str(best_value) ', T: ' num2str(T)]);
        end
    end
end


function neighbor = generate_neighbor(solution, lb, ub)
% generate_neighbor - 生成临近解，用于模拟退火随机扰动
% 
% 简单示例：在当前解附近添加小随机扰动，然后截断到上下界

    step_size = 0.1;  % 邻域扰动范围，可根据实际需求调整
    dim = length(solution);

    % 在 [-step_size, step_size] 范围内随机扰动
    disturbance = (rand(1, dim) - 0.5) * 2 * step_size;
    neighbor = solution + disturbance;

    % 保证结果在上下界之间
    neighbor = max(neighbor, lb);
    neighbor = min(neighbor, ub);
end
