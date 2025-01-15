% 主脚本
clear;
clc;

% 真实参数
true_params = [1.0, 2.0, 3.0, 4.0, 5.0];

% 生成模拟数据
num_samples = 10000;
noise_std = 0.1;
[x_samples, y_samples_noisy] = generate_data(true_params, num_samples, noise_std);

% 初始参数猜测
initial_params = [0.5, 1.5, 2.5, 3.5, 4.5];

% 使用最小二乘法进行优化
disp('Running Least Squares Optimization...');
options = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', 1e-8);
optimized_params_lsq = lsqnonlin(@(p) objective_function(p, x_samples, y_samples_noisy), initial_params, [], [], options);

disp('True Parameters:');
disp(true_params);
disp('Optimized Parameters (Least Squares):');
disp(optimized_params_lsq);

% 模拟退火优化参数
initial_temp = 1000;
final_temp = 1e-8;
alpha = 0.99;
max_iter = 100;

disp('Running Simulated Annealing...');
[best_params_sa, best_value_sa] = simulated_annealing(@multi_param_function, initial_params, initial_temp, final_temp, alpha, max_iter, x_samples, y_samples_noisy);

disp('Best Parameters (Simulated Annealing):');
disp(best_params_sa);
disp('Best Value (Simulated Annealing):');
disp(best_value_sa);

% 粒子群优化
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
y_fitted_lsq = arrayfun(@(i) multi_param_function(x_samples(i, :), optimized_params_lsq), 1:num_samples);
y_fitted_sa = arrayfun(@(i) multi_param_function(x_samples(i, :), best_params_sa), 1:num_samples);
y_fitted_pso = arrayfun(@(i) multi_param_function(x_samples(i, :), best_params_pso), 1:num_samples);

figure;
scatter(1:num_samples, y_samples_noisy, 'b', 'DisplayName', 'Noisy Data');
hold on;
plot(1:num_samples, y_fitted_lsq, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Least Squares Fit');
plot(1:num_samples, y_fitted_sa, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Simulated Annealing Fit');
plot(1:num_samples, y_fitted_pso, 'm-.', 'LineWidth', 1.5, 'DisplayName', 'PSO Fit');
legend;
xlabel('Sample Index');
ylabel('Function Value');
title('Data Fitting Comparison');
grid on;

%% 函数定义部分

% 定义多参数函数
function y = multi_param_function(x, params)
    a1 = params(1);
    a2 = params(2);
    a3 = params(3);
    a4 = params(4);
    a5 = params(5);
    y = a1 * sin(x(1)) + a2 * cos(x(2)) + a3 * x(3)^2 + a4 * x(4) + a5;
end

% 生成模拟数据
function [x_samples, y_samples_noisy] = generate_data(params, num_samples, noise_std)
    x_samples = -10 + 20 * rand(num_samples, 4); % 每行表示一个样本
    y_samples = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:num_samples).';
    noise = noise_std * randn(num_samples, 1);
    y_samples_noisy = y_samples + noise;
end

% 目标函数（误差平方和，用于最小化）
function F = objective_function(params, x_samples, y_samples)
    predicted = arrayfun(@(i) multi_param_function(x_samples(i, :), params), 1:size(x_samples, 1)).';
    F = predicted - y_samples;
end

% 模拟退火算法
function [best_params, best_value] = simulated_annealing(func, initial_params, initial_temp, final_temp, alpha, max_iter, x_samples, y_samples)
    current_params = initial_params;
    current_value = sum((objective_function(current_params, x_samples, y_samples)).^2);
    best_params = current_params;
    best_value = current_value;
    temperature = initial_temp;

    while temperature > final_temp
        for i = 1:max_iter
            new_params = current_params + (rand(size(current_params)) - 0.5) * 0.2;
            new_value = sum((objective_function(new_params, x_samples, y_samples)).^2);

            if new_value < current_value || rand < exp((current_value - new_value) / temperature)
                current_params = new_params;
                current_value = new_value;

                if current_value < best_value
                    best_params = current_params;
                    best_value = current_value;
                end
            end
        end
        temperature = temperature * alpha;
    end
end
