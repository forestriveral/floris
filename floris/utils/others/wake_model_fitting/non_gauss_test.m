close all;
clear;
clc;

% SOWFA data path (Change to your own data path)
data_path = 'C:/Users/Li Hang/OneDrive - CivilEng/8_Desktop/CSSC Wake Simulation/SOWFA_data';

% wake parameters
D_rotor = 0.608;
velocity = [4, 6, 10, 15, 18];
thrust = [0.8803, 0.7721, 0.6589, 0.1830, 0.1076];
streamdist = [2, 3, 4, 5, 6, 7, 8, 9, 10];
r_D = ((0:0.0024:2.4023) - 1.2) / D_rotor;
ind = (1:20:1001);

% case parameters
vel = 4;   % velocity = 4, 6, 10, 15, 18
turb = 2;  % turbulence = 2, 6, 10
yaw = 0;   % yaw angle = 0, 10, 20, 30, -10, -20, -30
% case_name_4 = [4, 2, 0; 4, 6, 0; 4, 6, 10; 4, 6, 20; 4, 6, 30; 4, 10, 0];
% case_name_6 = [4, 2, 0; 4, 6, 0; 4, 6, 10; 4, 6, 20; 4, 6, 30; 4, 10, 0];


% vel_data = sowfa_load(vel, turb, yaw, distance)
if vel == 10
    vel = 10.4;
end
if yaw >= 0
    yaw_label = strcat('yaw', num2str(yaw));
else
    yaw_label = strcat('yaw(', num2str(yaw), ')');
end
path = strcat([data_path, '/', num2str(vel), 'ms/', num2str(vel), '-', num2str(turb), '%%-', yaw_label]);
% fprintf(['path: ', path, '\n']);
sowfa_data = zeros(length(streamdist), length(r_D));
for i = 1:1:length(streamdist)
    % fprintf(['Streamdist: x/D = ', num2str(streamdist(i)), '\n']);
    if vel == 4
        fname = sprintf('x=%dD.csv', streamdist(i));
    else
        fname = sprintf('x=%dD-z=0.38.csv', streamdist(i));
    end
    fname_path = strcat([path, '/', fname]);
    % fprintf(['fname_path: ', fname_path, '\n']);
    % fname_path = 'C:\Users\Li Hang\OneDrive - CivilEng\8_Desktop\CSSC Wake Simulation\SOWFA_data\4ms\4-2%-yaw0\x=2D.csv';
    data = importdata(strrep(fname_path, '%%', '%'), ',', 1);
    % disp(length(data.data(:, 5)));
    sowfa_data(i, :) = data.data(:, 5) / vel;
end
% disp(sowfa_data(2, :));


% Gauss_deficit = Gauss_velocity(vel, turb, x_D, r_D, p1, p2, p3)
p1 = 0.014;
p2 = 0.24;
p3 = 0.005;
thrust = thrust(velocity == vel);
k0 = p1 * thrust^1.07 * turb^0.20;
ep0 = p2 * thrust^-0.25 * turb^0.17;
a0 = 4 * thrust^-0.5 * ep0;
b0 = 4 * thrust^-0.5 * k0;
c0 = p3 * thrust^-0.25 * turb^-0.7;

Gauss_deficit = zeros(length(streamdist), length(r_D));
for i = 1:1:length(streamdist)
    x_D = streamdist(i);
    sigma_D = k0 * x_D + ep0;
    A = 1 / (a0 + b0 * x_D + c0 * (1 + x_D)^-2)^2;
    Gauss_deficit(i, :) = A * exp(- r_D.^2 / (2 * sigma_D^2));
end
% disp(Gauss_deficit(1, :));


% Non_Gauss_deficit = Non_Gauss_velocity(vel, turb, x_D, r_D, p4, p5, p6)
p1 = 0.11;
p2 = 0.23;
p3 = 0.15;
p4 = 0.472;
p5 = 0.07;
p6 = 0.254;
thrust = thrust(velocity == vel);
k = p1 * thrust^1.07 * turb^0.20;
ep = p2 * thrust^-0.25 * turb^0.17;
a = 4 * thrust^-0.5 * ep;
b = 4 * thrust^-0.5 * k;
c = p3 * thrust^-0.25 * turb^-0.7;

Non_Gauss_deficit = zeros(length(streamdist), length(r_D));
for i = 1:1:length(streamdist)
    x_D = streamdist(i);
    sigma_D = k * x_D + ep;
    A = 1 / (a + b * x_D + c * (1 + x_D)^-2)^2;
    B = 1 ./ (p4 * r_D.^2 +  p5 * r_D + p6);
    % fprintf(['A: ', num2str(A), '\n']);
    Non_Gauss_deficit(i, :) = A * B .* exp(- r_D.^2 / (2 * sigma_D^2));
end
% disp(Non_Gauss_deficit(1, :));


% wake parameters fitting
p4_range = (0.42: 0.01: 0.49);
p5_range = (0.07: 0.01: 0.13);
p6_range = (0.22: 0.01: 0.28);
error_0 = inf;
solution = zeros(1, 3);
for p4 = p4_range
    for p5 = p5_range
        for p6 = p6_range
            error_1 = 0.;
            for i = 1:1:length(streamdist)
                sowfa_velocity = 1 - sowfa_data(i, :) / max(sowfa_data(i, :));
                % Non-gaussian wake velocity calculation
                x_D = streamdist(i);
                sigma_D = k * x_D + ep;
                A = 1 / (a + b * x_D + c * (1 + x_D)^-2)^2;
                B = 1 ./ (p4 * r_D.^2 +  p5 * r_D + p6);
                gauss_velocity = A * B .* exp(- r_D.^2 / (2 * sigma_D^2));
                % Error accumulation
                error_1 = error_1 + sum(abs(gauss_velocity - sowfa_velocity).^2);
            end
            if error_1 < error_0
                solution = [p1, p2, p3];
                error_0 = error_1;
            end
        end
    end
end
fprintf('\nOptimal parameters: ');
disp(solution)
fprintf('Error: ');
disp(error_0)


% wake_plot
% fig_title = strcat(['Wake velocity fitting with ', 'U=', num2str(vel), 'm/s, ', ...
% 'I=', num2str(turb), '%%, ', 'Yaw=', num2str(yaw), ' deg']);
% fprintf([fig_title, '\n']);
% fig = figure('Units','centimeter','Position',[10 10 35 25]);
% % tsub = tight_subplot(3, 3, [.01 .03], [.1 .01], [.01 .01]);
% for i = 1:1:length(streamdist)
%     subplot(3, 3, i);
%     sowfa_vel = 1 - sowfa_data(i, :) / max(sowfa_data(i, :));
%     sofwa_handle = plot(r_D(ind), sowfa_vel(ind), 'linestyle', 'none', 'color', 'w', ...
%     'linewidth', 1., 'marker', 'o',  'markersize', 6, 'markerfacecolor', 'w', 'markeredgecolor', 'k');
%     hold on;
%     gauss_handle = plot(r_D, Gauss_deficit(i, :), 'linestyle', '-', 'color', 'k', ...
%     'linewidth', 1.5, 'marker', 'none');
%     hold on;
%     non_gauss_handle = plot(r_D, Non_Gauss_deficit(i, :), 'linestyle', '-', 'color', 'r', ...
%     'linewidth', 1.5, 'marker', 'none');
%     if i == 7 || i == 8 || i == 9
%         xlabel('y/D', 'fontsize', 16, 'fontname', 'Times New Roman', 'fontangle', 'italic');
%     end
%     if i == 1 || i == 4 || i == 7
%         ylabel('deficit', 'fontsize', 16, 'fontname', 'Times New Roman', 'fontangle', 'italic');
%     end
%     axis([-1.5, 1.5, 0., 1.])
%     subplot_title = strcat(['x/D = ', num2str(streamdist(i))]);
%     text(0.7, 0.9, subplot_title, 'units', 'normalized', 'fontsize', 16, 'color', 'k');
%     if i == 1
%         legend_handles = [sofwa_handle, gauss_handle, non_gauss_handle];
%         legend(legend_handles, 'SOWFA', 'Gauss', 'Non-Gauss', 'location', 'North', ...
%         'orientation', 'horizontal', 'fontsize', 18, 'fontname', 'Times New Roman', ...
%         'Position', [0.32, 0.91, 0.5, 0.1], 'box', 'off');
%     end
%     if i == 8
%         title(fig_title, 'fontsize', 18, 'fontname', 'Times New Roman', ...
%         'fontweight', 'bold', 'units', 'normalized', 'Position', [0.5, -0.45]);
%     end
% end

