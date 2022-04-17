close all;
clear;
clc;

data_path = 'C:/Users/Li Hang/OneDrive - CivilEng/8_Desktop/CSSC Wake Simulation/SOWFA_data/';

D_rotor = 0.608;

vel = [4, 6, 10, 15, 18];
C_t = [0.8803, 0.7721, 0.6589, 0.1830, 0.1076];

% dist_font = {'family': 'Times New Roman',
%              'weight': 'normal',
%              'style': 'italic',
%              'size': 18,
%              'color': 'k', }

% label_font = {'family': 'Times New Roman',
%               'size': 18}

% legend_font = {'family': 'Times New Roman',
%                'size': 15}


function non_gauss_test(vel, turb, yaw)
    distance = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    sowfa_data = sowfa_load(vel, turb, yaw, distance);
    r_D = ((0:0.0024:2.4023) - 1.2) / D_rotor;
    title = strcat('Wake velocity fitting with ', sprintf('U=%dm/s, I=%d%%, Yaw=%d', vel, turb, yaw));
    % subplot(3, 3, sharey=True, figsize=(14, 12), dpi=100);

    % for i = 1:length(distance)
    %     subplot(3, 3, i);
    %     ind = linspace(0, length(r_D), 50);
    %     sowfa_vel = 1 - sowfa_data[i] / np.max(sowfa_data[i]);
    %     axi.plot(r_D[ind], sowfa_vel[ind], c="w", lw=0., label='SOWFA LES',
    %              markersize=6, marker="o", markeredgecolor='k', markeredgewidth=1.,);
    %     nongauss_vel = Non_Gauss_velocity(vel, turb, dist, r_D, 0.472, 0.07, 0.254);
    %     axi.plot(r_D, nongauss_vel, c='r', linestyle='-', lw=1.5, label="Non-gaussian",);
    %     gauss_vel = Gauss_velocity(vel, turb, dist, r_D, 0.014, 0.24, 0.005);
    %     axi.plot(r_D, gauss_vel, c='k', linestyle='--', lw=1.5, label="Ishihara-Qian",);
    %     if i in [6, 7, 8]:
    %         axi.set_xlabel('y/D', fontdict=dist_font);
    %     axi.set_xlim([-1.5, 1.5]);
    %     axi.set_xticks([-1.5, -1., -0.5, 0., 0.5, 1., 1.5]);
    %     axi.set_xticklabels(['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5'])
    %     if i in [0, 3, 6]:
    %         axi.set_ylabel('deficit', fontdict=dist_font, labelpad=5);
    %     axi.set_ylim([0., 1.]);
    %     axi.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.]);
    %     axi.set_yticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0']);
    %     axi.text(0.95, 0.9, f'x/d = {dist}', va='top', ha='right',
    %              fontdict=dist_font, transform=axi.transAxes, );
    %     labels = axi.get_xticklabels() + axi.get_yticklabels();
    %     [label.set_fontname('Times New Roman') for label in labels];
    %     axi.tick_params(labelsize=15, colors='k', direction='in',
    %                     top=True, bottom=True, left=True, right=True);
    % ax1 = ax.flatten()[1];
    % handles, labels = ax1.get_legend_handles_labels();
    % ax1.legend(handles, labels, loc="upper left", prop=legend_font, columnspacing=0.8,
    %            edgecolor='None', frameon=False, labelspacing=0.5, bbox_to_anchor=(-0.20, 1.2),
    %            bbox_transform=ax1.transAxes, ncol=len(labels), handletextpad=0.5);
    % fig.suptitle(title, position=(0.5, 0.02), va='bottom', ha='center', fontproperties=label_font);
    % plt.subplots_adjust(wspace=0.15, hspace=0.15);

    % plt.show();
end


function vel_data = sowfa_load(vel, turb, yaw, distance)
    if vel == 10
        vel = 10.4;
    else
        vel = vel;
    end

    if yaw >= 0
        yaw = strcat('yaw', num2str(yaw));
    else
        yaw = strcat('yaw(', num2str(yaw), ')');
    end

    path = strcat(sprintf('%s/%dms/%d-%d', data_path, vel, vel, turb, yaw), sprintf('%-%d', yaw));
    vel_data = []
    for i = 1:length(distance)
        if int(vel) == 4
            fname = sprintf('x=%dD.csv', distance(i));
        else
            fname = sprintf('x=%dD-z=0.38.csv', distance(i));
        end
        data = csvread(path + '/' + fname, 1, 4, [1, 4, 1002, 5]);
        vel_data = [vel_data, data / vel];
    end
end


function deficit = Gauss_velocity(vel, turb, x_D, r_D, p1, p2, p3)
    thrust = C_t(find(vel == 6));
    % p1, p2, p3 = 0.11, 0.23, 0.15
    k = p1 * thrust^1.07 * turb^0.20;
    ep = p2 * thrust^-0.25 * turb^0.17;
    a = 4 * thrust^-0.5 * ep;
    b = 4 * thrust^-0.5 * k;
    c = p3 * thrust^-0.25 * turb^-0.7;
    sigma_D = k * x_D + ep;
    A = 1. / (a + b * x_D + c * (1 + x_D)^-2)^2;
    deficit = A * np.exp(- r_D^2 / (2 * sigma_D^2));
end


function deficit = Non_Gauss_velocity(vel, turb, x_D, r_D, p4, p5, p6)
    thrust = C_t(find(vel == 6));
    p1, p2, p3 = 0.11, 0.23, 0.15;
    k = p1 * thrust^1.07 * turb^0.20;
    ep = p2 * thrust^-0.25 * turb^0.17;
    a = 4 * thrust^-0.5 * ep;
    b = 4 * thrust^-0.5 * k;
    c = p3 * thrust^-0.25 * turb^-0.7;
    sigma_D = k * x_D + ep;
    A = 1. / (a + b * x_D + c * (1 + x_D)^-2)^2;
    deficit = A / (p4 * r_D^2 + p5 * r_D + p6) * exp(- r_D^2 / (2 * sigma_D^2));
end