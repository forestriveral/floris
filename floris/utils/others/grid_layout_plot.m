close all;
clear;
clc;

% binary code matrix example
layout_size = 15;
binary_matrix = zeros(layout_size, layout_size);
binary_matrix([1, round(layout_size / 2), layout_size], :) = ones(3, layout_size);
% binary_matrix = rand(layout_size, layout_size);
% binary_matrix = round(binary_matrix);
% disp(binary_matrix);
% test_matrix = [1, 2, 3; 1, 1, 0; 0, 0, 1];
% grid_farm_plot();

% ==========================  Turbine coordinates ===================================
[w, h] = size(binary_matrix);
[y_index, x_index] = ind2sub(size(binary_matrix), find(binary_matrix == 1));
% disp(x_index);disp(y_index);
% disp([x_index, y_index])
turbine_coord = zeros(size([x_index, y_index]));
for i = 1:length(x_index)
    turbine_coord(i, :) = [x_index(i) * 1. + 0.5, h - y_index(i) + 1.5];
end
% disp(turbine_coord);

% ==========================  Figure Settings ========================================
fig_title = strcat('Wind farm grid layout');
fig = figure('Units','centimeter','Position',[10 10 15 15]);
% title(fig_title, 'fontsize', 15, 'fontname', 'Times New Roman', 'fontweight', 'bold');
% pcolor(binary_matrix);
% set(gcf,'Units','Inches');
% pos = get(gcf,'Position');
% set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

% ==========================  Axes Settings ==========================================
ax = gca;
axis([1., w + 1., 1., h + 1.]);
axis square;
hold on;

% ==========================  XY-axis Settings =======================================
set(gca, 'xtick', 1:1:w + 1, 'ytick', 1:1:h + 1);
set(gca, 'xticklabel', [], 'yticklabel', []);
% set(gca,'XTickLabel',{'a','b','c'})
% set(gca,'xtick',[])
% set(gca,'xTickMode','manual','XTick',[-21846,-10922,0,10922,21846]);
% a = get(gca,'xTickLabel');
% b = cell(size(a));
% b(mod(1:size(a,1),N)==1,:) = a(mod(1:size(a,1),N)==1,:);
% set(gca,'xTickLabel', b);

% set(gca,'XAxisLocation','bottom');
% set(gca,'XAxisLocation','top');
% set(gca,'XAxisLocation','origin');
% set(gca,'XDir','normal');
% set(gca,'XDir','reverse');

% ==========================  Grid Settings ==========================================
% grid on;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridColor = [0. 0. 0.];
% set(gca, 'GridLineStyle', ':');
ax.GridLineStyle = '-';
% set(gca, 'GridAlpha', 1);
ax.GridAlpha = 1.;
ax.LineWidth = 1.;
ax.Layer = 'top';
% set(gca,'ygrid','on','gridlinestyle','--','Gridalpha',0.4)
% set(gca, 'XMinorGrid','on');

% ==========================  Square Settings ==========================================
square_size = 23 / (min(w, h) / 10);
plot(turbine_coord(:, 1), turbine_coord(:, 2), 'linestyle', 'none', 'color', 'w', ...
'linewidth', 1., 'marker', 'square',  'markersize', square_size, ...
'markerfacecolor', 'k', 'markeredgecolor', 'k');


% ==========================  Output Settings ==========================================
% saveas(gcf, 'grid_layout.png');
% print(gcf,'filename','-dpdf','-r0');
% exportgraphics(ax,'BarChart.pdf','ContentType','vector')