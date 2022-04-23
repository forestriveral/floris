function grid_farm_plot(binary_matrix)
    % simple matlab function to plot a grid farm layout from a binary matrix.
    % the input is a binary matrix that only includes 0 or 1.
    % if no binary matrix is given, generate a matrix example with default size of 15¡Á15.
    if (~exist('binary_matrix','var'))
        layout_size = 15;
        binary_matrix = zeros(layout_size, layout_size);
        binary_matrix([1, round(layout_size / 2), layout_size], :) = ones(3, layout_size);
    else
        assert(isnumeric(binary_matrix), 'Provided binary matrix must be numeric !');
        assert(all(size(binary_matrix) > 1, 'all') || length(size(binary_matrix)) < 3, ...
        'The size of every dimension must be no less than Two and the dimension number must less than Three !');
        binary_matrix_copy = binary_matrix; binary_matrix_copy(binary_matrix_copy == 0) = 1;
        assert(all(binary_matrix_copy == 1, 'all'), ...
        'Provided binary matrix must only include 0 and 1 !');
    end
    [w, h] = size(binary_matrix);
    [y_index, x_index] = ind2sub(size(binary_matrix), find(binary_matrix == 1));
    turbine_coord = zeros(size([x_index, y_index]));
    for i = 1:length(x_index)
        turbine_coord(i, :) = [x_index(i) * 1. + 0.5, h - y_index(i) + 1.5];
    end
    fig_title = strcat('Wind farm grid layout');
    figure('Units','centimeter','Position',[10 10 15 15]);
    title(fig_title, 'fontsize', 15, 'fontname', 'Times New Roman', 'fontweight', 'bold');
    axis([1., w + 1., 1., h + 1.]);
    axis square;
    hold on;
    set(gca, 'xtick', 1:1:w + 1, 'ytick', 1:1:h + 1);
    set(gca, 'xticklabel', [], 'yticklabel', []);
    set(gca, 'xgrid', 'on', 'ygrid', 'on');
    set(gca, 'gridcolor', 'k', 'gridlinestyle', '-');
    set(gca, 'gridAlpha', 1, 'linewidth', 1, 'layer', 'top');
    plot(turbine_coord(:, 1), turbine_coord(:, 2), 'linestyle', 'none', 'color', 'w', ...
    'linewidth', 1., 'marker', 'square',  'markersize', 23 / (min(w, h) / 10), ...
    'markerfacecolor', 'k', 'markeredgecolor', 'k');
    % saveas(gcf, 'grid_layout.png', 'png');
end