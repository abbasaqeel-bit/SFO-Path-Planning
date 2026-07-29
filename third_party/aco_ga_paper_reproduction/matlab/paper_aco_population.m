function [population, pheromone, history] = paper_aco_population( ...
        reference_map, start_node, goal_node, ant_count, iteration_count, ...
        rho, alpha, beta, Q, maps, map_weights, length_weight, ...
        smoothness_weight, allow_corner_cutting)
% Improved ACO in Section 4.1 of the published paper.
[row_count, column_count] = size(reference_map);
pheromone = zeros(row_count, column_count);
for row = 1:row_count
    for column = 1:column_count
        if reference_map(row, column) ~= 0
            distance = norm(double([row, column] - goal_node));
            pheromone(row, column) = 1 / (distance + eps);
        end
    end
end
history = zeros(iteration_count, 1);
archive_paths = {};
archive_fitness = [];
last_paths = {};
last_fitness = [];
for iteration = 1:iteration_count
    iteration_paths = {};
    iteration_lengths = [];
    iteration_fitness = [];
    for ant = 1:ant_count
        path = construct_ant_path( ...
            reference_map, start_node, goal_node, pheromone, ...
            alpha, beta, allow_corner_cutting);
        if isempty(path)
            continue;
        end
        [fitness, ~] = paper_multimap_fitness( ...
            path, maps, map_weights, length_weight, smoothness_weight, ...
            allow_corner_cutting);
        if fitness <= 0 || ~isfinite(fitness)
            continue;
        end
        iteration_paths{end + 1} = path; %#ok<AGROW>
        iteration_lengths(end + 1) = paper_path_length(path); %#ok<AGROW>
        iteration_fitness(end + 1) = fitness; %#ok<AGROW>
    end
    deposit = zeros(row_count, column_count);
    for item = 1:length(iteration_paths)
        path = iteration_paths{item};
        path_length = iteration_lengths(item);
        for node_index = 1:size(path, 1) - 1
            node = path(node_index, :);
            sigma = safety_factor(node, maps, map_weights);
            deposit(node(1), node(2)) = ...
                deposit(node(1), node(2)) + ...
                (1 - sigma) * Q / (path_length + eps);
        end
    end
    pheromone = (1 - rho) * pheromone + deposit;
    pheromone(reference_map == 0) = 0;
    if ~isempty(iteration_paths)
        last_paths = iteration_paths;
        last_fitness = iteration_fitness;
        archive_paths = [archive_paths, iteration_paths]; %#ok<AGROW>
        archive_fitness = [archive_fitness, iteration_fitness]; %#ok<AGROW>
    end
    if isempty(archive_fitness)
        history(iteration) = 0;
    else
        history(iteration) = max(archive_fitness);
    end
end
% The paper passes the high-quality path population from the final ACO
% iteration to GA. If the stochastic final iteration has fewer than m valid
% paths, the best earlier valid paths fill only the missing slots.
population = select_best_unique(last_paths, last_fitness, ant_count);
if length(population) < ant_count
    fallback = select_best_unique(archive_paths, archive_fitness, ant_count);
    for index = 1:length(fallback)
        if length(population) >= ant_count
            break;
        end
        population{end + 1} = fallback{index}; %#ok<AGROW>
    end
end
if ~isempty(population)
    cursor = 1;
    while length(population) < ant_count
        population{end + 1} = population{cursor}; %#ok<AGROW>
        cursor = cursor + 1;
        if cursor > length(population)
            cursor = 1;
        end
    end
end
end

function path = construct_ant_path( ...
        grid_map, start_node, goal_node, pheromone, alpha, beta, ...
        allow_corner_cutting)
[row_count, column_count] = size(grid_map);
visited = false(row_count, column_count);
path = start_node;
current = start_node;
visited(current(1), current(2)) = true;
for step = 1:row_count * column_count
    if isequal(current, goal_node)
        return;
    end
    neighbors = paper_grid_neighbors( ...
        grid_map, current, allow_corner_cutting);
    keep = false(size(neighbors, 1), 1);
    for index = 1:size(neighbors, 1)
        keep(index) = ~visited(neighbors(index, 1), neighbors(index, 2));
    end
    allowed = neighbors(keep, :);
    if isempty(allowed)
        path = [];
        return;
    end
    desirability = zeros(size(allowed, 1), 1);
    for index = 1:size(allowed, 1)
        node = allowed(index, :);
        distance = norm(double(node - goal_node));
        desirability(index) = ...
            pheromone(node(1), node(2)) ^ alpha * ...
            (1 / (distance + eps)) ^ beta;
    end
    if sum(desirability) <= 0 || any(~isfinite(desirability))
        selected = randi(size(allowed, 1));
    else
        probabilities = desirability / sum(desirability);
        selected = find(cumsum(probabilities) >= rand, 1, 'first');
    end
    current = allowed(selected, :);
    path(end + 1, :) = current; %#ok<AGROW>
    visited(current(1), current(2)) = true;
end
path = [];
end

function sigma = safety_factor(node, maps, map_weights)
weights = double(map_weights(:)');
weights = weights / sum(weights);
sigma = 0;
for map_index = 1:length(maps)
    grid_map = maps{map_index};
    obstacle_count = 0;
    neighbor_count = 0;
    for row_delta = -1:1
        for column_delta = -1:1
            if row_delta == 0 && column_delta == 0
                continue;
            end
            row = node(1) + row_delta;
            column = node(2) + column_delta;
            if row < 1 || row > size(grid_map, 1) || ...
                    column < 1 || column > size(grid_map, 2)
                continue;
            end
            neighbor_count = neighbor_count + 1;
            obstacle_count = obstacle_count + (grid_map(row, column) == 0);
        end
    end
    if neighbor_count > 0
        sigma = sigma + weights(map_index) * ...
            obstacle_count / neighbor_count;
    end
end
sigma = min(1, max(0, sigma));
end

function selected = select_best_unique(paths, fitness, count)
selected = {};
if isempty(paths)
    return;
end
[~, order] = sort(fitness, 'descend');
signatures = {};
for item = order
    candidate = paths{item};
    signature = sprintf('%d,%d;', candidate');
    if any(strcmp(signatures, signature))
        continue;
    end
    selected{end + 1} = candidate; %#ok<AGROW>
    signatures{end + 1} = signature; %#ok<AGROW>
    if length(selected) >= count
        break;
    end
end
end
