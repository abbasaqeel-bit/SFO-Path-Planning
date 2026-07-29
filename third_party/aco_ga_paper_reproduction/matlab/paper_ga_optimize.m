function [best_path, best_evaluated_paths, best_fitness, history] = ...
        paper_ga_optimize( ...
            initial_population, maps, map_weights, generation_count, ...
            crossover_probability, mutation_probability, pheromone, ...
            length_weight, smoothness_weight, allow_corner_cutting)
% GA stage described in Sections 2.3 and 4.3 of the paper.
population = initial_population;
population_size = length(population);
best_path = [];
best_evaluated_paths = {};
best_fitness = -Inf;
history = zeros(generation_count, 1);
reference_map = maps{select_reference_map(maps)};

for generation = 1:generation_count
    [fitness, evaluated] = evaluate_population( ...
        population, maps, map_weights, length_weight, ...
        smoothness_weight, allow_corner_cutting);
    [generation_best, best_index] = max(fitness);
    if generation_best > best_fitness
        best_fitness = generation_best;
        best_path = population{best_index};
        best_evaluated_paths = evaluated{best_index};
    end
    history(generation) = best_fitness;

    selected_indices = roulette_selection(fitness, population_size);
    offspring = cell(1, population_size);
    for index = 1:2:population_size
        parent_a = population{selected_indices(index)};
        parent_b = population{selected_indices(min(index + 1, population_size))};
        child_a = parent_a;
        child_b = parent_b;
        if rand < crossover_probability
            [child_a, child_b] = common_node_crossover(parent_a, parent_b);
        end
        if rand < mutation_probability
            child_a = subpath_mutation( ...
                child_a, reference_map, pheromone, ...
                allow_corner_cutting);
        end
        if rand < mutation_probability
            child_b = subpath_mutation( ...
                child_b, reference_map, pheromone, ...
                allow_corner_cutting);
        end
        child_a = paper_loop_erase(child_a);
        child_b = paper_loop_erase(child_b);
        if is_valid_candidate( ...
                reference_map, child_a, parent_a(1, :), ...
                parent_a(end, :), allow_corner_cutting)
            offspring{index} = child_a;
        else
            offspring{index} = parent_a;
        end
        if index + 1 <= population_size
            if is_valid_candidate( ...
                    reference_map, child_b, parent_b(1, :), ...
                    parent_b(end, :), allow_corner_cutting)
                offspring{index + 1} = child_b;
            else
                offspring{index + 1} = parent_b;
            end
        end
    end
    population = offspring;
end

% Evaluate the final population too; otherwise generation_count=1 would
% never evaluate the children made by that generation.
[fitness, evaluated] = evaluate_population( ...
    population, maps, map_weights, length_weight, smoothness_weight, ...
    allow_corner_cutting);
[final_best, best_index] = max(fitness);
if final_best > best_fitness
    best_fitness = final_best;
    best_path = population{best_index};
    best_evaluated_paths = evaluated{best_index};
end
if ~isempty(history)
    history(end) = best_fitness;
end
end

function [fitness, evaluated] = evaluate_population( ...
        population, maps, map_weights, length_weight, ...
        smoothness_weight, allow_corner_cutting)
fitness = zeros(1, length(population));
evaluated = cell(1, length(population));
for index = 1:length(population)
    [fitness(index), evaluated{index}] = paper_multimap_fitness( ...
        population{index}, maps, map_weights, length_weight, ...
        smoothness_weight, allow_corner_cutting);
end
end

function indices = roulette_selection(fitness, count)
weights = max(0, double(fitness));
if sum(weights) <= 0 || any(~isfinite(weights))
    indices = randi(count, 1, count);
    return;
end
cumulative = cumsum(weights / sum(weights));
indices = zeros(1, count);
for index = 1:count
    chosen = find(cumulative >= rand, 1, 'first');
    if isempty(chosen)
        chosen = count;
    end
    indices(index) = chosen;
end
end

function [child_a, child_b] = common_node_crossover(parent_a, parent_b)
% Exchange only overlapping path sections. Arbitrary prefix/suffix cuts
% produce the disconnected multi-line artifact seen in the old bridge.
child_a = parent_a;
child_b = parent_b;
if size(parent_a, 1) < 3 || size(parent_b, 1) < 3
    return;
end
[common, index_a, index_b] = intersect( ...
    parent_a(2:end - 1, :), parent_b(2:end - 1, :), ...
    'rows', 'stable');
if isempty(common)
    return;
end
selected = randi(size(common, 1));
cut_a = index_a(selected) + 1;
cut_b = index_b(selected) + 1;
child_a = [parent_a(1:cut_a, :); parent_b(cut_b + 1:end, :)];
child_b = [parent_b(1:cut_b, :); parent_a(cut_a + 1:end, :)];
end

function mutated = subpath_mutation( ...
        path, reference_map, pheromone, allow_corner_cutting)
% Select two path nodes and insert a newly generated feasible subpath.
mutated = path;
if size(path, 1) < 5
    return;
end
first = randi([1, size(path, 1) - 2]);
second = randi([first + 2, size(path, 1)]);
connector = pheromone_guided_connector( ...
    reference_map, path(first, :), path(second, :), pheromone, ...
    allow_corner_cutting);
if isempty(connector)
    return;
end
mutated = [
    path(1:first - 1, :);
    connector;
    path(second + 1:end, :)
];
end

function path = pheromone_guided_connector( ...
        grid_map, start_node, goal_node, pheromone, allow_corner_cutting)
% The paper's improved mutation uses ACO pheromone. Search is bounded and
% returns only an actual connected path.
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
    neighbors = neighbors(keep, :);
    if isempty(neighbors)
        path = [];
        return;
    end
    scores = zeros(size(neighbors, 1), 1);
    for index = 1:size(neighbors, 1)
        node = neighbors(index, :);
        distance = norm(double(node - goal_node));
        scores(index) = max(pheromone(node(1), node(2)), eps) * ...
            (1 / (distance + eps)) ^ 2;
    end
    probabilities = scores / sum(scores);
    selected = find(cumsum(probabilities) >= rand, 1, 'first');
    current = neighbors(selected, :);
    path(end + 1, :) = current; %#ok<AGROW>
    visited(current(1), current(2)) = true;
end
path = [];
end

function valid = is_valid_candidate( ...
        grid_map, path, start_node, goal_node, allow_corner_cutting)
[valid, ~, ~] = paper_validate_grid_path( ...
    grid_map, path, start_node, goal_node, allow_corner_cutting);
end

function index = select_reference_map(maps)
counts = zeros(1, length(maps));
for map_index = 1:length(maps)
    counts(map_index) = nnz(maps{map_index} == 0);
end
[~, index] = max(counts);
end
