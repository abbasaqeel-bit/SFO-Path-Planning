function [fitness, evaluated_paths] = paper_multimap_fitness( ...
        path, maps, map_weights, length_weight, smoothness_weight, ...
        allow_corner_cutting)
% Equations (4) and (5): maximize weighted inverse length + smoothness.
map_count = length(maps);
fitness_components = zeros(1, map_count);
evaluated_paths = cell(1, map_count);
for map_index = 1:map_count
    candidate = paper_avoid_obstacles( ...
        maps{map_index}, path, allow_corner_cutting);
    evaluated_paths{map_index} = candidate;
    if isempty(candidate)
        fitness = 0;
        return;
    end
    length_value = paper_path_length(candidate);
    smoothness_value = paper_path_smoothness(candidate);
    fitness_components(map_index) = ...
        length_weight / (length_value + eps) + ...
        smoothness_weight * smoothness_value;
end
weights = double(map_weights(:)');
if length(weights) ~= map_count || any(weights < 0) || sum(weights) <= 0
    error('paper_multimap_fitness:InvalidMapWeights', ...
        'map_weights must contain one non-negative value per map.');
end
weights = weights / sum(weights);
fitness = sum(weights .* fitness_components);
end
