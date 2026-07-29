function adjusted = paper_avoid_obstacles( ...
        grid_map, initial_path, allow_corner_cutting)
% Preserve free portions and replace each blocked run by a BFS detour.
%
% The published Algorithm 1 expands a rectangular obstacle region and uses
% BFS to choose the shorter of its two boundary detours. The paper does not
% publish the low-level rectangle/edge routines. This implementation keeps
% the same observable contract: it preserves the initial route outside an
% encountered obstacle and inserts the shortest feasible BFS detour.
if nargin < 3
    allow_corner_cutting = false;
end
if isempty(initial_path)
    adjusted = [];
    return;
end
adjusted = initial_path(1, :);
index = 2;
while index <= size(initial_path, 1)
    target = initial_path(index, :);
    if grid_map(target(1), target(2)) ~= 0
        connector = paper_shortest_connector( ...
            grid_map, adjusted(end, :), target, allow_corner_cutting);
        if isempty(connector)
            adjusted = [];
            return;
        end
        adjusted = [adjusted; connector(2:end, :)]; %#ok<AGROW>
        index = index + 1;
        continue;
    end
    next_free = index + 1;
    while next_free <= size(initial_path, 1) && ...
            grid_map(initial_path(next_free, 1), ...
                     initial_path(next_free, 2)) == 0
        next_free = next_free + 1;
    end
    if next_free > size(initial_path, 1)
        adjusted = [];
        return;
    end
    connector = paper_shortest_connector( ...
        grid_map, adjusted(end, :), initial_path(next_free, :), ...
        allow_corner_cutting);
    if isempty(connector)
        adjusted = [];
        return;
    end
    adjusted = [adjusted; connector(2:end, :)]; %#ok<AGROW>
    index = next_free + 1;
end
% Canonical loop erasure retains connectivity and removes only cycles.
position = 1;
while position <= size(adjusted, 1)
    later = find( ...
        adjusted(position + 1:end, 1) == adjusted(position, 1) & ...
        adjusted(position + 1:end, 2) == adjusted(position, 2), ...
        1, 'last');
    if isempty(later)
        position = position + 1;
    else
        later = later + position;
        adjusted(position + 1:later, :) = [];
    end
end
end
