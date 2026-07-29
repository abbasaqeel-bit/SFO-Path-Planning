function path = paper_shortest_connector( ...
        grid_map, start_node, goal_node, allow_corner_cutting)
% Breadth-first connector used by the paper's obstacle-avoidance procedure.
if nargin < 4
    allow_corner_cutting = false;
end
if isequal(start_node, goal_node)
    path = start_node;
    return;
end
[row_count, column_count] = size(grid_map);
visited = false(row_count, column_count);
parent_row = zeros(row_count, column_count);
parent_column = zeros(row_count, column_count);
queue = zeros(row_count * column_count, 2);
head = 1;
tail = 1;
queue(tail, :) = start_node;
visited(start_node(1), start_node(2)) = true;
found = false;
while head <= tail
    current = queue(head, :);
    head = head + 1;
    neighbors = paper_grid_neighbors( ...
        grid_map, current, allow_corner_cutting);
    % Goal-directed deterministic tie breaking.
    if ~isempty(neighbors)
        distances = sum((double(neighbors) - double(goal_node)) .^ 2, 2);
        [~, order] = sort(distances, 'ascend');
        neighbors = neighbors(order, :);
    end
    for index = 1:size(neighbors, 1)
        candidate = neighbors(index, :);
        if visited(candidate(1), candidate(2))
            continue;
        end
        visited(candidate(1), candidate(2)) = true;
        parent_row(candidate(1), candidate(2)) = current(1);
        parent_column(candidate(1), candidate(2)) = current(2);
        tail = tail + 1;
        queue(tail, :) = candidate;
        if isequal(candidate, goal_node)
            found = true;
            break;
        end
    end
    if found
        break;
    end
end
if ~found
    path = [];
    return;
end
path = goal_node;
current = goal_node;
while ~isequal(current, start_node)
    current = [
        parent_row(current(1), current(2)), ...
        parent_column(current(1), current(2))
    ];
    if any(current == 0)
        path = [];
        return;
    end
    path = [current; path]; %#ok<AGROW>
end
end
