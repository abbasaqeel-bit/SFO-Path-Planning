function [valid, reason, max_jump] = paper_validate_grid_path( ...
        grid_map, path, start_node, goal_node, allow_corner_cutting)
if nargin < 5
    allow_corner_cutting = false;
end
valid = false;
reason = 'unknown';
max_jump = Inf;
if isempty(path) || size(path, 2) ~= 2
    reason = 'empty_or_malformed';
    return;
end
if ~isequal(path(1, :), start_node)
    reason = 'wrong_start';
    return;
end
if ~isequal(path(end, :), goal_node)
    reason = 'wrong_goal';
    return;
end
[row_count, column_count] = size(grid_map);
if any(path(:, 1) < 1 | path(:, 1) > row_count | ...
        path(:, 2) < 1 | path(:, 2) > column_count)
    reason = 'out_of_bounds';
    return;
end
indices = sub2ind(size(grid_map), path(:, 1), path(:, 2));
if any(grid_map(indices) == 0)
    reason = 'blocked_cell';
    return;
end
steps = diff(path, 1, 1);
if isempty(steps)
    reason = 'zero_length';
    return;
end
jumps = sqrt(sum(double(steps) .^ 2, 2));
max_jump = max(jumps);
if any(max(abs(steps), [], 2) > 1) || any(all(steps == 0, 2))
    reason = 'disconnected_step';
    return;
end
if ~allow_corner_cutting
    diagonal = find(all(abs(steps) == 1, 2));
    for item = diagonal'
        first = path(item, :);
        delta = steps(item, :);
        if grid_map(first(1) + delta(1), first(2)) == 0 || ...
                grid_map(first(1), first(2) + delta(2)) == 0
            reason = 'diagonal_corner_cut';
            return;
        end
    end
end
valid = true;
reason = 'ok';
end
