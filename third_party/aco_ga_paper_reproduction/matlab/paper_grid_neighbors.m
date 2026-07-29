function neighbors = paper_grid_neighbors(grid_map, node, allow_corner_cutting)
% Return free 8-neighbours. Corner policy is an exposed reproduction choice.
if nargin < 3
    allow_corner_cutting = false;
end
[row_count, column_count] = size(grid_map);
directions = [
    -1, -1; -1, 0; -1, 1;
     0, -1;          0, 1;
     1, -1;  1, 0;  1, 1
];
neighbors = zeros(0, 2);
for index = 1:size(directions, 1)
    candidate = node + directions(index, :);
    row = candidate(1);
    column = candidate(2);
    if row < 1 || row > row_count || column < 1 || column > column_count
        continue;
    end
    if grid_map(row, column) == 0
        continue;
    end
    if ~allow_corner_cutting && all(abs(directions(index, :)) == 1)
        side_a = [node(1) + directions(index, 1), node(2)];
        side_b = [node(1), node(2) + directions(index, 2)];
        if grid_map(side_a(1), side_a(2)) == 0 || ...
                grid_map(side_b(1), side_b(2)) == 0
            continue;
        end
    end
    neighbors(end + 1, :) = candidate; %#ok<AGROW>
end
end
