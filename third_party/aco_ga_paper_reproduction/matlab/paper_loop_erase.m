function path = paper_loop_erase(path)
% Remove cycles without changing the order or introducing new edges.
position = 1;
while position <= size(path, 1)
    later = find( ...
        path(position + 1:end, 1) == path(position, 1) & ...
        path(position + 1:end, 2) == path(position, 2), ...
        1, 'last');
    if isempty(later)
        position = position + 1;
    else
        later = later + position;
        path(position + 1:later, :) = [];
    end
end
end
