function value = paper_path_length(path)
if isempty(path) || size(path, 1) < 2
    value = Inf;
    return;
end
steps = diff(double(path), 1, 1);
value = sum(sqrt(sum(steps .^ 2, 2)));
end
