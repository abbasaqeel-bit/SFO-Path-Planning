function value = paper_path_smoothness(path)
% Mean internal angle, normalized to [0,1]. Straight continuation is 1.
if size(path, 1) < 3
    value = 1;
    return;
end
angles = zeros(size(path, 1) - 2, 1);
for index = 2:size(path, 1) - 1
    incoming = double(path(index - 1, :) - path(index, :));
    outgoing = double(path(index + 1, :) - path(index, :));
    denominator = norm(incoming) * norm(outgoing);
    if denominator <= eps
        angles(index - 1) = 0;
    else
        cosine = dot(incoming, outgoing) / denominator;
        cosine = min(1, max(-1, cosine));
        angles(index - 1) = acos(cosine) / pi;
    end
end
value = mean(angles);
end
