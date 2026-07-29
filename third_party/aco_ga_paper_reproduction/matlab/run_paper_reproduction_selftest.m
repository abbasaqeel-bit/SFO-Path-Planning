function report = run_paper_reproduction_selftest(output_directory)
% Execute deterministic structural tests without the Python benchmark.
if nargin < 1
    output_directory = pwd;
end
if ~exist(output_directory, 'dir')
    mkdir(output_directory);
end
cases = build_cases();
report = struct([]);
for case_index = 1:length(cases)
    item = cases{case_index};
    rng(7000 + case_index, 'twister');
    [population, pheromone, aco_history] = paper_aco_population( ...
        item.map, item.start, item.goal, 30, 15, ...
        0.25, 2, 8, 100, {item.map}, 1, 0.5, 0.5, false);
    if isempty(population)
        error('SelfTest:NoPopulation', ...
            'ACO produced no population for %s.', item.name);
    end
    [~, evaluated, fitness, ga_history] = paper_ga_optimize( ...
        population, {item.map}, 1, 8, 0.2, 0.05, pheromone, ...
        0.5, 0.5, false);
    path = evaluated{1};
    [valid, reason, max_jump] = paper_validate_grid_path( ...
        item.map, path, item.start, item.goal, false);
    if ~valid
        error('SelfTest:InvalidPath', ...
            '%s failed validation: %s.', item.name, reason);
    end
    direct_length = norm(double(item.goal - item.start));
    actual_length = paper_path_length(path);
    if strcmp(item.name, 'open') && actual_length > 1.25 * direct_length
        error('SelfTest:OpenMapDetour', ...
            'Open-map path is unexpectedly long: %.6f vs %.6f.', ...
            actual_length, direct_length);
    end
    report(case_index).name = item.name; %#ok<AGROW>
    report(case_index).valid = valid;
    report(case_index).reason = reason;
    report(case_index).path_length = actual_length;
    report(case_index).direct_length = direct_length;
    report(case_index).max_jump = max_jump;
    report(case_index).fitness = fitness;
    report(case_index).aco_history = aco_history;
    report(case_index).ga_history = ga_history;
    report(case_index).path = path;
end
save(fullfile(output_directory, 'paper_reproduction_selftest.mat'), ...
    'report');
write_text_report( ...
    fullfile(output_directory, 'paper_reproduction_selftest.txt'), ...
    report);
end

function cases = build_cases()
open_map = ones(25, 25);
open_start = [2, 2];
open_goal = [24, 24];
open_map(open_start(1), open_start(2)) = 2;
open_map(open_goal(1), open_goal(2)) = 3;

obstacle_map = ones(25, 25);
obstacle_map(5:20, 12:14) = 0;
obstacle_map(11:14, 12:14) = 1;
obstacle_start = [2, 2];
obstacle_goal = [24, 24];
obstacle_map(obstacle_start(1), obstacle_start(2)) = 2;
obstacle_map(obstacle_goal(1), obstacle_goal(2)) = 3;

cases = {
    struct('name', 'open', 'map', open_map, ...
        'start', open_start, 'goal', open_goal), ...
    struct('name', 'obstacle', 'map', obstacle_map, ...
        'start', obstacle_start, 'goal', obstacle_goal)
};
end

function write_text_report(filename, report)
file = fopen(filename, 'w');
if file < 0
    error('SelfTest:ReportOpenFailed', 'Could not open self-test report.');
end
cleanup = onCleanup(@() fclose(file));
fprintf(file, 'ACO-GA paper reproduction self-test\n');
for index = 1:length(report)
    fprintf(file, ...
        '%s: valid=%d reason=%s length=%.6f direct=%.6f max_jump=%.6f fitness=%.12f\n', ...
        report(index).name, report(index).valid, report(index).reason, ...
        report(index).path_length, report(index).direct_length, ...
        report(index).max_jump, report(index).fitness);
end
clear cleanup;
end
