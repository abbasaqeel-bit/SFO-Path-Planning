function run_aco_ga_paper_bridge(input_file, output_file)
best_path = [];
best_fitness = -Inf;
fitness_history = [];
aco_fitness_history = [];
ga_fitness_history = [];
success = 0;
failure_stage = 'bridge_initialization';
validation_reason = 'not_run';
max_grid_jump = Inf;
initial_population_count = 0;
reference_map_index = 0;
implementation_status = 'paper_faithful_independent_reproduction';
objective_scope = 'native_published_multimap_objective';
t0 = tic;

try
    input = load(input_file);
    rng(double(input.seed), 'twister');
    maps = cell(1, size(input.maps_stack, 3));
    for map_index = 1:size(input.maps_stack, 3)
        maps{map_index} = double(input.maps_stack(:, :, map_index));
    end
    obstacle_counts = zeros(1, length(maps));
    for map_index = 1:length(maps)
        obstacle_counts(map_index) = nnz(maps{map_index} == 0);
    end
    [~, reference_map_index] = max(obstacle_counts);
    reference_map = maps{reference_map_index};
    start_node = double(input.start_rc(:)');
    goal_node = double(input.goal_rc(:)');
    map_weights = double(input.map_weights(:)');
    allow_corner_cutting = logical(input.allow_corner_cutting);

    failure_stage = 'aco_initialization';
    [population, pheromone, aco_fitness_history] = ...
        paper_aco_population( ...
            reference_map, start_node, goal_node, ...
            double(input.population_size), ...
            double(input.aco_iterations), double(input.rho), ...
            double(input.alpha), double(input.beta), double(input.Q), ...
            maps, map_weights, double(input.length_weight), ...
            double(input.smoothness_weight), allow_corner_cutting);
    initial_population_count = length(population);
    if isempty(population)
        validation_reason = 'aco_returned_no_initial_population';
    else
        failure_stage = 'ga_optimization';
        [native_path, evaluated_paths, best_fitness, ga_fitness_history] = ...
            paper_ga_optimize( ...
                population, maps, map_weights, ...
                double(input.ga_iterations), double(input.Pc), ...
                double(input.Pm), pheromone, ...
                double(input.length_weight), ...
                double(input.smoothness_weight), ...
                allow_corner_cutting);
        if isempty(evaluated_paths)
            best_path = native_path;
        else
            best_path = evaluated_paths{reference_map_index};
        end
        fitness_history = [ ...
            aco_fitness_history(:); ga_fitness_history(:)
        ];
        failure_stage = 'final_validation';
        [success, validation_reason, max_grid_jump] = ...
            paper_validate_grid_path( ...
                reference_map, best_path, start_node, goal_node, ...
                allow_corner_cutting);
        if success
            failure_stage = 'none';
        end
    end
catch bridge_error
    validation_reason = [bridge_error.identifier ':' bridge_error.message];
end

execution_time = toc(t0);
save(output_file, 'best_path', 'best_fitness', 'fitness_history', ...
    'aco_fitness_history', 'ga_fitness_history', 'success', ...
    'execution_time', 'failure_stage', 'validation_reason', ...
    'max_grid_jump', 'initial_population_count', ...
    'reference_map_index', 'implementation_status', 'objective_scope');
end
