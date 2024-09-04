
using Graphs
include("../read_MALBP_W.jl")
include("../heuristics/preprocessing.jl")
include("../heuristics/constructive.jl")




function create_combined_precedence_diagram(instance)
    precedence_relations = []
    task_times = Dict{String, Float64}()
    precedence_graph = SimpleDiGraph(instance.equipment.n_tasks)
    for model in values(instance.models.models)
        if length(model.precendence_relations) > 0
            for (task, successor) in model.precendence_relations
                #creates a list of tuples with the edges of the graph
                add_edge!(precedence_graph, parse(Int, task), parse(Int, successor))
            end
            
        end
        for (task, time) in model.task_times[1]
            if !haskey(task_times, task)
                task_times[task] = time
            elseif task_times[task] < time
                task_times[task] = time
            end
        end
    end
    #removes unecessary edges from the graph
    precedence_graph = transitivereduction(precedence_graph)
    #creates a list of tuples with the edges of the graph
    for edge in edges(precedence_graph)
        pair = Tuple(edge)
        #converts the elements to strings
        edge = [string(pair[1]), string(pair[2])]
        push!(precedence_relations, edge)
    end
    return precedence_relations, task_times
end

#This function creates a new instance with a single model that combines all the models of the original instance
function combine_to_new_instance(orig_instance)
    instance = deepcopy(orig_instance)
    precedence_relations, task_times = create_combined_precedence_diagram(instance)
    nested_task_times = Dict{Int, Dict{String,Float64}}(1 => task_times)
    combined_model = ModelInstance("combined_model", 1.0, length(task_times), 1.0, precedence_relations, nested_task_times)
    #n_models = length(instance.models.models) + 1
    n_models=1
    models_dict = Dict{String, ModelInstance}()
    models_dict["combined_model"] = combined_model
    new_models = ModelsInstance("combined_model", "combined_model", n_models , instance.models.cycle_time, models_dict)
    new_instance = MALBP_W_instance(instance.filepath, instance.name, instance.config_name, new_models, instance.sequences, instance.equipment, instance.n_stations, instance.max_workers, instance.productivity_per_worker, instance.worker_cost, instance.recourse_cost, instance.num_cycles, instance.MILP_models)
    return new_instance
end



function count_covered_items(tasks, r_eo)
    #This function counts how many items 
    n_equipment = size(r_eo)[1]
    capabilities= zeros(n_equipment)
    covered_tasks = Dict()
    for e in 1:n_equipment
        covered_tasks[e] = []
        for o in tasks
            if r_eo[e, o] >0
                append!(covered_tasks[e], o)
                capabilities[e] +=1
            end
        end
    end
    #ADDS IN A SMALL AMOUNT TO AVOID DIVIDING BY ZERO
    return capabilities .+ 0.000001 , covered_tasks
end


#Greedy set cover with aln n error bounds
function greedy_set_cover_v2(tasks::Vector{Int}, instance::MALBP_W_instance, station::Int)
    tasks_to_assign = copy(tasks)
    #if no tasks to assign, return empty list
    if length(tasks_to_assign) == 0
        return [], zeros(Int, instance.equipment.n_tasks)
    end
    #assigns the equipment to the stations
    equipment_costs = instance.equipment.c_se[station,:][1]
    #sorts the equipment by cost, keeping track of their index in the original list
    #equipment_costs = sort(collect(enumerate(equipment_costs)), by=x->x[2])

    equipment_assignments = Vector{Int64}()
    capabilities = zeros(Int, instance.equipment.n_tasks)
    #converts r_oe vector of vectors to a matrix
    r_eo = stack(instance.equipment.r_oe)
    #counter = 1
    station_cost = 0.0
    while !isempty(tasks_to_assign) #&& counter < 5
        capability_counts, covered_tasks = count_covered_items(tasks_to_assign,r_eo)
        relative_costs = equipment_costs ./ capability_counts
        selected_equipment = argmin(relative_costs)
        tasks_to_assign = setdiff(tasks_to_assign, covered_tasks[selected_equipment])
        push!(equipment_assignments,selected_equipment)
        capabilities += r_eo[selected_equipment, :]
        station_cost += equipment_costs[selected_equipment]
        #counter +=1

    end
    
    return equipment_assignments, capabilities, station_cost
end

#This function creates a new instance with a single model that combines all the models of the original instance and then solves the problem
function task_equip_heuristic_combined_precedence(orig_instance::MALBP_W_instance; order_function::Function = positional_weight_order, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], set_cover_heuristic=greedy_set_cover)
    instance = combine_to_new_instance(orig_instance)
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    #we need to sort the tasks by the order function so that it is respected in the assignment
    remaining_tasks = [ [task for (_, task) in order_function(model)] for (model,_) in models]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks)
    c_time_si = calculate_c_time_si(instance)
    x_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    #Gets all of the remaining tasks for the two models
    for station in 1:instance.equipment.n_stations
        fill_station!(instance, remaining_tasks, station, models, c_time_si, x_so, equipment_assignments, capabilities_so,  precedence_matrices, set_cover_heuristic=set_cover_heuristic)
    end
    #x_soi is x_so copied to a 3D array with orig_instance number of models
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, orig_instance.models.n_models)
    for model in 1:orig_instance.models.n_models
        x_soi[:,:,model] = x_so[:,:,1]
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(orig_instance, x_soi, productivity_per_worker = productivity_per_worker)
    return x_soi, y, y_w, y_wts, equipment_assignments
end


#This function creates a new instance with a single model that combines all the models of the original instance and then solves the problem
function task_equip_heuristic_task_only_combined_precedence(orig_instance::MALBP_W_instance; order_function::Function = positional_weight_order, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.],
    set_cover_heuristic::Function = greedy_set_cover)
    instance = combine_to_new_instance(orig_instance)
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    #we need to sort the tasks by the order function so that it is respected in the assignment
    remaining_tasks = [ [task for (_, task) in order_function(model)] for (model,_) in models]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks)
    c_time_si = calculate_c_time_si(instance)
    x_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    #Gets all of the remaining tasks for the two models
    for station in 1:instance.equipment.n_stations
        fill_station!(instance, remaining_tasks, station, models, c_time_si, x_so, equipment_assignments, capabilities_so,  precedence_matrices, set_cover_heuristic = set_cover_heuristic)
    end
    #x_soi is x_so copied to a 3D array with orig_instance number of models
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, orig_instance.models.n_models)
    for model in 1:orig_instance.models.n_models
        x_soi[:,:,model] = x_so[:,:,1]
    end
    #y, y_w, y_wts, _ = worker_assignment_heuristic(orig_instance, x_soi, productivity_per_worker = productivity_per_worker)
    return x_soi, nothing, nothing, nothing, nothing
end

 #Function for calculating the stating number of workers at each station
 function chunk_out_tasks(productivity_per_worker::Array{Float64}, instance::MALBP_W_instance, remaining_time::Real, start_station::Int64 = 1, end_station::Int64 = instance.n_stations)
    #Here we assume tasks will be evenely distributed among stations.
    workers_per_station = zeros(Int, instance.n_stations)
    available_station_time = zeros(Float64, instance.n_stations)
    #assigns workers to stations, starting with the most productive workers (first workers)
    for (worker, productivity) in enumerate(productivity_per_worker)
        for station in start_station:end_station
            available_task_time = instance.models.cycle_time * productivity
            remaining_time -= available_task_time
            workers_per_station[station] += 1
            available_station_time[station] += available_task_time
            if remaining_time <= 0
                return workers_per_station, available_station_time
            end
        end
    end
    error("Not enough workers to complete the tasks: $(workers_per_station) $(remaining_time)" )
end


function calculate_min_workers(instance::MALBP_W_instance; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.], start_station::Int64=1, end_station::Int64 = instance.n_stations, tasks::Union{Vector{String}, Nothing} = nothing )
    min_workers = Dict{String, Dict}()
   
    for (model_name, model) in instance.models.models
        if isnothing(tasks)
            tasks = keys(model.task_times[1])
                end
        remaining_time = sum([time for (task, time) in model.task_times[1] if task in tasks ])
        workers_per_station, available_station_time = chunk_out_tasks(productivity_per_worker, instance, remaining_time, start_station, end_station)
        min_workers[model_name] = Dict("workers_per_station" => workers_per_station, 
                                    "available_station_time"=> available_station_time, 
                                    "total_workers" => sum(workers_per_station))
    end
    
    return min_workers
end

#This function undues the assignment of tasks to stations until it reaches the first station with the fewest workers
function backtrack(precedence_matrix::Array{Int}, best_sequence::Vector{Int}, small_station::Int)
    for task in length(best_sequence):-1:1
        if best_sequence[task] < -small_station
            best_sequence[task] = 0
            best_sequence += precedence_matrix[task, :] 
        end
    end
    return best_sequence
end

#cleanup worker assignment makes sure that there are only enough workers to complete the assigned tasks
function cleanup_worker_assignment(cycle_leftovers, min_workers, productivity_per_worker, cycle_time)
    for (station, workers) in enumerate(min_workers["combined_model"]["workers_per_station"])
        #if we did not record cycle leftovers, we did not use that station and can release the workers
        if station > length(cycle_leftovers)
            min_workers["combined_model"]["workers_per_station"][station] = 0
            min_workers["combined_model"]["available_station_time"][station] = 0
            continue
        #if the cycle leftover is greater than the productivity of the workers, we can release workers
        elseif cycle_leftovers[station] > cycle_time * productivity_per_worker[workers]
            while cycle_leftovers[station] > cycle_time * productivity_per_worker[workers]
                min_workers["combined_model"]["workers_per_station"][station] -= 1
                min_workers["combined_model"]["available_station_time"][station] -= cycle_time * productivity_per_worker[workers]
                cycle_leftovers[station] -= cycle_time * productivity_per_worker[workers]
                workers -= 1
            end
        end
    end
    return min_workers, cycle_leftovers
end

#This function tries to assign tasks using the hoffman heuristic, modified for workers at a fixed number of stations
function modified_hoffman(orig_instance::MALBP_W_instance; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.] )
    instance = combine_to_new_instance(orig_instance)
    best_sequence = []
    cycle_leftovers = []
    min_workers = calculate_min_workers(instance; productivity_per_worker = productivity_per_worker)
    precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(instance.models.models["combined_model"], order_function = positional_weight_order)
    iteration = 0
    while true 
        best_sequence, cycle_leftovers = hoffman_MALBP(instance.models.models["combined_model"], min_workers["combined_model"]["available_station_time"], precedence_matrix, index_to_task)
        if all(best_sequence .< 0)
            break
        elseif iteration > 1000 || minimum(min_workers["combined_model"]["workers_per_station"]) >= instance.max_workers
            @warn("Not enough stations to complete the tasks, will have to backtrack. Here is the best sequence so far $(best_sequence)")
            return [], cycle_leftovers, min_workers
        else
            #find the first station with the fewest workers
            small_station = findfirst(x -> x == minimum(min_workers["combined_model"]["workers_per_station"]), min_workers["combined_model"]["workers_per_station"])
            #undoes the assignment of tasks to stations until it reaches the first station with the fewest workers
            best_sequence = backtrack(precedence_matrix, best_sequence, small_station)
            #add one worker to the station with the fewest workers
            min_workers["combined_model"]["workers_per_station"][small_station] += 1
            #adds the productivity of the worker to the available time
            min_workers["combined_model"]["available_station_time"][small_station] += instance.models.cycle_time * productivity_per_worker[min_workers["combined_model"]["workers_per_station"][small_station]]
            iteration += 1 
        end
    end
    min_workers, cycle_leftovers = cleanup_worker_assignment(cycle_leftovers, min_workers, productivity_per_worker, instance.models.cycle_time)
    return best_sequence, cycle_leftovers, min_workers, index_to_task
end

#Hoffman heuristic for combined precedence diagram MALBP
function hoffman_MALBP(model_instance::ModelInstance, cycle_time::Vector{Float64}, precedence_matrix::Array{Int}, index_to_task::Dict{Int, String})
    function recursive_task_fill(station::Int, left_task::Int,  best_sequence::Vector{Int}, best_value::Real)
       #check if all tasks have been assigned
        if all(best_sequence .< 0)
            return best_sequence, best_value
        end
        sequence = copy(best_sequence)
        #finds the index of the first zero of the last row
        limited_seq = sequence[left_task:end]
        if isnothing(findfirst(x -> x == 0, limited_seq)) 
            return best_sequence, best_value
        end
        task_index = (findfirst(x -> x == 0, limited_seq)) + left_task - 1
        #Retrieves the task from the precedence matrix and assigns it to the station
        task  = index_to_task[task_index]
        test_leftovers = best_value - model_instance.task_times[1][task]
        if test_leftovers < 0
            left_task = task_index + 1
            seq, val = recursive_task_fill(station, left_task, sequence, best_value)
            return  seq, val
        else
            next_left = left_task + 1
            seq2, val2 = recursive_task_fill(station, next_left, sequence, best_value)
            sequence[task_index] = -station
            sequence -= precedence_matrix[task_index, :]
            seq, val = recursive_task_fill(station, left_task,   sequence, test_leftovers)
            if val < val2
                return seq, val
            else
                return seq2, val2
            end
            return seq, val
        end
    end
   
    best_sequence = copy(precedence_matrix[end, :])
    station  = 1
    cycle_leftovers = [] 
    left_task = 1  
    n_iterations = 0
    while any(best_sequence .>= 0) && n_iterations < 1000
        best_sequence, best_value = recursive_task_fill(station, left_task, best_sequence, cycle_time[station])
        push!(cycle_leftovers, best_value)
        left_task = findfirst(x -> x == 0, best_sequence)
        station += 1
        if station > length(cycle_time) && any(best_sequence .>= 0)
            @info("Not enough stations to complete the tasks, will have to backtrack. Here is the best sequence so far $(best_sequence)")
            break
        end
        n_iterations += 1
    end
    return best_sequence, cycle_leftovers
end

function two_step_combined_hoffman(orig_instance)
    best_sequence, cycle_leftovers, min_workers = modified_hoffman(orig_instance)
    best_sequence .*= -1
    x_soi = zeros(Int, orig_instance.equipment.n_stations, orig_instance.equipment.n_tasks, orig_instance.models.n_models)
    index_to_model = Dict{Int, String}(index => model for (index, model) in enumerate(keys(orig_instance.models.models)))
    for i in 1:orig_instance.models.n_models
        model = index_to_model[i]
        #one hot encoding of the best sequence
        for (task, station) in enumerate(best_sequence)
            if string(task) in keys(orig_instance.models.models[model].task_times[1])
                x_soi[station,task,i] = 1
            end
        end
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(orig_instance, x_soi)
    equipment = greedy_equipment_assignment_heuristic(orig_instance, x_soi)
    return x_soi, y, y_w, y_wts, equipment
end

function hoffman_task_only(orig_instance)
    best_sequence, cycle_leftovers, min_workers, index_to_task = modified_hoffman(orig_instance)
    best_sequence .*= -1
    x_soi = zeros(Int, orig_instance.equipment.n_stations, orig_instance.equipment.n_tasks, orig_instance.models.n_models)
    index_to_model = Dict{Int, String}(index => model for (index, model) in enumerate(keys(orig_instance.models.models)))
    for i in 1:orig_instance.models.n_models
        model = index_to_model[i]
        #one hot encoding of the best sequence
        for (ind, station) in enumerate(best_sequence)
            task = index_to_task[ind]
            if task in keys(orig_instance.models.models[model].task_times[1])
                o = parse(Int, task)
                x_soi[station, o ,i] = 1
            end
        end
    end
    return x_soi, nothing, nothing, nothing, nothing
end

# #config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/constructive_debug.yaml"
# config_filepath = "SALBP_benchmark/MM_instances/xp_yaml/medium_instance_config_S10.yaml"
# instances = read_MALBP_W_instances(config_filepath)
# # instance = instances[1]
# # x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic_combined_precedence(instance, set_cover_heuristic=greedy_set_cover_v2)
# # total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
# # println("new set cover combined cost: ", total_cost, " y ",y,  " u_se ", equipment_assignments)

# for instance in instances

#     model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance)
#     total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
#     println("two step cost: ", total_cost, " y ", y, " u_se ", equipment_assignments)
#     x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
#     total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
#     println("original cost: ", total_cost, " y ", y, " u_se ", equipment_assignments)

#     x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic_combined_precedence(instance)
#     total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
#     println("combined cost: ", total_cost, " y ",y,  " u_se ", equipment_assignments)


#     x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic_combined_precedence(instance, set_cover_heuristic=greedy_set_cover_v2)
#     total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
#     println("new set cover combined cost: ", total_cost, " y ",y,  " u_se ", equipment_assignments)

#     x_soi, y, y_w, y_wts, equipment_assignments = two_step_combined_hoffman(instance)
#     total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
#     println("two step combined cost: ", total_cost, " y ", y, " u_se ", equipment_assignments)
# end
