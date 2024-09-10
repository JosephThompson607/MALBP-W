# include("../read_MALBP_W.jl")
# include("preprocessing.jl")
# using BenchmarkTools

#orders the tasks for a model in descending order based on positional weight
#returns a list of tuples with the positional weight as the first element and the task name as the second element
function positional_weight_order(model::ModelInstance; _...)
    dependency_dict = calculate_positional_weight(model)
    #println("depedency dict ", dependency_dict)
    ordered_keys = sort(collect(zip(values(dependency_dict),keys(dependency_dict))), rev=true)
   # println("ordered keys ", ordered_keys)
    return ordered_keys
end

function reverse_positional_weight_order(model::ModelInstance; _...)
    dependency_dict = calculate_positional_weight(model, reverse=true)
    ordered_keys = sort(collect(zip(values(dependency_dict),keys(dependency_dict))))
    return ordered_keys
end

#Function that returns the first station that has the equipment to do the task
function find_first_compatible_station(equipment_assignments::Dict{Int, Vector{Int64}}, task::String, r_oe::Array{Int64,2})
    for (station, equipment_list) in equipment_assignments
        for equip in equipment_list
            if r_oe[parse(Int, task), equip] == 1
                return station
            end
        end
    end
end

#This function orders the tasks based on the previous equipment assignments and then by positional weight
function equip_positional_weight_order(model::ModelInstance; equipment_assignments::Dict{Int, Vector{Int64}}, equipment::EquipmentInstance, reverse::Bool=false, _...)
    dependency_dict = calculate_positional_weight(model, reverse=reverse)
    new_weights = []
    r_oe = permutedims((stack(equipment.r_oe)), [2,1])
    #Makes it first ordered by the earliest station, and then by the greatest positional weight
    for (task, weight) in dependency_dict
       push!(new_weights, (weight, -1 * find_first_compatible_station(equipment_assignments, task, r_oe), task))
    end
    ordered_keys = sort(new_weights, by= x->( x[2],x[1]), rev=true)
    #take only the task name
    ordered_keys = [ (weight,task) for (_, weight, task) in ordered_keys]
    #println("ordered keys ", ordered_keys)
    return ordered_keys
end

function create_precedence_matrix(model_instance::ModelInstance; order_function::Function, order_kwargs...)
    task_to_index = Dict{String, Int}()
    index_to_task = Dict{Int, String}()
    ordered_keys = order_function(model_instance; order_kwargs...)
    for (i,(_,task)) in enumerate(ordered_keys)
        task_to_index[task] = i
        index_to_task[i] = task
    end
    precedence_matrix = zeros(Int, model_instance.n_tasks+1, model_instance.n_tasks)
    for (pred, suc) in model_instance.precendence_relations
        pred_index = task_to_index[pred]
        suc_index = task_to_index[suc]
        precedence_matrix[pred_index, suc_index] = 1
    end
    precedence_matrix[end, :] = sum(precedence_matrix, dims=1)
    return precedence_matrix,  task_to_index, index_to_task
end

#Calculates the precedence matrix for each model in the instance
#The bottom row of the matrix is the "code", showing the number of immediate predecessors for each task
function create_precedence_matrices(instance::MALBP_W_instance; order_function::Function)
    matrix_dict = Dict{String, Dict}()
    for (model_name, model) in instance.models.models
        precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model, order_function= order_function)
        matrix_dict[model_name] = Dict{String, Any}("precedence_matrix" => precedence_matrix,
                                                    "task_to_index" => task_to_index,
                                                    "index_to_task" => index_to_task)
        
        
    end
    return matrix_dict
end

function calculate_new_cycle_time(model::ModelInstance, n_stations::Int, cycle_time::Int)
    #calculates the new cycle time based on the heuristic
    new_cycle_time = sum([time for (task, time) in model.task_times[1]]) / n_stations
    return max(cycle_time, new_cycle_time)
end

function ehsans_task_assign(instance::MALBP_W_instance, model::ModelInstance, precedence_matrix::Array{Int,2}, index_to_task::Dict{Int, String})
    new_cycle_time = calculate_new_cycle_time(model, instance.equipment.n_stations, instance.models.cycle_time)
    #creates a vector of remaining time left in the station. The length is the number of stations, and the value is the cycle time
    remaining_time = fill(new_cycle_time, instance.equipment.n_stations)
    x_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks)
    while any(precedence_matrix[end, :] .>= 0)
        #get the first zero element in the last row
        first_zero = findfirst(precedence_matrix[end, :] .== 0)
        #get the task name of the first zero element
        task = index_to_task[first_zero]
        #subtracts the task time from the remaining time of the first station with a positive balance
        station_no = findfirst(remaining_time .> 0)
        remaining_time[station_no] -= model.task_times[1][task]
        #subtracts the task from the "code"
        precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[first_zero, :]
        #marks the task as complete
        precedence_matrix[end, first_zero] = -1
        x_so[station_no, parse(Int, task)] = 1
    end
    return x_so
end

function ehsans_task_only(instance::MALBP_W_instance; order_function::Function = positional_weight_order, order_kwargs...)
    x_soi = ehsans_heuristic(instance, order_function= order_function, order_kwargs...)
    #y, y_w, y_wts, equipment_assignments
    return x_soi, nothing, nothing, nothing, nothing
end

function ehsans_heuristic(instance::MALBP_W_instance; order_function::Function = positional_weight_order, order_kwargs...)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    for (model_name, model) in instance.models.models
        i = findfirst( ==(model_name), model_indexes)
        precedence_matrix = precedence_matrices[model_name]["precedence_matrix"]
        index_to_task = precedence_matrices[model_name]["index_to_task"]
        x_soi[:,:,i] = ehsans_task_assign(instance, model, precedence_matrix, index_to_task)
    end
    return x_soi
end

function necessary_workers(tasks::Vector{String}, cycle_time::Real, model::ModelInstance, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    #calculates the number of workers needed to complete the tasks
    remaining_time = sum([model.task_times[1][task] for task in tasks])
    if remaining_time <= 0
        return 0
    end
    for (worker, productivity) in enumerate(productivity_per_worker)
        available_task_time = cycle_time * productivity
        remaining_time -= available_task_time
        if remaining_time <= 0
            return worker
        end
    end
   # @warn("Not enough workers to complete the tasks for model $(model.name): still have $(remaining_time) time left")
    return length(productivity_per_worker) 
end

function base_worker_assign_func(instance::MALBP_W_instance, x_soi::Array{Int64,3}; productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], start_station::Int=1, end_station::Int=instance.equipment.n_stations, assign_matrix::Array{Int,3}=zeros(Int, instance.sequences.n_scenarios, instance.num_cycles, instance.equipment.n_stations))
    model_index = [i for (i, model_dict) in instance.models.models]
    for (w,scenario) in enumerate(eachrow(instance.sequences.sequences))
        for (j, model) in enumerate(scenario.sequence)
            i = findfirst( ==(model), model_index)
            for s in start_station:end_station
                t = j + s - 1
                #gets the minimum number of workers needed to complete the tasks
                tasks = findall(x->x>0, x_soi[s,:,i])
                tasks = [string(task) for task in tasks if string(task) in keys(instance.models.models[model].task_times[1])]
                assign_matrix[w, t, s] = necessary_workers(tasks, 
                                                            instance.models.cycle_time, instance.models.models[model], 
                                                            productivity_per_worker)

            end
        end
    end
    return assign_matrix
end

function worker_assignment_heuristic(instance::MALBP_W_instance, x_soi::Array{Int,3}; productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    y_wts = base_worker_assign_func(instance, x_soi, productivity_per_worker = productivity_per_worker)
    y_w = zeros(Int, instance.sequences.n_scenarios)
    y = 0
    peak_wts = (0,0, 0)
    for (w, scenario) in enumerate(eachrow(instance.sequences.sequences))
        for t in 1:instance.num_cycles
            y_current = 0
            peak_current =0
            station_index = (0,0,0)
            for (station,station_workers) in enumerate(y_wts[w, t, :])
                #Just for identifying max station, not important for heuristic
                if station_workers > peak_current
                    peak_current = station_workers
                    station_index = (w, t, station)
                end
                y_current += station_workers
            end
            if y_current > y
                peak_wts = station_index
                y = y_current
            end

        end
    end
    return y, y_w, y_wts, peak_wts
end

function greedy_set_cover(tasks_to_assign::Vector{Int}, instance::MALBP_W_instance, station::Int)
    #if no tasks to assign, return empty list
    if length(tasks_to_assign) == 0
        return [], zeros(Int, instance.equipment.n_tasks), 0
    end
    #assigns the equipment to the stations
    equipment_costs = instance.equipment.c_se[station,:][1]
    #sorts the equipment by cost, keeping track of their index in the original list
    equipment_costs_tup = sort(collect(enumerate(equipment_costs)), by=x->x[2])
    equipment_assignments = Vector{Int64}()
    tasks = copy(tasks_to_assign)
    capabilities = zeros(Int, instance.equipment.n_tasks)
    #converts r_oe vector of vectors to a matrix
    r_oe = transpose(stack(instance.equipment.r_oe))
    #First pass: use the cheapest equipment to cover remaining tasks until all tasks are covered
    station_cost = 0
    while length(tasks) > 0
        o = popfirst!(tasks)
        for (e, cost) in equipment_costs_tup
            if instance.equipment.r_oe[o][ e] == 1
                push!(equipment_assignments, e)
                #removes the tasks that are covered by the equipment from the tasks
                tasks = filter(x->r_oe[ x, e] == 0, tasks)
                capabilities += r_oe[:, e]
                break
            end
        end
    end
    #second pass: remove equipment that are not needed
    #flip the equipment assignments, this will make the more expensive ones first
    equipment_assignments = reverse(equipment_assignments)
    filtered_equip = []
    for e in equipment_assignments
        reduced_cap = capabilities .- r_oe[:, e]
        equip_needed = false
        for task in tasks_to_assign
            if reduced_cap[ task] <= 0
                push!(filtered_equip, e)
                station_cost += equipment_costs[e]
                equip_needed = true
                break
            end
        end
        if !equip_needed
        #account for equipment being removed
            capabilities = reduced_cap
        end
    end
    return filtered_equip, capabilities, station_cost
end


function greedy_equipment_assignment_heuristic(instance::MALBP_W_instance, x_soi::Array{Int,3}, set_cover_heuristic::Function=greedy_set_cover)
    #assigns the equipment to the stations
    equipment_assignments = Dict{Int, Vector{Int64}}()
    
    for station in 1:instance.equipment.n_stations
        x_soi_station = x_soi[station, :, :]
        assigned_tasks = sum(x_soi_station, dims=2)
        assigned_tasks = dropdims(assigned_tasks, dims=2)
        #assigned tasks are all indices of nonzero elements
        assigned_tasks = findall(x->x>0, assigned_tasks)
        (equip_station_assignment, _ , _) = set_cover_heuristic(assigned_tasks, instance, station)
        equipment_assignments[station] = equip_station_assignment
    end
    return equipment_assignments
end

#heuristic start that first assigns tasks, then workers, then equipment
function sequential_heuristic_start_md( instance::MALBP_W_instance; 
    task_assign_func::Function = ehsans_heuristic, 
    worker_assign_func::Function = worker_assignment_heuristic, 
    equipment_assign_func::Function = greedy_equipment_assignment_heuristic, order_function::Function = positional_weight_order, task_order_kwargs...)
    #assigns tasks to stations
    x_soi = task_assign_func(instance, order_function= order_function, task_order_kwargs...)
    #assigns workers to stations
    y_start, y_w_start, y_wts_start = worker_assign_func(instance, x_soi)   
    equipment_assignments = equipment_assign_func(instance, x_soi)
    return x_soi, y_start, y_w_start, y_wts_start, equipment_assignments
end

#this heuristic alternates between assigning model tasks to stations and assigning equipment
function two_step_ehsans(instance::MALBP_W_instance; order_function::Function = positional_weight_order, reverse::Bool=false)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    models = [model for (model_name, model) in instance.models.models]
    #orders the models by decreasing probability
    models = sort(models, by=x->x.probability, rev=true)
    model1 = popfirst!(models)
    precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model1; order_function= order_function)
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    x_so = ehsans_task_assign(instance, model1, precedence_matrix, index_to_task)
    i = findfirst( ==(model1.name), model_indexes)
    x_soi[:,:,i] = x_so
    equipment_assignments = greedy_equipment_assignment_heuristic(instance, x_soi)
    for model in models
        precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model; order_function= equip_positional_weight_order, 
                                                                                            equipment_assignments = equipment_assignments, 
                                                                                            equipment = instance.equipment,
                                                                                            reverse=reverse)
        model_assignments = ehsans_task_assign(instance, model, precedence_matrix, index_to_task)
        i = findfirst( ==(model.name), model_indexes)
        x_soi[:,:,i] = model_assignments
        equipment_assignments = greedy_equipment_assignment_heuristic(instance, x_soi)
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(instance, x_soi)
    
    return x_soi, y, y_w, y_wts, equipment_assignments
end

function calculate_equip_cost(equipment_assignments::Dict{Int, Vector{Int64}}, instance::MALBP_W_instance)
    #calculates the total cost of the equipment assignments
    cost = 0
    for (station, equipment) in equipment_assignments
        for equip in equipment
            cost += instance.equipment.c_se[station][equip]
        end
    end
    return cost
end

function calculate_worker_cost(y::Int,y_w::Array{Int,1}, instance::MALBP_W_instance)
    #calculates the total cost of the workers
    main_worker_cost = y * instance.worker_cost
    recourse_cost = 0
    for (w,scenario) in enumerate(eachrow(instance.sequences.sequences))
        probability = scenario.probability
        recourse_cost += y_w[w] * instance.recourse_cost * probability
    end
    return main_worker_cost + recourse_cost
end


function calculate_c_time_si(instance::MALBP_W_instance)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    c_time_si = zeros(Float64, instance.equipment.n_stations, instance.models.n_models)
    for station in 1:instance.equipment.n_stations
        for (model_name, model) in instance.models.models
            i = findfirst( ==(model_name), model_indexes)
            c_time_si[station, i] = calculate_new_cycle_time(model, instance.equipment.n_stations, instance.models.cycle_time)
        end
    end
    return c_time_si
end


function fill_station!(instance::MALBP_W_instance,
                        unfinished_tasks::Vector{Vector{String}},
                        station::Int, 
                        models::Vector{Tuple{ModelInstance, Int64}}, 
                        c_time_si::Array{Float64, 2}, 
                        x_soi::Array{Int64, 3}, 
                        equipment_assignments::Dict{Int64, Vector{Int64}}, 
                        capabilities_so::Array{Int,2}, 
                        precedence_matrices::Dict{String, Dict};
                        set_cover_heuristic::Function=greedy_set_cover)
    while any(c_time_si[station,:] .> 0) && any([length(unfinished_tasks[i])>0 for i in 1:length(unfinished_tasks)]) 
        for (model, i) in models
            #skip if there is no capacity at the station or the model is finished
            if c_time_si[station, i] <= 0 || length(unfinished_tasks[i]) <=0
                if length(unfinished_tasks[i]) <= 0
                    c_time_si[station, i] = 0
                end
                continue
            end
            #equipment task indexes are different from the indexes in the precedence matrix
            precedence_matrix = precedence_matrices[model.name]["precedence_matrix"]
            task_to_index = precedence_matrices[model.name]["task_to_index"]
            progress = false
            for task in unfinished_tasks[i]
                task_index = task_to_index[task]
                o = parse(Int, task)
                if capabilities_so[station, o] > 0 && precedence_matrix[end, task_index] == 0
                    c_time_si[station, i] -= model.task_times[1][task]
                    precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[task_index, :]
                    #marks the task as complete
                    precedence_matrix[end, task_index] = -1
                    progress = true
                    #removes the tasks from the list of tasks 
                    unfinished_tasks[i] = filter(x->x!=task, unfinished_tasks[i])
                    x_soi[station, o, i] = 1
                    if c_time_si[station, i] <= 0
                        break
                    end
                end
            end
        #adds a task and recalculates the equipment to accomodate it
            if !progress
                task = popfirst!(unfinished_tasks[i])
                task_index = task_to_index[task]
                o = parse(Int, task)
                c_time_si[station, i] -= model.task_times[1][task]
                precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[task_index, :]
                #marks the task as complete
                precedence_matrix[end, task_index] = -1
                x_soi[station, o, i] = 1
                x_soi_station = x_soi[station, :, :]
                assigned_tasks = sum(x_soi_station, dims=2)
                assigned_tasks = dropdims(assigned_tasks, dims=2)
                #assigned tasks are all indices of nonzero elements
                assigned_tasks = findall(x->x>0, assigned_tasks)
                equipment_assignment, capabilities_so[station, :], _ = set_cover_heuristic(assigned_tasks, instance, station)
                equipment_assignments[station] = equipment_assignment
            end 
        end
    end
end


function task_equip_heuristic(instance::MALBP_W_instance; order_function::Function = positional_weight_order, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    #we need to sort the tasks by the order function so that it is respected in the assignment
    remaining_tasks = [ [task for (_, task) in order_function(model)] for (model,_) in models]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks)
    c_time_si = calculate_c_time_si(instance)
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    #Gets all of the remaining tasks for the two models
    for station in 1:instance.equipment.n_stations
        fill_station!(instance, remaining_tasks, station, models, c_time_si, x_soi, equipment_assignments, capabilities_so,  precedence_matrices)
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(instance, x_soi, productivity_per_worker = productivity_per_worker)
    return x_soi, y, y_w, y_wts, equipment_assignments
end

function task_equip_heuristic_task_only(instance::MALBP_W_instance; order_function::Function = positional_weight_order, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    #we need to sort the tasks by the order function so that it is respected in the assignment
    remaining_tasks = [ [task for (_, task) in order_function(model)] for (model,_) in models]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks)
    c_time_si = calculate_c_time_si(instance)
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    #Gets all of the remaining tasks for the two models
    for station in 1:instance.equipment.n_stations
        fill_station!(instance, remaining_tasks, station, models, c_time_si, x_soi, equipment_assignments, capabilities_so,  precedence_matrices)
    end
    #y, y_w, y_wts, _ = worker_assignment_heuristic(instance, x_soi, productivity_per_worker = productivity_per_worker)
    return x_soi, nothing, nothing, nothing, nothing
end





# function check_bounds(left::Int, right::Int, station::Int, x_os::Dict{String, Dict{String, Int}}, task::String, model::String, comp_dict::Dict{String, Dict{String, Vector{String}}}; max_stations::Int)
#     # We need to make sure that task2 is not being put before its predecessors
#     if left >= station
#         task_bounds = calculate_left(x_os, task, model, comp_dict)
#         if task_bounds > station
#             return false
#         else
#             return true
#         end
#     # We need to make sure that task2 is not being put after its successors
#     elseif right <= station
#         task_bounds = calculate_right(x_os, task, model, comp_dict, max_stations = max_stations)
#         if task_bounds < station
#             return false
#         else
#             return true
#         end
#     else
#         #println("left: ", left, " right: ", right, " station: ", station, " task: ", task, " model: ", model)
#         @warn("bounds improperly set for task $(task) at station $(station) in model $(model)")
#     end
# end

# function opt2_insertion!(x_soi::Array{Int,3}, 
#                         iteration ::Int,
#                         station::Int, 
#                         left::Int, 
#                         right::Int, 
#                         task::String,
#                         model::String, 
#                         i::Int, 
#                         instance::MALBP_W_instance, 
#                         comp_dict::Dict{String, Dict{String, Vector{String}}},
#                         x_os::Dict{String, Dict{String, Int}}, 
#                         x_so::Dict{String, Dict{Int64, Vector{String}}})
#     #we need to determine what function to use to check the other tasks
#     o = parse(Int, task)
#     #println("original station: ", station, " tasks: ", x_so[model][station])
#     y_s = necessary_workers_w_block(x_so[model][station], instance.models.cycle_time, instance.models.models[model])
#     #println("station: ", station, " task: ", task, " workers at station: ", y_s)
#     #println("left: ", left, " right: ", right)
#     for s_prime in left:right
#         #println("s prime: ", s_prime, "tasks: ", x_so[model][s_prime])
#         y_s_prime = necessary_workers_w_block(x_so[model][s_prime], instance.models.cycle_time, instance.models.models[model])
#         total_workers = y_s + y_s_prime
#         #println("s prime: ", s_prime, " s ", station, " task ", task, " total workers ", total_workers)
#         for task2 in x_so[model][s_prime]
#             #println("task2: ", task2)
#             removed_1 = filter(x->x!=task, x_so[model][station])
#             #println("removing task: ", task, " from station: ", station, " adding task: ", task2, )
#             new_y_s = necessary_workers_w_block([removed_1; task2], instance.models.cycle_time, instance.models.models[model])
#             #println(" removing task: ", task2, " from station: ", s_prime, " adding task: ", task,)
#             removed_2 = filter(x->x!=task2, x_so[model][s_prime])
#             new_y_sprime = necessary_workers_w_block([removed_2; task], instance.models.cycle_time, instance.models.models[model])
#             #Swaps are invalid if they violate worker or precedence constraints
            
#             # if check_bounds(left, right, station, x_os, task2, model, comp_dict, max_stations = instance.equipment.n_stations)
#             #     println("task2: ", task2, " sprime ", s_prime, " task1: ", task,  " station ", station, " left ", left, " right ", right,)
#             #     println( " x_o2s ", x_os[model][task2], " station ", station, " x_o1s ", x_os[model][task], " comp_dict ", comp_dict[model][task2])
#             # end
  

#             if isnothing(new_y_s) || isnothing(new_y_sprime) || !check_bounds(left, right, station, x_os, task2, model, comp_dict, max_stations = instance.equipment.n_stations)
#                 continue
#             end
#             new_workers = new_y_s + new_y_sprime
#             #println("y_s_prime: ", y_s_prime, " new_y_sprime: ", new_y_sprime, " y_s: ", y_s, " new y_s: ", new_y_s)
#             #println("new_workers: ", new_workers, " total workers ", total_workers)
            
#             if new_workers < total_workers || (new_workers == total_workers && rand() < (0.5))
#                 println("IMPROVEMENT", " total workers ", total_workers, " new workers ", new_workers, " diff: ", total_workers - new_workers)
#                 o2 = parse(Int, task2)
#                 # update x_so
#                 push!(x_so[model][s_prime], task)
#                 push!(x_so[model][station], task2)
#                 x_so[model][s_prime] = filter(x->x!=task2, x_so[model][s_prime])
#                 x_so[model][station] = filter(x->x!=task, x_so[model][station])
#                 #update x_os
#                 x_os[model][task] = station
#                 x_os[model][task2] = s_prime
#                 #update x_soi
#                 x_soi[s_prime, o, i] = 1
#                 x_soi[s_prime, o2, i] = 0
#                 x_soi[station, o, i] = 0
#                 x_soi[station, o2, i] = 1
#                 return true
#             end 
#         end
#     end
#     return false
# end

# function task_2opt(instance::MALBP_W_instance, x_soi::Array{Int,3},; n_iterations::Int=100)
#     #calculates the total cost of the equipment assignments
#     #equipment_assignments = greedy_equipment_assignment_heuristic(instance, x_soi)
#     #y, y_w, y_wts, peak = worker_assignment_heuristic(instance, x_soi)
#     #calculates the total cost of the workers
#     #cost = calculate_worker_cost(y, y_w, instance)
#     #cost += calculate_equip_cost(equipment_assignments, instance)
#     predecessors, successors = precedence_relations_dict(instance)
#     counter = 0
#     x_so, x_os = x_soi_to_dict(instance, x_soi)
#     while counter < n_iterations
#         for (i,(model_name, model)) in enumerate(instance.models.models)
#             for (task1, station) in x_os[model_name]
#                 left = calculate_left(x_os, task1, model_name, predecessors)
#                 #println("task 1: ", task1, " station: ", station)
#                 #println("LEFT: ", left)
#                 improvement = opt2_insertion!( x_soi, counter, station, left, station-1, task1, model_name, i, instance ,  successors, x_os, x_so)
#                 if improvement
#                     continue
#                 end
#                 right = calculate_right(x_os, task1, model_name, successors, max_stations = instance.equipment.n_stations)
#                 #println("RIGHT: ", right)
#                 improvement = opt2_insertion!(x_soi, counter, station, station+1, right, task1, model_name, i, instance, predecessors, x_os, x_so)
#             end
#         end
#         counter += 1
#     end
#     equipment_assignments = greedy_equipment_assignment_heuristic(instance, x_soi)
#     y, y_w, y_wts, peak = worker_assignment_heuristic(instance, x_soi)

#     return x_soi, y, y_w, y_wts, equipment_assignments
# end 


# println("x_soi: ", x_soi)
# println("y: ", y)
# println("y_w: ", y_w)
# println("y_wts: ", y_wts)
# println("equipment_assignments: ", equipment_assignments)
# results = []
# for instance in instances
#     global x = instance[1]
#     b = @belapsed (global y = task_equip_heuristic(x)) 
#     (push!(results, Dict("time"=>b, "instance"=>x.name)))
# end
# #turns results to DataFrame
# results_df = DataFrame(results)
# println(results_df)
# #saves the results to a csv file
# CSV.write("task_equip_time_results.csv", results_df)

# results = []
# for instance in instances
#     global x = instance[1]
#     b = @belapsed (global y = ehsans_task_only(x)) 
#     (push!(results, Dict("time"=>b, "instance"=>x.name)))
# end
# #turns results to DataFrame
# results_df = DataFrame(results)
# println(results_df)
# #saves the results to a csv file
# CSV.write("ehsans_time_results.csv", results_df)

#println("instances: ", instances)

#config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/julia_debug.yaml"
# #config_filepath = "SALBP_benchmark/MM_instances/medium_instance_config_S10.yaml"
#instances = read_MALBP_W_instances(config_filepath)
#instances
# instance = instances[2]

# x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
# total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
# println("original cost: ", total_cost, " y ", y, " u_se ", equipment_assignments)


# # x_soi, y, y_w, y_wts, equipment_assignments = task_1opt(instance, x_soi, n_iterations=100)
# # total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
# # println("new cost: ", total_cost, " y ", y, " u_se ", equipment_assignments)
# 5000
# x_soi, y, y_w, y_wts, equipment_assignments = task_2opt(instance, x_soi, n_iterations= 10000 )
# total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
# println("new cost 2opt: ", total_cost, " y ", y, " u_se ", equipment_assignments)

#x_soi, y, y_w, y_wts, equipment_assignments = task_1opt(instance, x_soi, equipment_assignments)
#println(findall(x->x>0, x_soi))
# for (i, instance) in enumerate(instances)
#     println(i, instance)
#      x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
#      end
# seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
# seq_worker_cost = calculate_worker_cost(y, y_w, instance)
# rev_seq_total = seq_equip_cost + seq_worker_cost
# println("seq_equip_cost: ", seq_equip_cost)
# println("seq_worker_cost: ", seq_worker_cost)
# println("rev_seq_total: ", rev_seq_total)

# model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance)
# x_soi = ehsans_heuristic(instance)
# u_se = greedy_equipment_assignment_heuristic(instance, x_soi)
# y, y_w, y_wts = worker_assignment_heuristic(instance, x_soi)
# results = []
# for instance in instances
#     model_task_assignments, y, y_w, y_wts, equipment_assignments = sequential_heuristic_start_md(instance; order_function = reverse_positional_weight_order)
#     seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     seq_worker_cost = calculate_worker_cost(y, y_w, instance)
#     rev_seq_total = seq_equip_cost + seq_worker_cost

#     model_task_assignments, y, y_w, y_wts, equipment_assignments = sequential_heuristic_start_md(instance)
#     seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     seq_worker_cost = calculate_worker_cost(y, y_w, instance)
#     seq_total = seq_equip_cost + seq_worker_cost

#     model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance)
#     two_step_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     two_step_worker_cost = calculate_worker_cost(y, y_w, instance)
#     two_step_total = two_step_equip_cost + two_step_worker_cost


#     model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance; order_function = reverse_positional_weight_order, reverse=true)
#     two_step_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     two_step_worker_cost = calculate_worker_cost(y, y_w, instance)
#     rev_two_step_total = two_step_equip_cost + two_step_worker_cost

#     x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
#     seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     seq_worker_cost = calculate_worker_cost(y, y_w, instance)
#     task_equip_total = seq_equip_cost + seq_worker_cost

#     x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance; order_function = reverse_positional_weight_order)
#     seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
#     seq_worker_cost = calculate_worker_cost(y, y_w, instance)
#     task_equip_rev_total = seq_equip_cost + seq_worker_cost


#      push!(results, (instance.name, seq_total, two_step_total ,rev_seq_total, rev_two_step_total, task_equip_total, task_equip_rev_total))
#  end
# # #turns results to DataFrame
#  results_df = DataFrame(results, [:instance, :seq_total, :two_step_total, :rev_seq_total, :rev_two_step_total, :task_equip_total, :task_equip_rev_total])
#  println(results_df)