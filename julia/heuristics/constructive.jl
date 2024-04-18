include("../read_MALBP_W.jl")
include("preprocessing.jl")

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
    precedence_matrix = zeros(Int, model_instance.no_tasks+1, model_instance.no_tasks)
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
        # println("precedence matrix: ", model_name)
        # #prints each row of the precedence matrix
        # for i in 1:(model.no_tasks + 1)
        #     if i <= model.no_tasks
        #         println(index_to_task[i], precedence_matrix[i, :])
        #     else
        #     println(precedence_matrix[i, :])
        #     end
        # end
        matrix_dict[model_name] = Dict{String, Any}("precedence_matrix" => precedence_matrix,
                                                    "task_to_index" => task_to_index,
                                                    "index_to_task" => index_to_task)
        
        
    end
    return matrix_dict
end
 
 
 #Function for calculating the stating number of workers at each station
 function chunk_out_tasks(productivity_per_worker::Dict{Int, Float64}, instance::MALBP_W_instance, remaining_time::Real)
    #Here we assume tasks will be evenely distributed among stations.
    workers_per_station = zeros(Int, instance.no_stations)
    available_station_time = zeros(Int, instance.no_stations)
    #sorts the dictionary by key
    productivity_per_worker = sort(collect(productivity_per_worker), by=x->x[1])
    #assigns workers to stations, starting with the most productive workers (first workers)
    for (worker, productivity) in productivity_per_worker
        for station in 1:instance.no_stations
            available_task_time = instance.models.cycle_time * productivity
            remaining_time -= available_task_time
            workers_per_station[station] += 1
            available_station_time[station] += available_task_time
            if remaining_time <= 0
                return workers_per_station, available_station_time
            end
        end
    end
    error("Not enough workers to complete the tasks")
end


function calculate_min_workers(instance::MALBP_W_instance, productivity_per_worker::Dict{Int, Float64})
    min_workers = Dict{String, Dict}()
    for (model_name, model) in instance.models.models
        remaining_time = sum([time for (task, time) in model.task_times[1]])
        workers_per_station, available_station_time = chunk_out_tasks(productivity_per_worker, instance, remaining_time)
        min_workers[model_name] = Dict("workers_per_station" => workers_per_station, 
                                    "available_station_time"=> available_station_time, 
                                    "total_workers" => sum(workers_per_station))
    end
    
    return min_workers
end

function md_assign_tasks(instance::MALBP_W_instance; productivity_per_worker::Dict{Int, Float64}= Dict(1=>1., 2=>1., 3=>1., 4=>1.), priority!::Function= x->x)
    #creates a vector of remaining time left in the station. The length is the number of stations, and the value is the cycle time
    min_workers = calculate_min_workers(instance, productivity_per_worker)
    remaining_time = fill(instance.cycle_time, instance.no_stations)

    #creates a matrix from the prededence constraints. The matrix is a boolean matrix with 1 if the task is a predecessor of the task in the column
    #and 0 otherwise
    precedence_matrix_dict = create_precedence_matrices(instance)


    for model in instance.models
        for (task, time) in model.task_times
            
        end
    end


end

function calculate_new_cycle_time(model::ModelInstance, no_stations::Int, cycle_time::Int)
    #calculates the new cycle time based on the heuristic
    new_cycle_time = sum([time for (task, time) in model.task_times[1]]) / no_stations
    return max(cycle_time, new_cycle_time)
end

function ehsans_task_assign(instance::MALBP_W_instance, model::ModelInstance, precedence_matrix::Array{Int,2}, index_to_task::Dict{Int, String})
    new_cycle_time = calculate_new_cycle_time(model, instance.equipment.no_stations, instance.models.cycle_time)
    #creates a vector of remaining time left in the station. The length is the number of stations, and the value is the cycle time
    remaining_time = fill(new_cycle_time, instance.equipment.no_stations)
    x_so = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks)
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

function ehsans_heuristic(instance::MALBP_W_instance; order_function::Function = positional_weight_order, order_kwargs...)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    x_soi = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks, instance.models.no_models)
    for (model_name, model) in instance.models.models
        i = findfirst( ==(model_name), model_indexes)
        precedence_matrix = precedence_matrices[model_name]["precedence_matrix"]
        index_to_task = precedence_matrices[model_name]["index_to_task"]
        x_soi[:,:,i] = ehsans_task_assign(instance, model, precedence_matrix, index_to_task)
    end
    return x_soi
end

function necessary_workers(tasks::Vector{String}, cycle_time::Real, model::ModelInstance, productivity_per_worker::Array{Float64})
    #calculates the number of workers needed to complete the tasks
    remaining_time = sum([model.task_times[1][task] for task in tasks])
    for (worker, productivity) in enumerate(productivity_per_worker)
        available_task_time = cycle_time * productivity
        remaining_time -= available_task_time
        if remaining_time <= 0
            return worker
        end
    end
    @warn("Not enough workers to complete the tasks")

    return length(productivity_per_worker)
end

function base_worker_assign_func(instance::MALBP_W_instance, x_soi::Array{Int64,3}; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.])
    model_index = [i for (i, model_dict) in instance.models.models]
    assign_matrix = zeros(Int, instance.no_scenarios, instance.no_cycles, instance.equipment.no_stations )
    for (w,scenario) in enumerate(eachrow(instance.scenarios))
        for (j, model) in enumerate(scenario.sequence)
            i = findfirst( ==(model), model_index)
            for s in 1:instance.equipment.no_stations
                t = j + s - 1
                #gets the minimum number of workers needed to complete the tasks
                tasks = findall(x->x>0, x_soi[s,:,i])
                tasks = [string(task) for task in tasks]
                assign_matrix[w, t, s] = necessary_workers(tasks, 
                                                            instance.models.cycle_time, instance.models.models[model], 
                                                            productivity_per_worker)

            end
        end
    end
    return assign_matrix
end

function worker_assignment_heuristic(instance::MALBP_W_instance, x_soi::Array{Int,3}; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.])
    y_wts = base_worker_assign_func(instance, x_soi, productivity_per_worker = productivity_per_worker)
    y_w = zeros(Int, instance.no_scenarios)
    y = 0
    for (w, scenario) in enumerate(eachrow(instance.scenarios))
        for t in 1:instance.no_cycles
            needed_workers = sum(y_wts[w, t, :])
            y = max(y, needed_workers)
        end
    end
    return y, y_w, y_wts
end

function greedy_set_cover(tasks_to_assign::Vector{Int}, instance::MALBP_W_instance, station::Int)
    #if no tasks to assign, return empty list
    if length(tasks_to_assign) == 0
        return [], zeros(Int, instance.equipment.no_tasks)
    end
    #assigns the equipment to the stations
    equipment_costs = instance.equipment.c_se[station,:][1]
    #sorts the equipment by cost, keeping track of their index in the original list
    equipment_costs = sort(collect(enumerate(equipment_costs)), by=x->x[2])
    equipment_assignments = Vector{Int64}()
    tasks = copy(tasks_to_assign)
    capabilities = zeros(Int, instance.equipment.no_tasks)
    #converts r_oe vector of vectors to a matrix
    r_oe = transpose(stack(instance.equipment.r_oe))
    #First pass: use the cheapest equipment to cover remaining tasks until all tasks are covered
    while length(tasks) > 0
        o = popfirst!(tasks)
        for (e, cost) in equipment_costs
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
                equip_needed = true
                break
            end
        end
        if !equip_needed
        #account for equipment being removed
            capabilities = reduced_cap
        end
    end
    return filtered_equip, capabilities
end



function greedy_equipment_assignment_heuristic(instance::MALBP_W_instance, x_soi::Array{Int,3})
    #assigns the equipment to the stations
    equipment_assignments = Dict{Int, Vector{Int64}}()
    
    for station in 1:instance.equipment.no_stations
        x_soi_station = x_soi[station, :, :]
        assigned_tasks = sum(x_soi_station, dims=2)
        assigned_tasks = dropdims(assigned_tasks, dims=2)
        #assigned tasks are all indices of nonzero elements
        assigned_tasks = findall(x->x>0, assigned_tasks)
        #changes assigned tasks to strings
        (equip_station_assignment, _) = greedy_set_cover(assigned_tasks, instance, station)
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
    x_soi = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks, instance.models.no_models)
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
    y, y_w, y_wts = worker_assignment_heuristic(instance, x_soi)
    
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
    for (w,scenario) in enumerate(eachrow(instance.scenarios))
        probability = scenario.probability
        recourse_cost += y_w[w] * instance.recourse_cost * probability
    end
    return main_worker_cost + recourse_cost
end

# function search_w_equip(instance::MALBP_W_instance, equipment_assignments::Dict{Int, Vector{Int64}}, model::ModelInstance, precedence_matrix::Array{Int,2}, index_to_task::Dict{Int, String})
#     #assigns the equipment to the stations
#     new_cycle_time = calculate_new_cycle_time(model, instance.equipment.no_stations, instance.models.cycle_time)
#     #creates a vector of remaining time left in the station. The length is the number of stations, and the value is the cycle time
#     remaining_time = fill(new_cycle_time, instance.equipment.no_stations)
#     station_assign = Dict{Int, Array{String,1}}()
#     capabilities_so = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks)
#     for station in 1:instance.equipment.no_stations
#         station_assign[station] = []
#     end
#     station_no = 1
#     r_oe = permutedims((stack(instance.equipment.r_oe)), [2,1])
#     println("r_oe: ", r_oe)
#     while any(precedence_matrix[end, :] .>= 0)
#         for equipment in equipment_assignments[station_no]
#             println("equipment: ", equipment)
#         end
#         #get the first zero element in the last row
#         first_zero = findfirst(precedence_matrix[end, :] .== 0)
#         #get the task name of the first zero element
#         task = index_to_task[first_zero]
#         #subtracts the task time from the remaining time of the first station with a positive balance
#         station_no = findfirst(remaining_time .> 0)
#         remaining_time[station_no] -= model.task_times[1][task]
#         #subtracts the task from the "code"
#         precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[first_zero, :]
#         #marks the task as complete
#         precedence_matrix[end, first_zero] = -1
#         #records the station assignment for the task
#         push!(station_assign[station_no], task)
#     end
#     return station_assign
# end

# #This function uses ehsan's heuristic for first model and then tries to assign the rest of the models with the equipment assignments
# function ehsan_then_search(instance::MALBP_W_instance; order_function::Function = positional_weight_order, reverse::Bool=false)
#     models = [model for (model_name, model) in instance.models.models]
#     #orders the models by decreasing probability
#     models = sort(models, by=x->x.probability, rev=true)
#     model1 = popfirst!(models)
#     precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model1; order_function= order_function)
#     model_task_assignments = Dict{String, Dict{Int, Vector{String}}}()
#     model_assignments = ehsans_task_assign(instance, model1, precedence_matrix, index_to_task)
#     model_task_assignments[model1.name] = model_assignments
#     equipment_assignments = greedy_equipment_assignment_heuristic(instance, model_task_assignments)
#     for model in models
#         precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model; order_function= equip_positional_weight_order, 
#                                                                                             equipment_assignments = equipment_assignments, 
#                                                                                             equipment = instance.equipment,
#                                                                                             reverse=reverse)
#         model_assignments, equipment_assignments = search_w_equip(instance, equipment_assignments, model, precedence_matrix, index_to_task)
#         model_task_assignments[model.name] = model_assignments
#     end
#     y, y_w, y_wts = worker_assignment_heuristic(instance, model_task_assignments)
    
#     return model_task_assignments, y, y_w, y_wts, equipment_assignments
# end

function calculate_c_time_si(instance::MALBP_W_instance)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    c_time_si = zeros(Float64, instance.equipment.no_stations, instance.models.no_models)
    for station in 1:instance.equipment.no_stations
        for (model_name, model) in instance.models.models
            i = findfirst( ==(model_name), model_indexes)
            c_time_si[station, i] = calculate_new_cycle_time(model, instance.equipment.no_stations, instance.models.cycle_time)
        end
    end
    return c_time_si
end


function fill_station!(instance::MALBP_W_instance,
                        model_unfinished::Vector{Bool},
                        station::Int, 
                        models::Vector{Tuple{ModelInstance, Int64}}, 
                        c_time_si::Array{Float64, 2}, 
                        x_soi::Array{Int64, 3}, 
                        equipment_assignments::Dict{Int64, Vector{Int64}}, 
                        capabilities_so::Array{Int,2}, 
                        precedence_matrices::Dict{String, Dict})
    dumb_dumb = 0
    while any(c_time_si[station,:] .> 0) && any(model_unfinished) && dumb_dumb< 100
        println("model unfinished: ", model_unfinished)
        for (model, i) in models
            #skip if there is no capacity at the station or the model is finished
            if c_time_si[station, i] <= 0 || !model_unfinished[i]
                continue
            end
            precedence_matrix = precedence_matrices[model.name]["precedence_matrix"]
            index_to_task = precedence_matrices[model.name]["index_to_task"]
            #available_tasks = findall(precedence_matrix[end, :] .== 0)
            progress = false
            for (task, value) in enumerate(precedence_matrix[end, :])
                if capabilities_so[station, task] > 0 && value == 0
                    c_time_si[station, i] -= model.task_times[1][index_to_task[task]]
                    precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[task, :]
                    #marks the task as complete
                    precedence_matrix[end, task] = -1
                    progress = true
                    #equipment task indexes are different from the indexes in the precedence matrix
                    o = parse(Int, index_to_task[task])
                    x_soi[station, o, i] = 1
                    if c_time_si[station, i] <= 0
                        break
                    end
                end
            end
            #adds a task and recalculates the equipment to accomodate it
            if !progress
                task = findfirst(precedence_matrix[end, :] .== 0)
                if isnothing(task)
                    println("model unfinished", model_unfinished)
                    println("i: ", i)
                    println("no task found")
                    println("precedence_matrix: ", precedence_matrix[end, :])
                end
                c_time_si[station, i] -= model.task_times[1][index_to_task[task]]
                precedence_matrix[end, :] = precedence_matrix[end, :] .- precedence_matrix[task, :]
                #marks the task as complete
                precedence_matrix[end, task] = -1
                x_soi[station, task, i] = 1
                x_soi_station = x_soi[station, :, :]
                assigned_tasks = sum(x_soi_station, dims=2)
                assigned_tasks = dropdims(assigned_tasks, dims=2)
                #assigned tasks are all indices of nonzero elements
                assigned_tasks = findall(x->x>0, assigned_tasks)
                equipment_assignment, capabilities_so[station, :] = greedy_set_cover(assigned_tasks, instance, station)
                equipment_assignments[station] = equipment_assignment
            end 
            println("precedence matrix end", precedence_matrix[end, :])
            println("all done: ", all(precedence_matrix[end, :] .< 0))
            if all(precedence_matrix[end, :] .< 0)
                model_unfinished[i] = false
            end
            println(" model unfinished: ", model_unfinished)
        end
        dumb_dumb += 1
    end
end

function task_equip_heuristic(instance::MALBP_W_instance; order_function::Function = positional_weight_order)
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks)
    r_oe = permutedims((stack(instance.equipment.r_oe)), [2,1])
    c_time_si = calculate_c_time_si(instance)
    x_soi = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks, instance.models.no_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    models_unfinished = fill(true, instance.models.no_models)
    for station in 1:instance.equipment.no_stations
        fill_station!(instance, models_unfinished, station, models, c_time_si, x_soi, equipment_assignments, capabilities_so,  precedence_matrices)
    end
    y, y_w, y_wts = worker_assignment_heuristic(instance, x_soi)
    return x_soi, y, y_w, y_wts, equipment_assignments
end


function task_equip_heuristic2(instance::MALBP_W_instance; order_function::Function = positional_weight_order)
    precedence_matrices = create_precedence_matrices(instance; order_function= order_function)
    #orders the models by decreasing probability
    models = [(model, index) for (index,(model_name, model)) in enumerate(instance.models.models)]
    models = sort(models, by=x->x[1].probability, rev=true)
    capabilities_so = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks)
    r_oe = permutedims((stack(instance.equipment.r_oe)), [2,1])
    c_time_si = calculate_c_time_si(instance)
    x_soi = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks, instance.models.no_models)
    equipment_assignments = Dict{Int, Vector{Int64}}()
    models_unfinished = fill(true, instance.models.no_models)
    for station in 1:instance.equipment.no_stations
        fill_station!(instance, models_unfinished, station, models, c_time_si, x_soi, equipment_assignments, capabilities_so,  precedence_matrices)
    end
    y, y_w, y_wts = worker_assignment_heuristic(instance, x_soi)
    return x_soi, y, y_w, y_wts, equipment_assignments
end

#config_filepath = "SALBP_benchmark/MM_instances/julia_debug.yaml"
 config_filepath = "SALBP_benchmark/MM_instances/medium_instance_config_S10.yaml"
instances = read_MALBP_W_instances(config_filepath)
instance = instances[2]

# x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
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
results = []
for instance in instances
    model_task_assignments, y, y_w, y_wts, equipment_assignments = sequential_heuristic_start_md(instance; order_function = reverse_positional_weight_order)
    seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
    seq_worker_cost = calculate_worker_cost(y, y_w, instance)
    rev_seq_total = seq_equip_cost + seq_worker_cost

    model_task_assignments, y, y_w, y_wts, equipment_assignments = sequential_heuristic_start_md(instance)
    seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
    seq_worker_cost = calculate_worker_cost(y, y_w, instance)
    seq_total = seq_equip_cost + seq_worker_cost

    model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance)
    two_step_equip_cost = calculate_equip_cost(equipment_assignments, instance)
    two_step_worker_cost = calculate_worker_cost(y, y_w, instance)
    two_step_total = two_step_equip_cost + two_step_worker_cost


    model_task_assignments, y, y_w, y_wts, equipment_assignments = two_step_ehsans(instance; order_function = reverse_positional_weight_order, reverse=true)
    two_step_equip_cost = calculate_equip_cost(equipment_assignments, instance)
    two_step_worker_cost = calculate_worker_cost(y, y_w, instance)
    rev_two_step_total = two_step_equip_cost + two_step_worker_cost

    x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
    seq_equip_cost = calculate_equip_cost(equipment_assignments, instance)
    seq_worker_cost = calculate_worker_cost(y, y_w, instance)
    task_equip_total = seq_equip_cost + seq_worker_cost

    push!(results, (instance.name, seq_total, two_step_total ,rev_seq_total, rev_two_step_total, task_equip_total))
end
#turns results to DataFrame
results_df = DataFrame(results, [:instance, :seq_total, :two_step_total, :rev_seq_total, :rev_two_step_total, :task_equip_total])
println(results_df)