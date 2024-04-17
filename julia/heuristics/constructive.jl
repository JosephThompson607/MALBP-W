include("../read_MALBP_W.jl")
include("preprocessing.jl")

#orders the tasks for a model in descending order based on positional weight
#returns a list of tuples with the positional weight as the first element and the task name as the second element
function positional_weight_order(model::ModelInstance)
    dependency_dict = calculate_positional_weight(model)
    #println("depedency dict ", dependency_dict)
    ordered_keys = sort(collect(zip(values(dependency_dict),keys(dependency_dict))), rev=true)
   # println("ordered keys ", ordered_keys)
    return ordered_keys
end

#Calculates the precedence matrix for each model in the instance
#The bottom row of the matrix is the "code", showing the number of immediate predecessors for each task
function create_precedence_matrix(instance::MALBP_W_instance; order_function::Function)
    matrix_dict = Dict{String, Dict}()
    for (model_name, model) in instance.models.models
        task_to_index = Dict{String, Int}()
        index_to_task = Dict{Int, String}()
        #sorts the tasks by the order function. This will later correspond to the precedence matrix
        ordered_keys = order_function(model)
        for (i,(_,task)) in enumerate(ordered_keys)
            task_to_index[task] = i
            index_to_task[i] = task
        end
        precedence_matrix = zeros(Int, model.no_tasks+1, model.no_tasks)
        
        for (pred, suc) in model.precendence_relations
            pred_index = task_to_index[pred]
            suc_index = task_to_index[suc]
            precedence_matrix[pred_index, suc_index] = 1
        end
        #final row is the sum of the previous rows
        precedence_matrix[end, :] = sum(precedence_matrix, dims=1)
        # println("precedence matrix: ", model_name)
        # #prints each row of the precedence matrix
        # for i in 1:(model.no_tasks + 1)
        #     if i <= model.no_tasks
        #         println(index_to_task[i], precedence_matrix[i, :])
        #     else
        #     println(precedence_matrix[i, :])
        #     end
        # end
        matrix_dict[model_name] = Dict("precedence_matrix" =>precedence_matrix, "task_to_index" => task_to_index, "index_to_task" => index_to_task)
        
        
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
    precedence_matrix_dict = create_precedence_matrix(instance)


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
    station_assign = Dict{Int, Array{String,1}}()
    for station in 1:instance.equipment.no_stations
        station_assign[station] = []
    end
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
        #records the station assignment for the task
        push!(station_assign[station_no], task)
    end
    return station_assign
end

function ehsans_heuristic(instance::MALBP_W_instance; order_function::Function = positional_weight_order)
    precedence_matrices = create_precedence_matrix(instance; order_function= order_function)
    model_task_assignments = Dict{String, Dict{Int, Vector{String}}}()
    for (model_name, model) in instance.models.models
        precedence_matrix = precedence_matrices[model_name]["precedence_matrix"]
        index_to_task = precedence_matrices[model_name]["index_to_task"]
        model_task_assignments[model_name] = ehsans_task_assign(instance, model, precedence_matrix, index_to_task)
    end
    return model_task_assignments
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

function base_worker_assign_func(instance::MALBP_W_instance, model_task_assignments::Dict{String, Dict{Int, Vector{String}}}; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.])
    assign_matrix = zeros(Int, instance.no_scenarios, instance.no_cycles, instance.equipment.no_stations )
    for (w,scenario) in enumerate(eachrow(instance.scenarios))
        for (j, model) in enumerate(scenario.sequence)
            for s in 1:instance.equipment.no_stations
                t = j + s - 1
                #gets the minimum number of workers needed to complete the tasks
                assign_matrix[w, t, s] = necessary_workers(model_task_assignments[model][s], 
                                                            instance.models.cycle_time, instance.models.models[model], 
                                                            productivity_per_worker)

            end
        end
    end
    return assign_matrix
end

function worker_assignment_heuristic(instance::MALBP_W_instance, model_task_assignments::Dict{String, Dict{Int, Vector{String}}}; productivity_per_worker::Array{Float64}= [1., 1., 1., 1.])
    worker_station_assignments = base_worker_assign_func(instance, model_task_assignments, productivity_per_worker = productivity_per_worker)
    println("worker station assignments: ", worker_station_assignments[1,1,1])
    y_w = zeros(Int, instance.no_scenarios)
    y = 0
    for (w, scenario) in enumerate(eachrow(instance.scenarios))
        for t in 1:instance.no_cycles
            needed_workers = sum(worker_station_assignments[w, t, :])
            y = max(y, needed_workers)
        end
    end
    return y, y_w, worker_station_assignments
end

function greedy_set_cover(tasks_to_assign::Vector{String}, instance::MALBP_W_instance, station::Int)
    #if no tasks to assign, return empty list
    if length(tasks_to_assign) == 0
        return []
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
    while length(tasks) > 0
        task = popfirst!(tasks)
        o = parse(Int, task)
        for (e, cost) in equipment_costs
            if instance.equipment.r_oe[o][ e] == 1
                push!(equipment_assignments, e)
                #removes the tasks that are covered by the equipment from the tasks
                tasks = filter(x->r_oe[parse(Int, x), e] == 0, tasks)
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
            if reduced_cap[parse(Int, task)] <= 0
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
    return filtered_equip
end

function greedy_equipment_assignment_heuristic(instance::MALBP_W_instance, model_task_assignments::Dict{String, Dict{Int, Vector{String}}})
    #assigns the equipment to the stations
    equipment_assignments = Dict{Int, Vector{Int64}}()
    merged_tasks = Dict{Int, Vector{String}}()
    for (model, station_assignments) in model_task_assignments
        merged_tasks = mergewith( vcat, merged_tasks, station_assignments)
    end
    for (station, tasks) in merged_tasks
        merged_tasks[station] = unique(tasks)
        equip_station_assignment = greedy_set_cover(merged_tasks[station], instance, station)
        equipment_assignments[station] = equip_station_assignment
    end
    return equipment_assignments
end
config_filepath = "SALBP_benchmark/MM_instances/julia_debug.yaml"
# # # #config_filepath = "SALBP_benchmark/MM_instances/medium_instance_config_S10.yaml"
instance = read_MALBP_W_instances(config_filepath)[2]

model_task_assignments =ehsans_heuristic(instance)

# println("model task assignment: ", model_task_assignments)

# # #worker_assignment_matrix = base_worker_assign_func(instance, model_task_assignments)

# # y, y_w, worker_station_assignments = worker_assignment_heuristic(instance, model_task_assignments)
# # println("total workers needed: ", y)
greedy_equipment_assignment_heuristic(instance, model_task_assignments)