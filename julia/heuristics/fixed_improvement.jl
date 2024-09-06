include("fixed_constructive.jl")
include("improvement.jl")

#For each task generates the list of tasks that are depending on it, and the list of tasks that it depends on
function precedence_relations_dict(instance::MALBP_W_instance)
    predecessors = Dict{String, Dict{String, Vector{String}}}()
    successors = Dict{String, Dict{String, Vector{String}}}()
    for (model_name, model) in instance.models.models
        predecessors[model_name] = Dict{String, Vector{String}}()
        successors[model_name] = Dict{String, Vector{String}}()
        for (task_name , task) in model.task_times[1]
            predecessors[model_name][task_name] = []
            successors[model_name][task_name] = []
            for (pred, suc) in model.precendence_relations
                if pred == task_name
                    push!(successors[model_name][task_name], suc)
                end
                if suc == task_name
                    push!(predecessors[model_name][task_name], pred)
                end
            end
        end
    end
    return predecessors, successors
end


function calculate_left(x_soi::Array{Int, 3}, o::Int, i::Int, model::String, predecessors::Dict{String, Dict{String, Vector{String}}}; _...)
    left = 1
    for pred in predecessors[model][String(task)]
        pred_station = findfirst(x>0,x_soi[:,parse(Int, pred),i])
        if pred_station > left
            left = pred_station
        end
    end
    return left
end

function calculate_right(x_soi::Array{Int, 3}, o::Int, i::Int, model::String,  successors::Dict{String, Dict{String, Vector{String}}}; max_stations::Int)
    right = max_stations
    for suc in successors[model][String(o)]
        suc_station = findfirst(x>0,x_soi[:,parse(Int, suc),i])
        if suc_station < right
            right = suc_station
        end
    end
    return right
end



# function calculate_left(x_os::Dict{String, Dict{String, Int}}, task::String, model::String, predecessors::Dict{String, Dict{String, Vector{String}}}; _...)
#     left = 1
#     for pred in predecessors[model][task]
#         pred_station = x_os[model][pred]
#         if pred_station > left
#             left = pred_station
#         end
#     end
#     return left
# end

# function calculate_right(x_os::Dict{String, Dict{String, Int}}, task::String, model::String, successors::Dict{String, Dict{String, Vector{String}}}; max_stations::Int)
#     right = max_stations
#     for suc in successors[model][task]
#         suc_station = x_os[model][suc]
#         if suc_station < right
#             right = suc_station
#         end
#     end
#     return right
# end
#Calculates the number of workers needed at a station, returns nothing if the tasks cannot be completed
function necessary_workers_w_block(tasks::Vector{String}, cycle_time::Real, model::ModelInstance, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    #calculates the number of workers needed to complete the tasks
    remaining_time = sum([model.task_times[1][task] for task in tasks])
    for (worker, productivity) in enumerate(productivity_per_worker)
        available_task_time = cycle_time * productivity
        remaining_time -= available_task_time
        if remaining_time <= 0
            return worker
        end
    end
    return nothing
end

#Calculates the number of workers needed at a station, returns nothing if the tasks cannot be completed
function necessary_workers_w_block(tasks::Vector{Int}, cycle_time::Real, model::ModelInstance, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    #calculates the number of workers needed to complete the tasks
    remaining_time = sum([model.task_times[1][String(task)] for task in tasks])
    for (worker, productivity) in enumerate(productivity_per_worker)
        available_task_time = cycle_time * productivity
        remaining_time -= available_task_time
        if remaining_time <= 0
            return worker
        end
    end
    return nothing
end





function opt1_insertion_w_equip!(x_soi::Array{Int,3}, s::Int, left::Int, right::Int, o::Int, instance::MALBP_W_instance, equipment_assignments::Dict{Int, Vector{Int64}}, equip_costs::Vector, productivity_per_worker::Vector{Float64}; i =1, model::String = "combined_model")
    original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s,:,:], dims=(2)), dims=2))
    x_soi[s,o,i] = 0
    new_tasks = Stringfindall(x-> x > 0, dropdims(sum(x_soi[s,:,:], dims=(2)), dims=2))
    y_s = necessary_workers_w_block(original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
    y_s_without_task = necessary_workers_w_block(new_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker= productivity_per_worker)
    for s_prime in left:right
        s_prime_original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s_prime,:,:], dims=(2)), dims=2))
        x_soi[s_prime,o, i] = 1
        s_prime_new_tasks = [s_prime_original_tasks; task]
        y_s_prime = necessary_workers_w_block( s_prime_original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
        new_y_sprime = necessary_workers_w_block(s_prime_new_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
        #Check if task assignment respects cycle_time and worker constraints
        if isnothing(new_y_sprime)
            x_soi[s_prime,o, i] = 0
            x_soi[s,o,i] = 1
            continue
        end

        #calculation of equipment costs of swap
        old_cost_s = equip_costs[s]
       #println(old_cost_s)
        old_cost_s_prime = equip_costs[s_prime]
        #println(old_cost_s_prime)
        println("y_s_without_task: ", y_s_without_task, "new_y_sprime: ", new_y_sprime)
        new_workers = y_s_without_task + new_y_sprime
        old_workers = y_s + y_s_prime   
        new_equip_s, _, new_cost_s = greedy_set_cover_v2(new_tasks, instance, s)
        new_equip_s_prime, _, new_cost_s_prime = greedy_set_cover_v2(s_prime_new_tasks, instance, s_prime)
        #new cost is the workers at the stations and the equipment
        new_cost = instance.worker_cost * new_workers + new_cost_s + new_cost_s_prime
        old_cost = instance.worker_cost * old_workers + old_cost_s + old_cost_s_prime
        #println("NEW COST: ", new_cost)
        #println("OLD COST: ", old_cost)
        if new_cost < old_cost
            #changing tasks
            #moves task to new station
            #remove the task from the previous station
            # for model in keys(instance.models.models)
            #     push!(x_so[model][s_prime], task)
            #     x_so[model][station] = new_tasks
            #     x_os[model][task] = station
            # end
            #changing equipment
            equip_costs[station] = new_cost_s
            equip_costs[s_prime] = new_cost_s_prime
            equipment_assignments[station] = new_equip_s
            equipment_assignments[s_prime] = new_equip_s_prime
            return true
        else
            x_soi[s_prime,o, i] = 0
            x_soi[s,o,i] = 1
        end 

    end
    return false
end

function calculate_station_equip_cost(equipment_assignments::Dict{Int, Vector{Int64}}, station::Int, instance::MALBP_W_instance)
    #calculates the total cost of the equipment assignments
    station_cost = 0
    println("station assignments: ", equipment_assignments[station])

    for equip in equipment_assignments[station]
        station_cost += instance.equipment.c_se[station][equip]
    end
    println("station cost: ", station_cost)
    return station_cost
end

function calculate_equip_cost_per_station(equipment_assignments::Dict{Int, Vector{Int64}},instance::MALBP_W_instance)
    station_costs = zeros(Float64, instance.equipment.n_stations)
    for station in 1:instance.equipment.n_stations
        station_costs[station] = calculate_station_equip_cost(equipment_assignments, station, instance)
    end
    return station_costs
end

function task_1opt_fixed(instance::MALBP_W_instance, x_soi::Array{Int,3}, equipment_assignments; n_iterations::Int=100, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], model_name::String="combined_model")
    equip_costs = calculate_equip_cost_per_station(equipment_assignments, instance)
    predecessors, successors = precedence_relations_dict(instance)
    counter = 0
    improvement = false
    while counter < n_iterations
            for (s, o) in findall(a-> a > 0, x_soi[:,:,1])
                left = calculate_left(x_soi, o, 1, model_name, predecessors)
                improvement = opt1_insertion_w_equip!( x_soi, s, left, s-1, o, instance , equipment_assignments, equip_costs, productivity_per_worker)
                if improvement
                    continue
                end
                right = calculate_right(x_soi, o,1, model_name, successors, max_stations = instance.equipment.n_stations)
                improvement = opt1_insertion_w_equip!(x_soi, s, s+1, right, o,  instance, equipment_assignments, equip_costs, productivity_per_worker)
                if improvement
                    println("Improvement !")
                end
        end
        counter += 1
    end


    return x_so, equipment_assignments
end 

function x_soi_to_dict(instance::MALBP_W_instance, x_soi::Array{Int,3})
    model_task_assignments_so = Dict{String, Dict{Int, Vector{String}}}()
    model_task_assignments_os = Dict{String, Dict{String, Int}}()
    for (i,(model_name, model)) in enumerate(instance.models.models)
        model_task_assignments_so[model_name] = Dict{Int, Vector{String}}()
        model_task_assignments_os[model_name] = Dict{String, Int}()
        for station in 1:instance.equipment.n_stations
            model_task_assignments_so[model_name][station] = []
            for (task, _) in instance.models.models[model_name].task_times[1]
                if x_soi[station, parse(Int, task), i] == 1
                    model_task_assignments_os[model_name][task] = station
                    push!(model_task_assignments_so[model_name][station], task)
                end
            end
        end
    end
    return model_task_assignments_so, model_task_assignments_os
end


function get_tasks_assigned_from_xsoi(x_soi, station::Int)
    station_assignments = dropdims(sum(x_soi[station,:,:], dims = (2)), dims = 2)
    station_assignments = findall(x -> x > 0 , station_assignments)
    #string_vector = map(string, station_assignments)
    return station_assignments
end


function construct_then_improve(orig_instance::MALBP_W_instance, set_cover_heuristic::Function=greedy_set_cover_v2, productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.],n_iterations=100)
    x_so, equipment_assignments = task_equip_heuristic_combined_precedence_fixed(orig_instance,  set_cover_heuristic=set_cover_heuristic)
    x_so, equipment_assignments = task_1opt_fixed(instance, x_so, equipment_assignments, n_iterations = n_iterations, productivity_per_worker=productivity_per_worker)
    x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, orig_instance.models.n_models)
    for model in 1:orig_instance.models.n_models
        x_soi[:,:,model] = x_so[:,:,1]
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(orig_instance, x_soi, productivity_per_worker = productivity_per_worker)
end

#config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/constructive_debug.yaml"
config_filepath = "SALBP_benchmark/MM_instances/xp_yaml/medium_instance_config_S10.yaml"
instances = read_MALBP_W_instances(config_filepath)



instance = instances[2]
println("Min workers: ", calculate_min_workers(instance, start_station = 1, end_station = 1, tasks=[ "7"]))
x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic_combined_precedence(instance, set_cover_heuristic=greedy_set_cover_v2)
total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
println("original cost: ", total_cost)
# assigned_tasks = get_tasks_assigned_from_xsoi(x_soi,2)
# println("These are the assigned tasks:", assigned_tasks)

x_soi, y, y_w, y_wts, equipment_assignments = task_1opt(instance, x_soi, equipment_assignments, n_iterations= 1000)
total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
println("cost after 1 opt: ", total_cost)
print("x_soi", sum(x_soi, dims=(2,3)))
# new_equip_assign, new_capabilities = greedy_set_cover_v2(assigned_tasks, instance, 2)
# println("new equip assignment", new_equip_assign)
# #task_location, station_capababilities = get_equipment_capable_stations(instance, equipment_assignments)
# task_times = get_task_assignment_times_x_wsot(instance, x_soi)

# println("equipment_assignments: ", equipment_assignments)
# println("task_location: ", task_location)
# println("station_capababilities: ", station_capababilities)
# println("task_times: ", task_times)