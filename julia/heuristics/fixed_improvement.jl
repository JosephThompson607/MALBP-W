# include("fixed_constructive.jl")
# include("improvement.jl")

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
    for pred in predecessors[model][string(o)]
        
        pred_station = findfirst(x-> x > 0,x_soi[:,parse(Int, pred),i])
        #if the predecessor is not there, we should return the original station
        if isnothing(pred_station)
            return findfirst(x-> x > 0,x_soi[:,o,i])
        elseif pred_station > left
            left = pred_station
        end
    end
    return left
end

function calculate_right(x_soi::Array{Int, 3}, o::Int, i::Int, model::String,  successors::Dict{String, Dict{String, Vector{String}}}; max_stations::Int)
    right = max_stations
    for suc in successors[model][string(o)]
        suc_station = findfirst(x-> x >0,x_soi[:,parse(Int, suc),i])
        #if the predecessor is not there, we should return the original station
        if isnothing(suc_station)
            return findfirst(x-> x > 0,x_soi[:,o,i])
        elseif suc_station < right
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
function necessary_workers_w_block(tasks::Vector{Int}, cycle_time::Real, model::ModelInstance; productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.])
    #calculates the number of workers needed to complete the tasks
    remaining_time = sum([model.task_times[1][string(task)] for task in tasks])
    for (worker, productivity) in enumerate(productivity_per_worker)
        available_task_time = cycle_time * productivity
        remaining_time -= available_task_time
        if remaining_time <= 0
            return worker
        end
    end
    return nothing
end


function opt2_task_swap!(x_soi::Array{Int,3}, s::Int64, s_prime::Int64, o::Int64, o_prime::Int64, i::Int64, instance::MALBP_W_instance, original_tasks::Vector{Int64}, s_prime_original_tasks::Vector{Int64}, model::String, productivity_per_worker::Vector{Float64} )
    x_soi[s_prime,o, i] = 1
    x_soi[s_prime, o_prime, i] = 0
    x_soi[s, o_prime, i] = 1
    #removes the old tasks
   new_s_tasks  = filter(x -> x != o, original_tasks)
    new_s_prime_tasks = filter(x-> x != o_prime, s_prime_original_tasks)
    #adds the new tasks, Taking union bc sometimes the task is already there in md case
    new_s_prime_tasks = union(new_s_prime_tasks, o)
    new_s_tasks = union(new_s_tasks, o_prime)
    new_y_s_prime = necessary_workers_w_block(new_s_prime_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
    new_y_s = necessary_workers_w_block(new_s_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
    #Check if task assignment respects cycle_time and worker constraints
    if isnothing(new_y_s_prime) || isnothing(new_y_s)
        return nothing, nothing, nothing, nothing
    end
    return new_y_s, new_y_s_prime, new_s_tasks, new_s_prime_tasks

end


function evaluate_swap!(instance, equip_costs, s, s_prime, y_s, y_s_prime, new_y_s, new_y_s_prime, new_s_tasks, new_s_prime_tasks)
    old_cost_s = equip_costs[s]
    old_cost_s_prime = equip_costs[s_prime]
    new_workers = new_y_s + new_y_s_prime
    old_workers = y_s + y_s_prime   
    new_equip_s, _, new_cost_s = greedy_set_cover_v2(new_s_tasks, instance, s)
    new_equip_s_prime, _, new_cost_s_prime = greedy_set_cover_v2(new_s_prime_tasks, instance, s_prime)
    #new cost is the workers at the stations and the equipment
    new_cost = instance.worker_cost * new_workers + new_cost_s + new_cost_s_prime
    old_cost = instance.worker_cost * old_workers + old_cost_s + old_cost_s_prime
    if new_cost < old_cost
        return true, new_equip_s, new_equip_s_prime, new_cost_s, new_cost_s_prime
    else
        return false, nothing, nothing, nothing, nothing
    end
end


function opt2_insertion_w_equip!(x_soi::Array{Int,3}, s::Int, left::Int, right::Int, o::Int, instance::MALBP_W_instance, equipment_assignments::Dict{Int, Vector{Int64}}, equip_costs::Vector, productivity_per_worker::Vector{Float64}; i =1, model::String = "combined_model", predecessors, successors)
    original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s,:,:], dims=(2)), dims=2))
    y_s = necessary_workers_w_block(original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
    x_soi[s,o,i] = 0
    for s_prime in left:right
        s_prime_original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s_prime,:,:], dims=(2)), dims=2))
        y_s_prime = necessary_workers_w_block( s_prime_original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
        for o_prime in s_prime_original_tasks
            #check if it is viable to swap the two tasks at the station
            if left < s && s < calculate_right(x_soi, o_prime, i, model, successors, max_stations= s)
                new_y_s ,new_y_s_prime, new_s_tasks, new_s_prime_tasks = opt2_task_swap!(x_soi, s, s_prime, o, o_prime, i, instance, original_tasks, s_prime_original_tasks, model, productivity_per_worker )
            elseif right > s && s > calculate_left(x_soi, o_prime, i , model, predecessors)
                new_y_s, new_y_s_prime, new_s_tasks, new_s_prime_tasks  = opt2_task_swap!(x_soi, s, s_prime, o, o_prime, i, instance, original_tasks, s_prime_original_tasks, model, productivity_per_worker )
            else
                #No eligible tasks to swap
                continue
            end

            #checks to make sure the swap was feasible, if not feasible, resets assignments
            if isnothing(new_y_s) || isnothing(new_y_s_prime) 
                #resets the assignments back to what they were
                x_soi[s, o_prime, i] = 0
                x_soi[s_prime, o_prime, i] = 1
                x_soi[s_prime,o, i] = 0
                x_soi[s,o,i] = 0
                continue
            
            elseif !isnothing(y_s) && !isnothing(y_s_prime)
                accept, new_equip_s, new_equip_s_prime, new_cost_s, new_cost_s_prime = evaluate_swap!(instance, equip_costs, s, s_prime, y_s, y_s_prime, new_y_s, new_y_s_prime, new_s_tasks, new_s_prime_tasks)
                if accept
                    equip_costs[s] = new_cost_s
                    equip_costs[s_prime] = new_cost_s_prime
                    equipment_assignments[s] = new_equip_s
                    equipment_assignments[s_prime] = new_equip_s_prime
                    return true
                      
                else
                    #resets the assignments back to what they were
                    x_soi[s, o_prime, i] = 0
                    x_soi[s_prime, o_prime, i] = 1
                    x_soi[s_prime,o, i] = 0
                    x_soi[s,o,i] = 0
                    continue
                end 
                #previous assignments were infeasible
            else
                new_equip_s, _, new_cost_s = greedy_set_cover_v2(new_s_tasks, instance, s)
                new_equip_s_prime, _, new_cost_s_prime = greedy_set_cover_v2(new_s_prime_tasks, instance, s_prime)
                equip_costs[s] = new_cost_s
                equip_costs[s_prime] = new_cost_s_prime
                equipment_assignments[s] = new_equip_s
                equipment_assignments[s_prime] = new_equip_s_prime
                return true
            end

            #calculation of equipment costs of swap

          
        end
    end
    #sets x_soi back to the original value
    x_soi[s,o,i] = 1
    return false
end


function opt1_insertion_w_equip!(x_soi::Array{Int,3}, s::Int, left::Int, right::Int, o::Int, instance::MALBP_W_instance, equipment_assignments::Dict{Int, Vector{Int64}}, equip_costs::Vector, productivity_per_worker::Vector{Float64}; i =1, model::String = "combined_model", relax_worker_infeasibility=true, kwargs...)
    original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s,:,:], dims=(2)), dims=2))
    x_soi[s,o,i] = 0
    new_s_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s,:,:], dims=(2)), dims=2))
    y_s = necessary_workers_w_block(original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
    #In some edge cases, can be passed a solution that is infeasible with the combined precedence diagram approach, if we relax worker relax_worker_infeasibility, we just move on to the next station
    if isnothing(y_s) && relax_worker_infeasibility
        x_soi[s,o,i] = 1
        return false
    end
    new_y_s = necessary_workers_w_block(new_s_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker= productivity_per_worker)
    for s_prime in left:right
        s_prime_original_tasks = findall(x-> x > 0, dropdims(sum(x_soi[s_prime,:,:], dims=(2)), dims=2))
        x_soi[s_prime,o, i] = 1
        #Taking union bc sometimes the task is already there
        new_s_prime_tasks = union(s_prime_original_tasks, o)
        y_s_prime = necessary_workers_w_block( s_prime_original_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
        new_y_s_prime = necessary_workers_w_block(new_s_prime_tasks, instance.models.cycle_time, instance.models.models[model], productivity_per_worker=productivity_per_worker)
        #Check if task assignment respects cycle_time and worker constraints
        if isnothing(new_y_s_prime)
            x_soi[s_prime,o, i] = 0
            continue
        end
        accept, new_equip_s, new_equip_s_prime, new_cost_s, new_cost_s_prime = evaluate_swap!(instance, equip_costs, s, s_prime, y_s, y_s_prime, new_y_s, new_y_s_prime, new_s_tasks, new_s_prime_tasks)
        if accept
            equip_costs[s] = new_cost_s
            equip_costs[s_prime] = new_cost_s_prime
            equipment_assignments[s] = new_equip_s
            equipment_assignments[s_prime] = new_equip_s_prime
            return true
        else
            x_soi[s_prime,o, i] = 0
        end 
    end
    #sets x_soi back to the original value
    x_soi[s,o,i] = 1
    return false
end

function calculate_station_equip_cost(equipment_assignments::Dict{Int, Vector{Int64}}, station::Int, instance::MALBP_W_instance)
    #calculates the total cost of the equipment assignments
    station_cost = 0
    for equip in equipment_assignments[station]
        station_cost += instance.equipment.c_se[station][equip]
    end
    return station_cost
end

function calculate_equip_cost_per_station(equipment_assignments::Dict{Int, Vector{Int64}},instance::MALBP_W_instance)
    station_costs = zeros(Float64, instance.equipment.n_stations)
    for station in 1:instance.equipment.n_stations
        station_costs[station] = calculate_station_equip_cost(equipment_assignments, station, instance)
    end
    return station_costs
end

function task_opt_fixed(instance::MALBP_W_instance, 
                        x_soi::Array{Int,3}, 
                        equipment_assignments; 
                        n_iterations::Int=100, 
                        productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], 
                        model_name::String="combined_model", 
                        opt_function!::Function=opt1_insertion_w_equip!)
    equip_costs = calculate_equip_cost_per_station(equipment_assignments, instance)
    predecessors, successors = precedence_relations_dict(instance)
    counter = 0
    improvement = false
    while counter < n_iterations
            for (s, o) in Tuple.(findall(a-> a > 0, x_soi[:,:,1]))
                left = calculate_left(x_soi, o, 1, model_name, predecessors)
                improvement = opt_function!( x_soi, s, left, s-1, o, instance , equipment_assignments, equip_costs, productivity_per_worker; predecessors=predecessors, successors=successors)
                if improvement
                    continue
                end
                right = calculate_right(x_soi, o,1, model_name, successors, max_stations = instance.equipment.n_stations)
                improvement = opt_function!(x_soi, s, s+1, right, o,  instance, equipment_assignments, equip_costs, productivity_per_worker; predecessors=predecessors, successors=successors)
                if improvement
                    continue
                end
        end
        counter += 1
    end


    return x_soi, equipment_assignments
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


function get_tasks_assigned_from_xsoi(x_soi::Array{Int,3}, station::Int)
    station_assignments = dropdims(sum(x_soi[station,:,:], dims = (2)), dims = 2)
    station_assignments = findall(x -> x > 0 , station_assignments)
    #string_vector = map(string, station_assignments)
    return station_assignments
end

#inserts the empty stations into the equipment assignment so the improvement heuristic does not break
function fix_empty_stations!(equipment_assignments::Dict, instance::MALBP_W_instance)
    for station in 1:instance.equipment.n_stations
        if !haskey(equipment_assignments, station)
            equipment_assignments[station] = []
        end
    end

end

function both_opts(instance::MALBP_W_instance, 
    x_soi::Array{Int,3}, 
    equipment_assignments; 
    n_iterations::Int=100, 
    productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], 
    model_name::String="combined_model", 
    opt_function!::Function = opt1_insertion_w_equip!
    )
    x_so, equipment_assignments = task_opt_fixed(instance, x_soi, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt1_insertion_w_equip!)
    x_so, equipment_assignments = task_opt_fixed(instance, x_soi, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt2_insertion_w_equip!)
    return x_so, equipment_assignments
end

function both_opts2(instance::MALBP_W_instance, 
    x_soi::Array{Int,3}, 
    equipment_assignments; 
    n_iterations::Int=100, 
    productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.], 
    model_name::String="combined_model", 
    opt_function!::Function = opt1_insertion_w_equip!
    )
    x_so, equipment_assignments = task_opt_fixed(instance, x_soi, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt1_insertion_w_equip!)
    x_so, equipment_assignments = task_opt_fixed(instance, x_soi, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt2_insertion_w_equip!)
    x_so, equipment_assignments = task_opt_fixed(instance, x_soi, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt1_insertion_w_equip!)

    return x_so, equipment_assignments
end

function construct_then_improve(orig_instance::MALBP_W_instance; 
                                set_cover_heuristic::Function=greedy_set_cover_v2, 
                                productivity_per_worker::Vector{Float64}= [1., 1., 1., 1.],
                                n_iterations::Int=100,
                                swapper::Function=task_opt_fixed,
                                opt_function::Function= opt1_insertion_w_equip!)
    x_so, equipment_assignments, combined_model = task_equip_heuristic_combined_precedence_fixed(orig_instance,  set_cover_heuristic=set_cover_heuristic)
    fix_empty_stations!(equipment_assignments, orig_instance)
    x_so, equipment_assignments = swapper(combined_model, x_so, equipment_assignments; n_iterations = n_iterations, productivity_per_worker=productivity_per_worker, opt_function! = opt_function)
    x_soi = zeros(Int, orig_instance.equipment.n_stations, orig_instance.equipment.n_tasks, orig_instance.models.n_models)
    for model in 1:orig_instance.models.n_models
        x_soi[:,:,model] = x_so[:,:,1]
    end
    y, y_w, y_wts, _ = worker_assignment_heuristic(orig_instance, x_soi, productivity_per_worker = productivity_per_worker)
    return x_soi, y, y_w, y_wts, equipment_assignments
end

#config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/constructive_debug.yaml"
config_filepath = "xps/paper1_large_instances.yaml"
#config_filepath = "xps/paper1_instances.yaml"
instances = read_MALBP_W_instances(config_filepath)

n_iterations_list = [10, 100, 200]
for n_iterations in n_iterations_list
    println("now with $n_iterations: ")
    for instance in instances
        x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic_combined_precedence(instance, set_cover_heuristic=greedy_set_cover_v2)
        original_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
        # println("original y: ", y)
        #println("original cost: ", original_cost)
        # # assigned_tasks = get_tasks_assigned_from_xsoi(x_soi,2)
        # # println("These are the assigned tasks:", assigned_tasks)

        x_soi, y, y_w, y_wts, equipment_assignments = construct_then_improve(instance; set_cover_heuristic=greedy_set_cover_v2, n_iterations=100, opt_function = opt1_insertion_w_equip!)
        opt1_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
        # print("opt1 y ", y)
        # println("opt1 equip assign: ", equipment_assignments)
        #println("opt1 cost: ", opt1_cost)

        x_soi, y, y_w, y_wts, equipment_assignments = construct_then_improve(instance; set_cover_heuristic=greedy_set_cover_v2, n_iterations=100, opt_function = opt2_insertion_w_equip!)
        opt2_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
        # print("opt2 y ", y)
        # println("opt2 equip assign: ", equipment_assignments)
        #println("opt2 cost: ", opt2_cost)
        x_soi, y, y_w, y_wts, equipment_assignments = construct_then_improve(instance; set_cover_heuristic=greedy_set_cover_v2, n_iterations=100, swapper=both_opts)
        bothopt_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
        # print("bothopt y ", y)
        # println("bothopt equip assign: ", equipment_assignments)
        #println("bothopt cost: ", total_cost)
        x_soi, y, y_w, y_wts, equipment_assignments = construct_then_improve(instance; set_cover_heuristic=greedy_set_cover_v2, n_iterations=100, swapper=both_opts2)
        bothopt2_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y, y_w, instance)
        println("instance_name: $(instance.name) original cost $(original_cost) , opt1_cost $(opt1_cost) , opt2_cost $(opt2_cost) , bothopt $(bothopt_cost) , multi_apply $(bothopt2_cost)")

    end
end
# new_equip_assign, new_capabilities = greedy_set_cover_v2(assigned_tasks, instance, 2)
# println("new equip assignment", new_equip_assign)
# #task_location, station_capababilities = get_equipment_capable_stations(instance, equipment_assignments)
# task_times = get_task_assignment_times_x_wsot(instance, x_soi)

# println("equipment_assignments: ", equipment_assignments)
# println("task_location: ", task_location)
# println("station_capababilities: ", station_capababilities)
# println("task_times: ", task_times)