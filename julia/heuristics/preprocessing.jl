using GLMakie
using GraphMakie
using Graphs
#include("../read_MALBP_W.jl")





function create_correspondance_dicts(model::ModelInstance)
    #creates a dictionary that maps the task names to the index of the task in the model
    task_to_index = Dict{String, Int}()
    index_to_task = Dict{Int, String}()
    for (i, task) in enumerate(keys(model.task_times[1]))
        task_to_index[task] = i
        index_to_task[i] = task
    end
    return task_to_index, index_to_task
end

function generate_precedence_graph(model::ModelInstance, task_to_index::Dict{String, Int})
    #generates a directed graph that represents the precedence relations of the tasks in the model
    g = SimpleDiGraph{}(model.no_tasks)
    for (pred, suc) in model.precendence_relations
        predecessor_index = task_to_index[pred]
        successor_index = task_to_index[suc]
        add_edge!(g, predecessor_index, successor_index)
    end
    return g
end

function calculate_positional_weight( model::ModelInstance; reverse::Bool=false)
    #calculates the positional weight of the tasks in the model
    task_to_index, index_to_task = create_correspondance_dicts(model)
    g = generate_precedence_graph(model, task_to_index)
    if reverse
        reverse!(g)
    end
    trans_g = transitiveclosure(g)
    nodes = collect(vertices(trans_g))
    dependency_dict = Dict{String, Float64}()
    for node in nodes
        if !haskey(dependency_dict, string(index_to_task[node]))
            dependency_dict[string(index_to_task[node])] = model.task_times[1][index_to_task[node]]
        else
            dependency_dict[string(index_to_task[node])] += model.task_times[1][index_to_task[node]]
        end
        for child in outneighbors(trans_g,node)
            dependency_dict[string(index_to_task[node])] += model.task_times[1][index_to_task[child]]
        end
    end
    return dependency_dict
end
   

function calculate_depedency_time(instance::MALBP_W_instance; reverse=false)
    #calculates the total task time of the tasks that are in the dependency graph of the model
    dependency_time = Dict{String, Dict{String, Float64}}()
    for (model_name, model) in instance.models.models
        dependency_time[model_name] = calculate_positional_weight( model, reverse=reverse)
    end
    return dependency_time
end

function calculate_max_station_capacity(instance::MALBP_W_instance, productivity_per_worker::Dict{Int, Float64})
    #calculates the maximum number of workers that can be assigned to each station
    max_station_cap = zeros(Int, instance.equipment.no_stations)
    for station in 1:instance.equipment.no_stations
        max_station_cap[station] = sum([instance.models.cycle_time * productivity for productivity in values(productivity_per_worker)])
    end
    return max_station_cap
end
  


function find_infeasible_assignments(instance::MALBP_W_instance; productivity_per_worker::Dict{Int, Float64}= Dict(1=>1., 2=>1., 3=>1., 4=>1.), reverse=true)
    dependency_time = calculate_depedency_time(instance; reverse=reverse)
    #println(dependency_time)
    max_station_cap = calculate_max_station_capacity(instance, productivity_per_worker)
    #println("max_station_cap: ", max_station_cap)
    station_bans = get_station_bans(instance, dependency_time, max_station_cap, reverse)
    return station_bans
end

#Returns the infeasible assignments for tasks to stations
function get_infeasible_task_assignments(instance::MALBP_W_instance; productivity_per_worker::Dict{Int, Float64}= Dict(1=>1., 2=>1., 3=>1., 4=>1.))
    infeasible_assignments_forward = find_infeasible_assignments(instance; productivity_per_worker=productivity_per_worker)
    infeasible_assignments_backward = find_infeasible_assignments(instance; productivity_per_worker=productivity_per_worker, reverse=false)
    return infeasible_assignments_forward, infeasible_assignments_backward
end 



function get_station_bans(instance::MALBP_W_instance, depedency_time::Dict{String, Dict{T, Float64}}, max_station_cap::Vector{Int64}, reverse::Bool) where T
    if reverse
        stations = 1:instance.equipment.no_stations
    else
        stations = instance.equipment.no_stations:-1:1
    end
    model_task_bans = Dict{String, Dict{T, Array}}()
    for (model, orig_tasks) in depedency_time
        tasks  = copy(orig_tasks)
        task_bans = Dict{T, Array}()
        for (time1, task1) in sort(collect(zip(values(tasks),keys(tasks))))
            banned_stations = Vector{Int64}()
            for station in stations
                if time1 > max_station_cap[station]
                    push!(banned_stations, station)
                    time1 -= max_station_cap[station]
                end
            end
            if length(banned_stations) > 0
                task_bans[task1] = banned_stations
            end
        end
        model_task_bans[model] = task_bans
    end
    return model_task_bans
end

function find_infeasible_pairs(instance;productivity_per_worker::Dict{Int, Float64}= Dict(1=>1., 2=>1., 3=>1., 4=>1.), reverse::Bool=true)
    #possible_pairs = calculate_possible_pairs(instance, pair_size)
    dependency_time = combined_task_weight(instance; reverse=reverse)
    max_station_cap = calculate_max_station_capacity(instance, productivity_per_worker)
    station_bans = get_station_bans(instance, dependency_time, max_station_cap, reverse)
    return station_bans

end


function combined_task_weight(instance::MALBP_W_instance; reverse::Bool=true)
    dependency_time = Dict{String, Dict{Tuple{String,String}, Float64}}()
    for (model_name, model) in instance.models.models
        dependency_time[model_name] = Dict{Tuple{String,String}, Float64}()
        task_to_index, index_to_task = create_correspondance_dicts(model)
        g = generate_precedence_graph(model, task_to_index)
        if reverse
            reverse!(g)
        end
        trans_g = transitiveclosure(g)
        nodes = collect(vertices(trans_g))

        for (i, node1) in enumerate(nodes)
            for node2 in nodes[i+1:end]
                outneighbors_1 = outneighbors(trans_g, node1)
                if node2 in outneighbors_1
                    continue
                
                else
                    outneighbors_2 = outneighbors(trans_g, node2)
                    task_union = union(outneighbors_1, outneighbors_2)
                    if !haskey(dependency_time[model_name], (string(index_to_task[node1]), string(index_to_task[node2])))
                        dependency_time[model_name][(string(index_to_task[node1]), string(index_to_task[node2]))] = model.task_times[1][index_to_task[node1]] + model.task_times[1][index_to_task[node2]]
                    else
                        dependency_time[model_name][(string(index_to_task[node1]), string(index_to_task[node2]))] += model.task_times[1][index_to_task[node1]] + model.task_times[1][index_to_task[node2]]
                    end
                    for child in task_union
                        dependency_time[model_name][(string(index_to_task[node1]), string(index_to_task[node2]))] += model.task_times[1][index_to_task[child]]
                    end
                end
            end

        end
    end
    return dependency_time
end



#Returns the infeasible assignment pairs for tasks to stations
function get_infeasible_assignment_pairs(instance::MALBP_W_instance; productivity_per_worker::Dict{Int, Float64}= Dict(1=>1., 2=>1., 3=>1., 4=>1.))
    #infeasible stations at the beginning of the line
    infeasible_pairs_forward = find_infeasible_pairs(instance; productivity_per_worker=productivity_per_worker, reverse=true)
    #infeasible stations at the end of the line
    infeasible_pairs_backward = find_infeasible_pairs(instance; productivity_per_worker=productivity_per_worker, reverse=false)
    return infeasible_pairs_forward, infeasible_pairs_backward
end

#config_filepath = "SALBP_benchmark/MM_instances/julia_debug.yaml"

# #config_filepath = "SALBP_benchmark/MM_instances/medium_instance_config_S10.yaml"
#instance = read_MALBP_W_instances(config_filepath)[1]


# for (model_name, model) in instance.models.models

#     summer = 0
#     for (task, time) in model.task_times[1]
#         summer += time
#     end
#     println("model: ", model_name, " sum: ", summer)
# end
# #prec_dict = create_precedence_matrix(instance)
# #println(prec_dict["A"]["precedence_matrix"])

# # min_workers = calculate_min_workers(instance, Dict(1=>1., 2=>1., 3=>1., 4=>1.))
# #find_infeasible_assignments(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
#forward, backward = get_infeasible_task_assignments(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
# #calculate_possible_pairs(instance, 2)
# #infeasible stations at the beginning of the line
# #combined_task_weight(instance; reverse=true)
# #This one will be used for infeasible stations at the end of the line
# #combined_task_weight(instance; reverse=false)
# get_infeasible_assignment_pairs(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))



