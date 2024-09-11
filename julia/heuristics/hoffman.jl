# include("../read_MALBP_W.jl")
# include("preprocessing.jl")
# include("constructive.jl")



#This function returns the last element of a vector that is the smallest
function get_last_smallest_station(workers_per_station::Array{Int})
    last_min = 0
    for (station, workers) in enumerate(workers_per_station)
        if workers == minimum(workers_per_station)
            last_min = station
        end
    end
    return last_min
end

# #This function assigns the tasks to the stations
# function recursive_task_fill(instance::MALBP_W_instance, model::ModelInstance, min_workers::Dict{String, Any}, precedence_matrix::Dict{String, Any}, productivity_per_worker::Array{Float64}= [1., 1., 1., 1.] )
#     println("min workers: ", min_workers)
#     available_station_time = copy(min_workers["available_station_time"])
#     workers_per_station = copy(min_workers["workers_per_station"])
#     # x_soi = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models)
#     station_assignments = Dict{Int, Vector{String}}()
#     function station_fill!(station::Int, update_station::Int, remaining_task_time::Real)
#         if remaining_task_time <= 0
#             return
#         elseif station == instance.equipment.n_stations && remaining_task_time > sum(available_station_time[station])
#             #need to reset the tasks that will be unassigned and reassigned
#             for i in update_station:instance.equipment.n_stations
#                 available_station_time[i] += sum([ time for (task, time) in model.task_times[o] for o in station_assignments[i]])
#                 remaining_task_time += sum([ time for (task, time) in model.task_times[o] for o in station_assignments[i]])
#                 station_assignments[i] = []
#             end
#             workers_per_station[update_station] += 1
#             available_station_time[update_station] += instance.models.cycle_time * productivity_per_worker[workers_per_station[update_station]]
#             update_station = get_last_smallest_station(workers_per_station)
#             station_fill!(station, update_station, remaining_task_time)
#         else

#         end
#     end


#     return
# end







function hoffman_salbp(model_instance::ModelInstance, cycle_time::Real, precedence_matrix::Array{Int}, index_to_task::Dict{Int, String})
    #precedence_matrix, task_to_index, index_to_task = create_precedence_matrix(model_instance, order_function = positional_weight_order)
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
    cycle_leftovers = [cycle_time]
    best_sequence = copy(precedence_matrix[end, :])
    station  = 1
    left_task = 1  
    n_iterations = 0
    while any(best_sequence .>= 0) && n_iterations < 1000
        best_sequence, best_value = recursive_task_fill(station, left_task, best_sequence, cycle_leftovers[station])
        cycle_leftovers[station] = best_value
        push!(cycle_leftovers, cycle_time)
        station += 1
        left_task = findfirst(x -> x == 0, best_sequence)
        n_iterations += 1
    end
    println("solved after ", n_iterations, " iterations")
    return best_sequence, cycle_leftovers
end


# config_filepath = "SALBP_benchmark/MM_instances/xp_yaml/julia_debug.yaml"
# instances = read_MALBP_W_instances(config_filepath)
# instance = instances[1]
# # println("running instance", instance.config_name)
# min_workers = calculate_min_workers(instance)



# #sums the values of my_dict
# sum([time for (task, time) in my_dict])
# # modelA = instance.models.models["B"]

# # hoffman_salbp(modelA, instance.models.cycle_time)
# # println("The model name is ", modelA.name)
# # min_workers = calculate_min_workers(instance)
# # println("min workers", min_workers)
# # # precedence_matrices = create_precedence_matrices(instance; order_function = positional_weight_order)
# # # precedence_matrices

# # modified_hoffman(instance)

# # get_last_smallest_station([1, 2, 3, 4, 1])
