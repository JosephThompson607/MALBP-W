include("../read_MALBP_W.jl")
include("../heuristics/constructive.jl")
include("preprocessing.jl")

function get_equipment_capable_stations(instance, equipment_assignments)
    #For each task, gets the stations that are capable of performing the task
    task_locations = Dict{Int, Vector{Int}}()   
    #turns r_oe into a matrix
    r_eo = transpose(permutedims((stack(instance.equipment.r_oe)), [2,1]))
    #station capabilities is a 2D array where each row is a task and each column is a station
    station_capabilities = zeros(Int, instance.equipment.n_tasks, instance.equipment.n_stations)
    for station in keys(equipment_assignments)
        for equipment in equipment_assignments[station]
            #station capabilities is the sum of the indexes of r_oe
            station_capabilities[:, station] .+= r_eo[equipment, :] 
        end
        #each non-zero index is a task that can be performed at the station
        doable_tasks = findall(x->x>0, station_capabilities[:, station])
        for task in doable_tasks
            if haskey(task_locations, task)
                push!(task_locations[task], station)
            else
                task_locations[task] = [station]
            end
        end
    end
    return task_locations, station_capabilities
end


#Calculates the expected number of workers at each station for each production cycle
function get_expected_station_workers(instance, y_wts)
    #converts y_wts to a real array
    expected_workers = convert(Array{Float64, 3}, y_wts)
    #multiplies each scenario of y_wts by the probability of that scenario
    for scenario in 1:instance.sequences.n_scenarios
        expected_workers[scenario,:,:] .*= instance.sequences.sequences[scenario,:].probability
    end
    #sums the expected number of workers at each station and production cycle for all scenarios
    return expected_workers
end

#Calculates the task times for x_soi
function get_task_assignment_times_x_wsot(instance, x_soi)
    #task times is a 3D array where each row is a station, each column is a task, and each depth is a model
    task_times = zeros(Real, instance.sequences.n_scenarios, instance.equipment.n_stations, instance.equipment.n_tasks, instance.num_cycles )
    model_indexes = [i for (i, model_dict) in instance.models.models]
    model_to_index = Dict{String, Int}(zip(model_indexes, 1:length(model_indexes)))

    for s in 1:instance.equipment.n_stations
        for o in 1:instance.equipment.n_tasks
            for j in 1:instance.sequences.sequence_length
                for w in 1:instance.sequences.n_scenarios
                    model = instance.sequences.sequences[w, :].sequence[j]
                    i = model_to_index[model]
                    #if the task is assigned to the station in the cycle, the time is the task time
                    if x_soi[s, o, i] == 1
                        task = string(o)
                        t = j+ s -1
                        task_times[w, s, o, t] = instance.models.models[model].task_times[1][task] #* instance.sequences.sequences[w, :].probability
                    end
                end
            end
        end
    end
    return task_times

end

#This function stores the task times for each model in each station in each cycle for each model and scenario
function get_task_assignment_times_x_wsoit(instance, x_soi)
    #task times is a 3D array where each row is a station, each column is a task, and each depth is a model
    task_times = zeros(Real, instance.sequences.n_scenarios, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models, instance.num_cycles )
    model_indexes = [i for (i, model_dict) in instance.models.models]
    model_to_index = Dict{String, Int}(zip(model_indexes, 1:length(model_indexes)))

    for s in 1:instance.equipment.n_stations
        for o in 1:instance.equipment.n_tasks
            for j in 1:instance.sequences.sequence_length
                for w in 1:instance.sequences.n_scenarios
                    model = instance.sequences.sequences[w, :].sequence[j]
                    i = model_to_index[model]
                    #if the task is assigned to the station in the cycle, the time is the task time
                    if x_soi[s, o, i] == 1
                        task = string(o)
                        t = j+ s -1
                        task_times[w, s, o, i, t] = instance.models.models[model].task_times[1][task] #* instance.sequences.sequences[w, :].probability
                    end
                end
            end
        end
    end
    return task_times

end

function weight_task_times_probability_wsoit(task_times_wsoit, instance)
    #multiplies each task time by the probability of the scenario
    for scenario in 1:instance.sequences.n_scenarios
        task_times[scenario,:,:,:,:] .*= instance.sequences.sequences[scenario,:].probability
    end
    return task_times
end

#function that gets the peak cycle for a given task assignment
function get_peak_cycle(instance, x_soi, y_wts)
    model_indexes = [i for (i, model_dict) in instance.models.models]
    model_to_index = Dict{String, Int}(zip(model_indexes, 1:length(model_indexes)))
    index_to_model = Dict{Int, String}(zip(1:length(model_indexes), model_indexes))
    expected_workers = get_expected_station_workers(instance, y_wts)
    task_assignment_times_wsoit = get_task_assignment_times_x_wsoit(instance, x_soi)
    weighted_times_wsoit = weight_task_times_probability(task_assignment_times_wsoit, instance)
    cycle_stations = dropdims(sum(expected_workers, dims=(1)), dims=(1))

    println("total cycle assignment durations", dropdims(sum(weighted_times_wsoit, dims=(1,2,3,4)), dims=(1,2,3,4)))
    println("cycle stations shape", size(cycle_stations))
    println("cycle stations", cycle_stations)

end


function get_precedence_feasible_stations(instance, task_locations, station_capababilities)
    #gets all of the task_location pairs that only have one value in the task_locations dictionary

end

# config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/constructive_debug.yaml"
# # #config_filepath = "SALBP_benchmark/MM_instances/medium_instance_config_S10.yaml"
# instances = read_MALBP_W_instances(config_filepath)

# instance = instances[1]
# println("instance scenarios: ", instance.sequences.sequences)
# x_soi, y, y_w, y_wts, equipment_assignments = task_equip_heuristic(instance)
# println("x_soi: ", x_soi)
# println("y: ", y)
# println("y_w: ", y_w)
# println("y_wts: ", y_wts)

# task_locations, station_capababilities = get_equipment_capable_stations(instance, equipment_assignments)

# get_peak_model(instance, x_soi, y_wts)
# expected_workers = get_expected_station_workers(instance, y_wts)
# println("shape of expected workers", size(expected_workers))
# println("peak scenario", sum(expected_workers, dims=(2,3)))
# peak_cycle = dropdims(sum(expected_workers, dims=(1,3)), dims=(1,3))
# println("peak cycle", sum(expected_workers, dims=(1,3)))
# println("peak station", sum(expected_workers, dims=(1,2)))
# println(" y vs max y_wts: $y $(maximum(peak_cycle))")

# task_times_wsot = get_task_assignment_times_x_wsot(instance, x_soi)
# println("task times shape", size(task_times))
# println("task times", dropdims(sum(task_times_wsot, dims=(3)) , dims=(3)))
# println("task times scenario 1: ", dropdims(sum(task_times_wsot[1,:,:,:], dims=(2)) , dims=(2)))
# println("task times scenario 1 station 1: ", dropdims(sum(task_times_wsot[1,1,:,:], dims=(1)) , dims=(1)))
# println("task times scenario 1 station 2: ", dropdims(sum(task_times_wsot[1,2,:,:], dims=(1)) , dims=(1)))

# task_times_wsoit = get_task_assignment_times_x_wsoit(instance, x_soi)
# println("task times shape", size(task_times_wsoit))
# println("task times", dropdims(sum(task_times_wsoit, dims=(3,4)) , dims=(3,4)))
# println("task times scenario 1: ", dropdims(sum(task_times_wsot[1,:,:,:], dims=(2)) , dims=(2)))
# println("task times scenario 1 station 1: ", dropdims(sum(task_times_wsot[1,1,:,:], dims=(1)) , dims=(1)))
# println("task times scenario 1 station 2: ", dropdims(sum(task_times_wsot[1,2,:,:], dims=(1)) , dims=(1)))
#task_locations, station_capababilities = get_precedence_feasible_stations(instance, task_locations, station_capababilities)
#Gets the production cycle with the highest amount of workers

