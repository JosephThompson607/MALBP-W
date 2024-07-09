function random_model_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25,_...)
    n_destroy = max(1,round(Int, instance.models.n_models * percent_destroy))
    if n_destroy >= instance.models.n_models
        if percent_destroy >= 1.0
        @info "n_destroy is greater than the number of models, setting n_destroy to the number of models"
        return
        else
            @info "percent destroy is not at 100% yet, will set 1 below the number of models"
            n_destroy = instance.models.n_models - 1
        end
    end
    models = sample(rng,collect(keys(instance.models.models)), n_destroy)
    x_wsoj = m[:x_wsoj]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the models that are not in the models list
    println("models: ", models)
    for (w,seq) in enumerate(eachrow(instance.sequences.sequences))
        for j in 1:instance.sequences.sequence_length
            if  seq["sequence"][j] in models
                continue
            else
                fix.(x_wsoj[w, :, :, j], start_value.(x_wsoj[w, :, :, j]), force=true)
                #can also fix worker assignment at stations when the model passes through
                for s in 1:(instance.equipment.n_stations)
                    t = j + s - 1
                    fix(y_wts[w, t, s], start_value(y_wts[w, t, s]), force=true)
                end

            end
        end
    end
end

#calls the random station destroy and the random model destroy functions, good for larger instances
function random_station_model_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, _...)
    #chooses randomly between splitting the percent destroy between the two destroy functions or using the percent destroy for one and set the other to 1
    percent_station, percent_model = rand(rng,[(percent_destroy, 1.00), (1.00,percent_destroy), (sqrt(percent_destroy), sqrt(percent_destroy))])
    random_model_destroy!(m, instance; rng = rng, percent_destroy=percent_model)
    random_station_destroy!(m, instance; rng = rng, percent_destroy=percent_station)
end

#calls the random station destroy and the random subtree destroy functions
function random_station_subtree_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, depth::Int=1, _...)
    #chooses randomly between splitting the percent destroy between the two destroy functions or using the percent destroy for one and set the other to 1
    percent_subtree, percent_station = rand(rng,[(percent_destroy, 1.00), (1.00,percent_destroy), (sqrt(percent_destroy), sqrt(percent_destroy))])
    random_subtree_destroy!(m, instance; rng = rng, percent_destroy=percent_subtree, depth=depth)
    random_station_destroy!(m, instance; rng = rng, percent_destroy=percent_station)
end

#calls the random model and the random subtree destroy functions
function random_model_subtree_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, depth::Int=1, _...)
    #chooses randomly between splitting the percent destroy between the two destroy functions or using the percent destroy for one and set the other to 1
    percent_subtree, percent_model = rand([(percent_destroy, 1.00), (1.00,percent_destroy), (sqrt(percent_destroy), sqrt(percent_destroy))])
    random_subtree_destroy!(m, instance; rng = rng, percent_destroy=percent_subtree, depth=depth)
    random_model_destroy!(m, instance; rng = rng, percent_destroy=percent_model)
end

#calls the random station destroy and the random model destroy functions for the model dependent formulation, good for larger instances
function random_station_model_destroy_md!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, _...)
    random_model_destroy_md!(m, instance; rng = rng, percent_destroy=percent_destroy)
    random_station_destroy_md!(m, instance; rng = rng, percent_destroy=percent_destroy)
end


#randomly destroys models for the md formulation
function random_model_destroy_md!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25,_...)
    n_destroy = max(1,round(Int, instance.models.n_models * percent_destroy))
    if n_destroy >= instance.models.n_models
        if percent_destroy >= 1.0
        @info "n_destroy is greater than the number of models, setting n_destroy to the number of models"
        return
        else
            @info "percent destroy is not at 100% yet, will set 1 below the number of models"
            n_destroy = instance.models.n_models - 1
        end
    end
    models = sample(rng, collect(keys(instance.models.models)), n_destroy)
    
    x_soi = m[:x_soi]
    #fixes the task assignment, equipment, and worker assignment for the models that are not in the models list
    println("models: ", models)
    model_indexes = Dict(model_name => i for (i, model_name) in enumerate(keys(instance.models.models)))
    for (model_name, data) in instance.models.models
        if  model_name in models
            continue
        else
            i  = model_indexes[model_name]
            fix.(x_soi[ :, :, i], start_value.(x_soi[ :, :, i]), force=true)
        end
    end
end

function random_station_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, _...)
    n_destroy = max(1,round(Int, instance.equipment.n_stations * percent_destroy))

    if n_destroy >= instance.equipment.n_stations
        if percent_destroy >= 1.0
            @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
            return
        else
            @info "percent destroy is not at 100% yet, will set 1 below the number of stations"
            n_destroy = instance.equipment.n_stations - 1
        end
    end
    station = rand(rng, 1:(instance.equipment.n_stations - n_destroy+1) )
    stations = [station:station + n_destroy-1;]
    println("stations: ", stations)
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.n_stations
        if s in stations
            continue
        else
            fix.(x_wsoj[:, s, :, :], start_value.(x_wsoj[:, s, :, :]), force=true)
            fix.(u_se[s, :], start_value.(u_se[s, :]), force=true)
            fix.(y_wts[:, :, s], start_value.(y_wts[:, :, s]), force=true)
        end
    end
end

#Randomly destroys stations for the md formulation
function random_station_destroy_md!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64 = 0.25, _...)
    n_destroy = max(1,round(Int, instance.equipment.n_stations * percent_destroy))
    if n_destroy >= instance.equipment.n_stations
        if percent_destroy >= 1.0
            @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
            return
        else
            @info "percent destroy is not at 100% yet, will set 1 below the number of stations"
            n_destroy = instance.equipment.n_stations - 1
        end

    end
    station = rand(rng, 1:(instance.equipment.n_stations - n_destroy+1) )
    stations = [station:station + n_destroy-1;]

    x_soi = m[:x_soi]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.n_stations
        if s in stations
            continue
        else
            fix.(x_soi[ s, :, :], start_value.(x_soi[ s, :, :]), force=true)
            fix.(u_se[s, :], start_value.(u_se[s, :]), force=true)
            fix.(y_wts[:, :, s], start_value.(y_wts[:, :, s]), force=true)
        end
    end
end

function random_subtree_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64=0.25, depth::Int=1, _...)
    n_destroy = max(1, round(Int, instance.models.n_models ^ depth * percent_destroy))

    if percent_destroy >= 1.0
        @info "n_destroy is greater than the number of models, setting n_destroy to the number of models"
        return
    #if the percent is not at 100, set the n_destroy to the number of models^depth -1
    elseif n_destroy == instance.models.n_models ^ depth  
        @info "percent destroy is not at 100% yet, will set 1 below the number of models"
        n_destroy = instance.models.n_models ^ depth - 1
    end
    #randomly selects n_destroy sequences of tasks to remove
    x_wsoj = m[:x_wsoj]
    y_wts = m[:y_wts]
    prod_sequences = instance.sequences.sequences[rand(rng, 1:instance.sequences.n_scenarios, n_destroy), :]
    for w in 1:instance.sequences.n_scenarios
        shared_scenario = false
        for to_remove in eachrow(prod_sequences)
            if to_remove["sequence"][1:depth] == instance.sequences.sequences[w, "sequence"][1:depth]
                shared_scenario = true
                break
            end
        end
        if ! shared_scenario
            #fixes the task assignment and worker assignment for the stations that are not in the stations list
            fix.(x_wsoj[w, :, :, :], start_value.(x_wsoj[w, :, :, :]), force=true)
            fix.(y_wts[w, :, :], start_value.(y_wts[w, :, :]), force=true)
        end
    end
end

function peak_station_destroy!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64, _...)
    n_destroy = max(1,round(Int, instance.equipment.n_stations * percent_destroy))
    y_wts = m[:y_wts]

    station_workers = []
    for s in 1:instance.equipment.n_stations
        push!(station_workers, sum(start_value.(y_wts[:, :, s])))
    end
    max_station = argmax(station_workers)


    #randomly select n_stations stations to remove
    if n_destroy >= instance.equipment.n_stations
        @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
        return
    else
        left = max_station - floor(n_destroy/2)
        if left < 1
            left = 1
            right = n_destroy
        else
            right = max_station + round(n_destroy/2)
        end
        stations = [left:right;]
        println("stations: ", stations)
    end
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.n_stations
        if s in stations
            continue
        else
            fix.(x_wsoj[:, s, :, :], start_value.(x_wsoj[:, s, :, :]), force=true)
            fix.(u_se[s, :], start_value.(u_se[s, :]), force=true)
            fix.(y_wts[:, :, s], start_value.(y_wts[:, :, s]), force=true)
        end
    end
end

function peak_station_destroy_md!(m::Model, instance::MALBP_W_instance; rng = Xoshiro(), percent_destroy::Float64=0.25, _...)
    n_destroy = max(1,round(Int, instance.equipment.n_stations * percent_destroy))
    y_wts = m[:y_wts]

    station_workers = []
    for s in 1:instance.equipment.n_stations
        push!(station_workers, sum(start_value.(y_wts[:, :, s])))
    end
    max_station = argmax(station_workers)


    #randomly select n_stations stations to remove
    if n_destroy >= instance.equipment.n_stations
        @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
        return
    else
        left = max_station - floor(n_destroy/2)
        if left < 1
            left = 1
            right = n_destroy
        else
            right = max_station + round(n_destroy/2)
        end
        stations = [left:right;]
        println("stations: ", stations)
    end
    x_soi = m[:x_soi]
    u_se = m[:u_se]
    
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.n_stations
        if s in stations
            continue
        else
            fix.(x_soi[ s, :, :], start_value.(x_soi[s, :, :]), force=true)
            fix.(u_se[s, :], start_value.(u_se[s, :]), force=true)
            fix.(y_wts[:, :, s], start_value.(y_wts[:, :, s]), force=true)
        end
    end
end