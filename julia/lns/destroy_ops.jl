function random_model_destroy!(m::Model, instance::MALBP_W_instance; seed:: Union{Nothing, Int}=nothing, n_destroy::Int=1,_...)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    if n_destroy >= instance.models.no_models
        @info "n_destroy is greater than the number of models, setting n_destroy to the number of models"
        return
    else
        models = sample(collect(keys(instance.models.models)), n_destroy)
    end
    x_wsoj = m[:x_wsoj]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the models that are not in the models list
    println("models: ", models)
    for (w,seq) in enumerate(eachrow(instance.scenarios))
        for j in 1:instance.sequence_length
            if  seq["sequence"][j] in models
                continue
            else
                fix.(x_wsoj[w, :, :, j], start_value.(x_wsoj[w, :, :, j]), force=true)
                #can also fix worker assignment at stations when the model passes through
                for s in 1:(instance.equipment.no_stations)
                    t = j + s - 1
                    fix(y_wts[w, t, s], start_value(y_wts[w, t, s]), force=true)
                end

            end
        end
    end
end

function random_model_destroy_md!(m::Model, instance::MALBP_W_instance; seed:: Union{Nothing, Int}=nothing, n_destroy::Int=1,_...)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    if n_destroy >= instance.models.no_models
        @info "n_destroy is greater than the number of models, setting n_destroy to the number of models"
        return
    else
        models = sample(collect(keys(instance.models.models)), n_destroy)
    end
    x_soi = m[:x_soi]
    y_wts = m[:y_wts]
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

function random_station_destroy!(m::Model, instance::MALBP_W_instance; seed::Union{Nothing, Int}=nothing, n_destroy::Int=1, _...)
    #if the seed is not none, set the seed
    if !isnothing(seed)
        Random.seed!(seed)
    end
    #randomly select no_stations stations to remove
    if n_destroy >= instance.equipment.no_stations
        @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
        return
    else
        station = rand(1:(instance.equipment.no_stations - n_destroy+1) )
        stations = [station:station + n_destroy-1;]
    end
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.no_stations
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
function random_station_destroy_md!(m::Model, instance::MALBP_W_instance; seed::Union{Nothing, Int}=nothing, n_destroy::Int=1, _...)
    #if the seed is not none, set the seed
    if !isnothing(seed)
        Random.seed!(seed)
    end
    #randomly select no_stations stations to remove
    if n_destroy >= instance.equipment.no_stations
        @info "n_destroy is greater than the number of stations, setting n_destroy to the number of stations"
        return
    else
        station = rand(1:(instance.equipment.no_stations - n_destroy+1) )
        stations = [station:station + n_destroy-1;]
    end
    x_soi = m[:x_soi]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    for s in 1:instance.equipment.no_stations
        if s in stations
            continue
        else
            fix.(x_soi[ s, :, :], start_value.(x_soi[ s, :, :]), force=true)
            fix.(u_se[s, :], start_value.(u_se[s, :]), force=true)
            fix.(y_wts[:, :, s], start_value.(y_wts[:, :, s]), force=true)
        end
    end
end

function random_subtree_destroy!(m::Model, instance::MALBP_W_instance; seed::Union{Nothing, Int}=nothing, n_destroy::Int=1, depth::Int=1, _...)
    #if the seed is not none, set the seed
    if !isnothing(seed)
        Random.seed!(seed)
    end
    #randomly selects n_destroy sequences of tasks to remove
    x_wsoj = m[:x_wsoj]
    y_wts = m[:y_wts]
    prod_sequences = instance.scenarios[rand(1:instance.no_scenarios, n_destroy), :]
    for w in 1:instance.no_scenarios
        shared_scenario = false
        for to_remove in eachrow(prod_sequences)
            if to_remove["sequence"][1:depth] == instance.scenarios[w, "sequence"][1:depth]
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