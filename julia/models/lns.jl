
function read_search_strategy_YAML(config_filepath::String, run_time::Float64)
    config_file = YAML.load(open(config_filepath))
    #if the LNS section is not in the config file, return an empty dictionary
    println(config_file, typeof(config_file))

    if !haskey(config_file, "lns")
        return Dict()
    elseif !haskey(config_file["lns"], "time_limit")
        @info "no time limit specified in config file, defaulting to command line defined $run_time seconds"
        config_file["lns"]["time_limit"] = run_time
    end
    search_strategy = config_file["lns"]
    return search_strategy
end

function random_station_deconstructor!(m::Model, instance::MALBP_W_instance; seed::Union{Nothing, Int}=nothing, n_destroy::Int=1)
    #if the seed is not none, set the seed
    if !isnothing(seed)
        Random.seed!(seed)
    end
    #randomly select no_stations stations to remove
    if n_destroy >= instance.equipment.no_stations
        stations = [1:instance.equipment.no_stations;]
    else
        station = rand(1:(instance.equipment.no_stations - n_destroy+1) )
        stations = [station:station + n_destroy-1;]
    end
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    #fixes the task assignment, equipment, and worker assignment for the stations that are not in the stations list
    println(start_value(x_wsoj[1,1,1,1]))
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



function get_search_strategy_config(search_strategy::Dict)
    if !haskey(search_strategy, "n_iterations")
        @info "No number of iterations specified, defaulting to 3"
       n_iterations = 3
    else
        @info "Number of iterations specified: $(search_strategy["n_iterations"])"
       n_iterations = search_strategy["n_iterations"]
    end
    if !haskey(search_strategy, "time_limit")
        @info "No time limit specified, defaulting to 600 seconds"
        time_limit = 600
    else
        @info "Time limit specified: $(search_strategy["time_limit"]) seconds"
        time_limit = search_strategy["time_limit"]
    end
    if !haskey(search_strategy, "repair")
        @info "No repair specified, defaulting to MILP  for 10 min"
        repair_time_limit = 600
    else
        if !haskey(search_strategy["repair"], "time_limit")
            @info "No iteration time limit specified, defaulting to 100 seconds"
            repair_time_limit = 100
        else
            @info "Iteration time limit specified: $(search_strategy["repair"]["time_limit"]) seconds"
            repair_time_limit = search_strategy["repair"]["time_limit"]
        end
    end
    if !haskey(search_strategy, "deconstructor")
        @info "No deconstructor specified, defaulting to random_station_deconstructor"
        deconstructor = random_station_deconstructor!
    else
        @info "Deconstructor specified: $(search_strategy["deconstructor"]["operator"])"
        deconstructor = search_strategy["deconstructor"]["operator"]
        if deconstructor == "random_station"
            deconstructor = random_station_deconstructor!
        else
            @error "Deconstructor operator $(deconstructor) not recognized"
        end
        if !haskey(search_strategy["deconstructor"], "kwargs")
            @info "No deconstructor arguments specified, defaulting to an empty dictionary"
            deconstructor_kwargs = Dict()
        else
            @info "Deconstructor arguments specified: $(search_strategy["deconstructor"]["kwargs"])"
            deconstructor_kwargs = search_strategy["deconstructor"]["kwargs"]
            #converts the keys to symbols
            deconstructor_kwargs = Dict(Symbol(k) => v for (k, v) in deconstructor_kwargs)
        end
        
    end
    #setting the seed (if not none)
    if haskey(search_strategy, "seed")
        seed = search_strategy["seed"]
    else
        seed = nothing
    end
    
    
    return n_iterations, time_limit, repair_time_limit, deconstructor, deconstructor_kwargs, seed
end


#function to unfix all fixed variables in the model and load their 
function unfix_vars!(m::Model)
    for v in all_variables(m)
        if is_fixed(v)
            unfix(v)
        end
    end
end

# function lns!(m::Model, instance::MALBP_W_instance, search_strategy::Dict)
#     n_iterations, time_limit, repair_time_limit, deconstructor!, deconstructor_kwargs, seed = get_search_strategy_config(search_strategy)
#      #sets the time limit for the model
#      set_optimizer_attribute(m, "TimeLimit", repair_time_limit)
#      start_time = time()
#      for i in 1: n_iterations
#          deconstructor!(m, instance, seed=seed ; deconstructor_kwargs...)
#          optimize!(m)
#          push!(obj_vals, res_dict)
#          if i < n_iterations
#              x = all_variables(m)
#              solution = value.(x)
#              unfix_vars!(m)
#              set_start_value.(x, solution)
#          end
#          if time() - start_time > time_limit
#              @info "Time limit reached, stopping LNS at iteration $i"
#              break
#          end
#      end
#  end


function lns!(m::Model, instance::MALBP_W_instance, search_strategy::Dict; lns_res_fp::String="")
   n_iterations, time_limit, repair_time_limit, deconstructor!, deconstructor_kwargs, seed = get_search_strategy_config(search_strategy)
    #sets the time limit for the model
    set_optimizer_attribute(m, "TimeLimit", repair_time_limit)
    start_time = time()
    obj_vals = []
    for i in 1: n_iterations
        deconstructor!(m, instance, seed=seed ; deconstructor_kwargs...)
        optimize!(m)
        res_dict = Dict("iteration"=>i, "obj_val"=>objective_value(m), "time"=>time()-start_time, search_strategy...)
        push!(obj_vals, res_dict)
        if i < n_iterations
            x = all_variables(m)
            solution = value.(x)
            unfix_vars!(m)
            set_start_value.(x, solution)
        end
        if time() - start_time > time_limit
            @info "Time limit reached, stopping LNS at iteration $i"
            break
        end
    end
    #if the objective values are to be saved, save them
    obj_df = DataFrame(obj_vals)
    CSV.write(lns_res_fp, obj_df)

end