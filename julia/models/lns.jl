
mutable struct RepairOp
    repair!::Function
    repair_kwargs::Dict
end

mutable struct DestroyOp
    name::String
    destroy!::Function
    destroy_change!::Function
    destroy_kwargs::Dict
    old_destroy_kwargs::Dict
end

function DestroyOp(destroy!::Function, destroy_change!::Function, destroy_kwargs::Dict)
    old_destroy_kwargs = deepcopy(destroy_kwargs)
    return DestroyOp(string(destroy!), destroy!, destroy_change!, destroy_kwargs, old_destroy_kwargs)
end

function update_destroy_operator!(des::DestroyOp, new_destroy!::Function)
    des.destroy! = new_destroy!
    des.name = string(new_destroy!)
end

struct LNSConf
    n_iterations::Union{Int, Nothing}
    n_iter_no_improve::Int
    time_limit::Float64
    rep::RepairOp
    des::DestroyOp
    seed::Union{Nothing, Int}
end

function read_search_strategy_YAML(config_filepath::String, run_time::Float64)
    config_file = YAML.load(open(config_filepath))
    #if the LNS section is not in the config file, return an empty dictionary

    if !haskey(config_file, "lns")
        return Dict()
    elseif !haskey(config_file["lns"], "time_limit")
        @info "no time limit specified in config file, defaulting to command line defined $run_time seconds"
        config_file["lns"]["time_limit"] = run_time
    end
    search_strategy = config_file["lns"]
    search_strategy = get_search_strategy_config(search_strategy)
    return search_strategy
end

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

function random_station_destroy!(m::Model, instance::MALBP_W_instance; seed::Union{Nothing, Int}=nothing, n_destroy::Int=1, _...)
    println("n_destroy: ", n_destroy)
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

function no_change(iter_no_improve::Int, lns_obj::LNSConf, m::Model)
    return iter_no_improve, lns_obj, m
end

function decrement_y!(iter_no_improve::Int, lns_obj::LNSConf, m::Model)
    if iter_no_improve % lns_obj.des.destroy_kwargs[:change_freq] == 0
        y = m[:y]
        println("Fixing y at ", start_value(y)-1)
        fix.(y, start_value(y)-1, force=true)

    elseif iter_no_improve % lns_obj.des.destroy_kwargs[:change_freq] in [1:lns_obj.des.destroy_kwargs[:fix_steps];];
        y = m[:y]
        println("Fixing y at ", start_value(y))
        fix.(y, start_value(y), force=true)
    end
    return iter_no_improve, lns_obj, m
end

function increase_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model)
    if iter_no_improve % lns_obj.des.destroy_kwargs[:change_freq] == 0
        lns_obj.des.destroy_kwargs[:n_destroy] += 1
    end
    return iter_no_improve, lns_obj, m
end

function change_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model)
    if iter_no_improve % lns_obj.des.destroy_kwargs[:change_freq] == 0
        #randomly chooses from the destroy operators
        operator_list = [random_station_destroy!, random_subtree_destroy!,random_model_destroy!]
        #filters out the current operator
        operator_list = filter(x -> x != lns_obj.des.destroy!, operator_list)
        destroy = sample(operator_list, 1)[1]
        update_destroy_operator!(lns_obj.des, destroy)
    end
    return iter_no_improve, lns_obj, m
end

function change_destroy_increase_size!(iter_no_improve::Int, lns_obj::LNSConf, m::Model)
    if iter_no_improve % lns_obj.des.destroy_kwargs[:change_freq] == 0
        if lns_obj.des.destroy_kwargs[:n_destroy] < lns_obj.des.destroy_kwargs[:destroy_limit] 
            lns_obj.des.destroy_kwargs[:n_destroy] += 1
        else
            #resets the size of block to destroy
            lns_obj.des.destroy_kwargs[:n_destroy]= lns_obj.des.old_destroy_kwargs[:n_destroy]
            #randomly chooses from the destroy operators
            operator_list = [random_station_destroy!, random_subtree_destroy!,random_model_destroy!]
            #filters out the current operator
            operator_list = filter(x -> x != lns_obj.des.destroy!, operator_list)
            destroy = sample(operator_list, 1)[1]
            update_destroy_operator!(lns_obj.des, destroy)
        end
            
    end
    return iter_no_improve, lns_obj, m
end

function configure_destroy(search_strategy::Dict)
    if !haskey(search_strategy, "destroy") || !haskey(search_strategy["destroy"], "operator")
        @info "No destroy specified, defaulting to random_station_destroy"
        search_strategy["destroy"] = Dict()
        destroy_op = random_station_destroy!
        search_strategy["destroy"]["kwargs"] = Dict(Symbol("n_destroy")=>2)
        search_strategy["destroy"]["change"] = no_change
    else
        @info "Deconstructor specified: $(search_strategy["destroy"]["operator"])"
        destroy = search_strategy["destroy"]["operator"]
        if destroy == "random_station" || destroy == "random_station_destroy!"
            destroy_op = random_station_destroy!
        elseif destroy == "random_subtree" || destroy == "random_subtree_destroy!"
            destroy_op = random_subtree_destroy!
        elseif destroy == "random_model" || destroy == "random_model_destroy!"
            destroy_op = random_model_destroy!
        else
            @error "Deconstructor operator $(destroy) not recognized"
        end
        if !haskey(search_strategy["destroy"], "kwargs")
            @info "No destroy arguments specified, defaulting to n_destroy=2"
            destroy_kwargs = Dict("n_destroy"=>2)
        else
            @info "Deconstructor arguments specified: $(search_strategy["destroy"]["kwargs"])"
            destroy_kwargs = search_strategy["destroy"]["kwargs"]
            #converts the keys to symbols
            destroy_kwargs = Dict(Symbol(k) => v for (k, v) in destroy_kwargs)
        end
        #destroy operator change configuration
        if !haskey(search_strategy["destroy"], "change")
            @info "No destroy change specified, defaulting to no_change"
            destroy_change = no_change
        else
            if search_strategy["destroy"]["change"] == "increase_destroy!" || search_strategy["destroy"]["change"] == "increase_destroy"
                @info "Deconstructor change operator $(search_strategy["destroy"]["change"]) recognized"
                destroy_change = increase_destroy!
            elseif search_strategy["destroy"]["change"] == "no_change"
                @info "Deconstructor change operator $(search_strategy["destroy"]["change"]) recognized"
                destroy_change = no_change
            elseif search_strategy["destroy"]["change"] == "decrement_y!"
                @info "Deconstructor change operator $(search_strategy["destroy"]["change"]) recognized"
                destroy_change = decrement_y!
            elseif search_strategy["destroy"]["change"] == "change_destroy!"
                @info "Deconstructor change operator $(search_strategy["destroy"]["change"]) recognized"
                destroy_change = change_destroy!
            elseif search_strategy["destroy"]["change"] == "change_destroy_increase_size!"
                @info "Deconstructor change operator $(search_strategy["destroy"]["change"]) recognized"
                destroy_change = change_destroy_increase_size!
            else
                @error "Deconstructor change operator $(search_strategy["destroy"]["change"]) not recognized"
            end
        end 
        
    end
    destroy_operator = DestroyOp(destroy_op,  destroy_change, destroy_kwargs)
    return destroy_operator
end

function configure_repair(search_strategy::Dict)
    if !haskey(search_strategy, "repair") || !haskey(search_strategy["repair"], "operator")
        @info "No repair specified, defaulting to MILP"
        search_strategy["repair"] = Dict()
        repair_op = optimize!
        search_strategy["repair"]["kwargs"] = Dict("time_limit"=>100)
    else
        @info "Repair operator specified: $(search_strategy["repair"]["operator"])"
        repair = search_strategy["repair"]["operator"]
        if repair == "optimize!" || repair == "MILP" || repair == "milp!"
            repair_op = optimize!
        else
            @error "Repair operator $(repair) not recognized"
        end
        if !haskey(search_strategy["repair"], "kwargs")
            @info "No repair arguments specified, defaulting to time_limit=100"
            repair_kwargs = Dict("time_limit"=>100)
        else
            @info "Repair arguments specified: $(search_strategy["repair"]["kwargs"])"
            repair_kwargs = search_strategy["repair"]["kwargs"]
            #converts the keys to symbols
            search_strategy["repair"]["kwargs"] = Dict(Symbol(k) => v for (k, v) in repair_kwargs)
        end
    end
    repair_operator = RepairOp(repair_op, repair_kwargs)
    return repair_operator
end

function get_search_strategy_config(search_strategy::Dict)
    if !haskey(search_strategy, "n_iterations")
        @info "No number of iterations specified, defaulting to 10000"
        search_strategy["n_iterations"] = 10000
    else
        @info "Number of iterations specified: $(search_strategy["n_iterations"])"
    end
    if !haskey(search_strategy, "n_iter_no_improve")
        @info "No number of iterations with no improvement specified, defaulting to 2"
        search_strategy["n_iter_no_improve"] = 2
    else
        @info "Number of iterations with no improvement specified: $(search_strategy["n_iter_no_improve"])"
    end
    if !haskey(search_strategy, "time_limit")
        @info "No time limit specified, defaulting to 600 seconds"
        search_strategy["time_limit"] = 600
    else
        @info "Time limit specified: $(search_strategy["time_limit"]) seconds"
    end
    #repair configuration
    repair_op = configure_repair(search_strategy)
    #destroy configuration
    destroy_op = configure_destroy(search_strategy)
    #setting the seed (if not none)
    if !haskey(search_strategy, "seed")
        search_strategy["seed"] = nothing
    end
    lns_obj = LNSConf(search_strategy["n_iterations"], 
    search_strategy["n_iter_no_improve"], 
    search_strategy["time_limit"], 
    repair_op, 
    destroy_op, 
    search_strategy["seed"])
    return lns_obj
end


#function to unfix all fixed variables in the model and load their 
function unfix_vars!(m::Model , instance::MALBP_W_instance)
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    #unfixes task assignment variables
    for x in x_wsoj
        if is_fixed(x)
            unfix(x)
        end
    end
    #unfixes equipment assignment variables
    for u in u_se
        if is_fixed(u)
            unfix(u)
        end
    end
    #unfixes total recourse worker assignment
    for y in y_w
        if is_fixed(y)
            unfix(y)
            set_lower_bound(y, 0)
        end
    end
    #unfixes worker_assignment, reassigns max workers bounds
    for worker_assignment in y_wts
        if is_fixed(worker_assignment)
            unfix(worker_assignment)
            set_upper_bound(worker_assignment, instance.max_workers)
            set_lower_bound(worker_assignment, 0)
        end
    end

end




function large_neighborhood_search!(m::Model, instance::MALBP_W_instance, search_strategy_fp::String; lns_res_fp::String="", md_obj_val::Union{Nothing, Float64}=nothing, run_time::Real=600.0 )
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time)
    println(lns_conf)
    println("lns_conf", lns_conf.rep.repair_kwargs)
    seed = lns_conf.seed
    #sets the time limit for the model
    set_optimizer_attribute(m, "TimeLimit", lns_conf.rep.repair_kwargs["time_limit"])
    start_time = time()
    obj_vals = []
    incumbent = Inf
    incumbent_dict = Dict()
    #best_m = copy(m)
    #saves the initial objective value
    if !isnothing(md_obj_val)
        res_dict = Dict("instance"=> instance.config_name,"iteration"=>0, "obj_val"=>md_obj_val, "time"=>0.0, "operator"=>"initial")
        push!(obj_vals, res_dict)
        incumbent = md_obj_val
    end

    #main loop of algorithm
    iter_no_improve = 0
    for i in 1: lns_conf.n_iterations
        #calls the destroy operator
        lns_conf.des.destroy!(m, instance, seed=seed ; lns_conf.des.destroy_kwargs...)
        #repairs using MILP TODO: add other repair operators
        optimize!(m)
        #saves the results
        res_dict = Dict("instance"=> instance.config_name,"iteration"=>i, "obj_val"=>objective_value(m), "time"=>time()-start_time, "operator"=>lns_conf.des.name)
        push!(obj_vals, res_dict)
        if objective_value(m) < incumbent
            incumbent = objective_value(m)
            incumbent_dict = res_dict
            #best_m = copy(m)
            iter_no_improve = 0
        else
            iter_no_improve += 1
        end
        if time() - start_time > lns_conf.time_limit
            @info "Time limit reached, stopping LNS at iteration $i"
            break
        end
        if i < lns_conf.n_iterations
            #x = all_variables(m)
            #solution = value.(x)
            unfix_vars!(m, instance)
            #set_start_value.(x, solution)
            lns_conf.des.destroy_change!(iter_no_improve, lns_conf, m)
        end

    end
    #writes the results to a csv
    obj_df = DataFrame(obj_vals)
    CSV.write(lns_res_fp, obj_df)
    #writes the lns_config to a yaml file
    YAML.write_file(lns_res_fp * "lns_config.yaml",lns_conf )
    println("best obj val: ", incumbent_dict)
    return incumbent_dict
end
