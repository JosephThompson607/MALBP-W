mutable struct RepairOp
    repair!::Function
    repair_kwargs::Dict
end

mutable struct DestroyOp
    name::String
    destroy!::Function
    kwargs::Dict
    old_kwargs::Dict
    destroy_weights::Dict{String, Float64}
end

mutable struct ChangeOp
    change!::Function
    kwargs::Dict
    change_weights::Dict{String, Float64}
end

struct LNSConf
    n_iterations::Union{Int, Nothing}
    n_iter_no_improve::Int
    time_limit::Float64
    rep::RepairOp
    des::DestroyOp
    shake::ChangeOp
    adaptation!::Function
    seed::Union{Nothing, Int}
end

function DestroyOp(destroy!::Function,  destroy_kwargs::Dict; destroy_weights::Dict=Dict{String, Float64}("random_model_destroy!"=>0.33, "random_station_destroy!"=>0.33, "random_subtree_destroy!"=>0.33))
    old_destroy_kwargs = deepcopy(destroy_kwargs)
    return DestroyOp(string(destroy!), destroy!, destroy_kwargs, old_destroy_kwargs, destroy_weights)
end

function update_destroy_operator!(des::DestroyOp, new_destroy!::Function)
    des.destroy! = new_destroy!
    des.name = string(new_destroy!)
end

function ChangeOp(change!::Function, change_kwargs::Dict; change_weights::Dict=Dict{String, Float64}("no_change"=>0.2, "increase_destroy!"=>0.2, "decrement_y!"=>0.2, "change_destroy!"=>0.2, "change_destroy_increase_size!"=>0.2))
    return ChangeOp(change!, change_kwargs, change_weights)
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

function configure_change(search_strategy::Dict)
    #destroy operator change configuration
    if !haskey(search_strategy, "change")
        @info "No destroy change specified, defaulting to no_change"
        destroy_change = no_change
        search_strategy["change"] = Dict("kwargs"=>Dict("change_freq"=>10), 
                                    "change_weights"=>Dict("no_change"=>1.0, 
                                                        "increase_destroy!"=>1.0, 
                                                            "decrement_y!"=>1.0, 
                                                            "change_destroy!"=>1.0, 
                                                            "change_destroy_increase_size!"=>1.0))
    else
        if !haskey(search_strategy["change"], "kwargs")
            @info "No destroy change arguments specified, defaulting to change_freq=3"
            search_strategy["change"]["kwargs"] = Dict("change_freq"=>3)
        else 
            @info "Destroy change arguments specified: $(search_strategy["change"]["kwargs"])"
        end
        if search_strategy["change"]["operator"] == "increase_destroy!" || search_strategy["change"]["operator"] == "increase_destroy"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = increase_destroy!
        elseif search_strategy["change"]["operator"] == "no_change"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = no_change
        elseif search_strategy["change"]["operator"] == "decrement_y!"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = decrement_y!
            if !haskey(search_strategy["change"]["kwargs"], "fix_steps")
                @info "no fix steps specified, defaulting to 1"
                search_strategy["change"]["kwargs"]["fix_steps"] = 1
            end
        elseif search_strategy["change"]["operator"] == "change_destroy!"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = change_destroy!
        elseif search_strategy["change"]["operator"] == "change_destroy_increase_size!"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = change_destroy_increase_size!
        elseif search_strategy["change"]["operator"] == "adapt_lns!"
            @info "Deconstructor change operator $(search_strategy["change"]["operator"]) recognized"
            destroy_change = adapt_lns!
            if !haskey(search_strategy["change"], "change_weights")
                @info "no change reward specified, defaulting to 1 across all change operators"
                search_strategy["change"]["change_weights"] = Dict("no_change"=>1.0, "increase_destroy!"=>1.0, "decrement_y!"=>1.0, "change_destroy!"=>1.0)
            end
        else
            @error "Deconstructor change operator $(search_strategy["change"]) not recognized"
        end
    end
    #converst change_op kwargs to symbols
    search_strategy["change"]["kwargs"] = Dict(Symbol(k) => v for (k, v) in search_strategy["change"]["kwargs"])
    change_op = ChangeOp(destroy_change, search_strategy["change"]["kwargs"]; change_weights= search_strategy["change"]["change_weights"])
    return change_op
end

function configure_destroy(search_strategy::Dict)
    if !haskey(search_strategy, "destroy") || !haskey(search_strategy["destroy"], "operator")
        @info "No destroy specified, defaulting to random_station_destroy"
        search_strategy["destroy"] = Dict()
        destroy_op = random_station_destroy!
        search_strategy["destroy"]["kwargs"] = Dict(Symbol("n_destroy")=>2)
        search_strategy["change"]["operator"] = no_change
    else
        @info "Deconstructor specified: $(search_strategy["destroy"]["operator"])"
        destroy = search_strategy["destroy"]["operator"]
        if destroy == "random_station" || destroy == "random_station_destroy!"
            destroy_op = random_station_destroy!
        elseif destroy == "random_subtree" || destroy == "random_subtree_destroy!"
            destroy_op = random_subtree_destroy!
        elseif destroy == "random_model" || destroy == "random_model_destroy!"
            destroy_op = random_model_destroy!
        elseif destroy == "random_start" || destroy == "random"
            destroy_op = rand([random_station_destroy!, random_subtree_destroy!, random_model_destroy!])
            @info "Deconstructor operator $(destroy) recognized, randomly selected $(destroy_op) from destroy operators"
        else
            @error "Deconstructor operator $(destroy) not recognized"
        end
        if !haskey(search_strategy["destroy"], "kwargs")
            @info "No destroy arguments specified, defaulting to n_destroy=2"
            destroy_kwargs = Dict("n_destroy"=>2, "des_decay"=>0.9)
        else
            @info "Deconstructor arguments specified: $(search_strategy["destroy"]["kwargs"])"
            destroy_kwargs = search_strategy["destroy"]["kwargs"]
            #converts the keys to symbols
            destroy_kwargs = Dict(Symbol(k) => v for (k, v) in destroy_kwargs)
        end
        if !haskey(search_strategy["destroy"], "destroy_weights")
            @info "No destroy weights specified, defaulting to equal weights"
            search_strategy["destroy"]["destroy_weights"] = Dict("random_station_destroy!"=>0.33, "random_subtree_destroy!"=>0.33, "random_model_destroy!"=>0.33)
        else
            @info "Destroy weights specified: $(search_strategy["destroy"]["destroy_weights"])"
        end
    end 
        
    destroy_operator = DestroyOp(destroy_op, destroy_kwargs; destroy_weights=search_strategy["destroy"]["destroy_weights"])
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
    if !haskey(search_strategy, "adaptation")
        @info "No LNS adaptation specified, defaulting to no_adapt_lns"
        adaptation_technique = no_adapt_lns!
    else
        @info "LNS adaptation specified: $(search_strategy["adaptation"])"
        if search_strategy["adaptation"] == "adapt_lns!"
            adaptation_technique = adapt_lns!
        elseif search_strategy["adaptation"] == "adapt_lns_des!"
            adaptation_technique = adapt_lns_des!
        else
            @error "LNS adaptation operator $(search_strategy["adaptation"]) not recognized"
        end
    end
    #repair configuration
    repair_op = configure_repair(search_strategy)
    #destroy configuration
    destroy_op = configure_destroy(search_strategy)
    #change configuration
    change_op = configure_change(search_strategy)
    #setting the seed (if not none)
    if !haskey(search_strategy, "seed")
        search_strategy["seed"] = nothing
    end
    lns_obj = LNSConf(search_strategy["n_iterations"], 
    search_strategy["n_iter_no_improve"], 
    search_strategy["time_limit"], 
    repair_op, 
    destroy_op, 
    change_op,
    adaptation_technique,
    search_strategy["seed"])
    return lns_obj
end