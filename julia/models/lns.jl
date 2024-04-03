include("lns_config.jl")

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

function no_change(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    return iter_no_improve, lns_obj, m
end

function decrement_y!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        y = m[:y]
        println("Fixing y at ", start_value(y)-1)
        fix.(y, start_value(y)-1, force=true)

    elseif iter_no_improve % lns_obj.change.kwargs[:change_freq] in [1:lns_obj.change.kwargs[:fix_steps];];
        y = m[:y]
        println("Fixing y at ", start_value(y))
        fix.(y, start_value(y), force=true)
    end
    return iter_no_improve, lns_obj, m
end

function increase_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        lns_obj.des.kwargs[:n_destroy] += 1
    end
    return iter_no_improve, lns_obj, m
end

function change_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model;  
                            filter_out_current = true, _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        #randomly chooses from the destroy operators
        select_destroy!(lns_obj; filter_out_current=filter_out_current)
    end
    return iter_no_improve, lns_obj, m
end

#selects new destroy randomly from destroy operators using their weight
function select_destroy!(lns_obj::LNSConf; filter_out_current=true)
    operator_list = [random_station_destroy!, random_subtree_destroy!,random_model_destroy!]
    destroy_weights = copy(lns_obj.des.destroy_weights)
    #filters out the current operator
    if filter_out_current
        operator_list = filter(x -> x != lns_obj.des.destroy!, operator_list)
        delete!(destroy_weights, lns_obj.des.name)
    end
    println("destroy weights: ", destroy_weights)
    weights = collect(values(destroy_weights))
    destroy_names = collect(keys(destroy_weights))
    destroy_choice = sample(destroy_names, Weights(weights))
    println("destroy choice: ", destroy_choice)
    println("findfirst", findfirst(x -> string(x) == destroy_choice, operator_list))
    destroy = operator_list[findfirst(x -> string(x) == destroy_choice, operator_list)]
    update_destroy_operator!(lns_obj.des, destroy)
end

#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
function change_destroy_increase_size!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=true,  _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        if lns_obj.des.kwargs[:n_destroy] < lns_obj.des.kwargs[:destroy_limit] 
            lns_obj.des.kwargs[:n_destroy] += 1
        else
            #resets the size of block to destroy
            lns_obj.des.kwargs[:n_destroy]= lns_obj.des.old_kwargs[:n_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current)
        end
    end
    return iter_no_improve, lns_obj, m
end

function no_adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    return iter_no_improve, lns_obj, m
end

#adaptive lns for destroy operator selection
function adapt_lns_des!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    #decays the rewards of the destroy and change operator
    println("iter_no_improve: ", iter_no_improve)
    println("iteration: ", iteration)
    println("iteration_time: ", iteration_time)
    
    println("old weights: ", lns_obj.des.destroy_weights)
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        println("weight update", (1-decay) * 1 * iteration / (1 + iteration_time))
        lns_obj.des.destroy_weights[lns_obj.des.name] += (1-decay) * 1 * iteration / (1 + (iteration_time/10))
    end
    println("new weights: ", lns_obj.des.destroy_weights)
    return iter_no_improve, lns_obj, m
end

#adaptive lns for destroy and change operator selection
function adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    #retrieves the decay and change decay parameters
    decay = lns.des.kwargs[:des_decay]
    change_decay = lns.change.kwargs[:change_decay]
    #decays the rewards of the destroy and change operator
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    lns_obj.change.change_weights[str(lns_obj.change.change!)] *= change_decay
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] += (1-decay) * 1 * iteration / (1 + iteration_time)
        lns_obj.change.change_weights[str(lns_obj.change.change!)] += (1-change_decay) * 1 * iteration / (1 + iteration_time)
    end
    #selects new change operator
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        #randomly choses the destroy operator based on the rewards
        change_dict = lns_obj.change.change_weights
        change_names = collect(keys(change_dict))
        change_rewards = collect(values(operator_dict))
        change_list = [no_change, increase_destroy!, decrement_y!, change_destroy!]
        change_name = sample(change_names, Weights(change_rewards))
        change! = change_list[findfirst(x -> str(x) == change_name, change_list)]
        change!(iter_no_improve, lns_obj, m;  filter_out_current=false)
    end
    return iter_no_improve, lns_obj, m
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
        step_start = time()
        lns_conf.des.destroy!(m, instance, seed=seed ; lns_conf.des.kwargs...)
        #repairs using MILP TODO: add other repair operators
        optimize!(m)
        iteration_time = time() - step_start
        #saves the results
        res_dict = Dict("instance"=> instance.config_name,"iteration"=>i, "obj_val"=>objective_value(m), "time"=>iteration_time, "operator"=>lns_conf.des.name)
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
            x = all_variables(m)
            solution = value.(x)
            unfix_vars!(m, instance)
            set_start_value.(x, solution)
            lns_conf.adaptation!(iter_no_improve, lns_conf, m; iteration=i, iteration_time=iteration_time)
            lns_conf.change.change!(iter_no_improve, lns_conf, m; iteration=i, iteration_time=iteration_time, lns_conf.change.kwargs... )
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
