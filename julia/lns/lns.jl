include("lns_config.jl")
include("destroy_ops.jl")
include("shake_ops.jl")



function no_adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    return iter_no_improve, lns_obj, m
end

#adaptive lns for destroy operator selection
function adapt_lns_des!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    #decays the rewards of the destroy and change operator
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] += (1-decay) * 1 * iteration / (1 + (iteration_time/10))
    end
    return iter_no_improve, lns_obj, m
end

#adaptive lns for destroy and change operator selection
function adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    change_decay = lns_obj.change.kwargs[:change_decay]
    #decays the rewards of the destroy and change operator
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    lns_obj.change.change_weights[string(lns_obj.change.change!)] *= change_decay
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] += (1-decay) * 1 * iteration / (1 + (iteration_time/10))
        lns_obj.change.change_weights[string(lns_obj.change.change!)] += (1-change_decay) * 1 * iteration / (1 + (iteration_time/10))
    end
    #selects new change operator
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        #randomly choses the destroy operator based on the rewards
        change_dict = lns_obj.change.change_weights
        change_names = collect(keys(change_dict))
        change_rewards = collect(values(change_dict))
        change_list = [no_change!, increase_destroy!, decrement_y!, change_destroy!]
        change_name = sample(change_names, Weights(change_rewards))
        change! = change_list[findfirst(x -> string(x) == change_name, change_list)]
        lns_obj.change.change! = change!
    end
    return iter_no_improve, lns_obj, m
end

#function to unfix all fixed variables in the model and load their 
function unfix_vars!(m::Model , instance::MALBP_W_instance)
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    if is_fixed(y)
        unfix(y)
        set_lower_bound(y, 0)
    end
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
    for y_recorse in y_w
        if is_fixed(y_recorse)
            unfix(y_recorse)
            set_lower_bound(y_recorse, 0)
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
        res_dict = Dict("instance"=> instance.config_name,
                        "iteration"=>i, 
                        "obj_val"=>objective_value(m), 
                        "time"=>iteration_time, 
                        "operator"=>lns_conf.des.name,
                        "change_operator"=>string(lns_conf.change.change!))
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
            @info "iter_no_improve: $iter_no_improve , iteration: $i, operator: $(lns_conf.des.name), change_operator: $(lns_conf.change.change!)"
            lns_conf.change.change!(iter_no_improve, lns_conf, m; iteration=i, iteration_time=iteration_time, lns_conf.change.kwargs... )
        end

    end
    #writes the results to a csv
    obj_df = DataFrame(obj_vals)
    CSV.write(lns_res_fp, obj_df)
    #writes the lns_config to a yaml file
    YAML.write_file(lns_res_fp * "lns_config.yaml",lns_conf )
    println("best obj val: ", incumbent_dict)
    return incumbent_dict, incumbent
end
