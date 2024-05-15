include("lns_config.jl")
include("destroy_ops.jl")
include("change_ops.jl")
include("adapt_strategies.jl")




#function to unfix all fixed variables in the model and load their 
function unfix_vars!(m::Model , instance::MALBP_W_instance)
    

    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    if is_fixed(y)
        unfix(y)
        set_lower_bound(y, 0)
    end
    #unfixes task assignment variables
    #if the model has x_wsoj
    if haskey(m, :x_wsoj)
        x_wsoj = m[:x_wsoj]
        for x in x_wsoj
            if is_fixed(x)
                unfix(x)
            end
        end
   
    #if the model has x_soi
    elseif haskey(m, :x_soi)
        x_soi = m[:x_soi]
        for x in x_soi
            if is_fixed(x)
                unfix(x)
            end
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




function large_neighborhood_search!(m::Model, instance::MALBP_W_instance, search_strategy_fp::String; lns_res_fp::String="", md_obj_val::Union{Nothing, Float64}=nothing, run_time::Real=600.0, model_dependent=false )
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=model_dependent)
    set_destroy_size!(lns_conf.des, instance)
    seed = lns_conf.seed
    #sets the time limit for the model
    set_optimizer_attribute(m, "TimeLimit", lns_conf.rep.kwargs[:time_limit])
    #write_to_file(m, "my_model.mps")
    start_time = time()
    obj_vals = []
    incumbent = Inf
    incumbent_dict = Dict()
    #best_m = copy(m)
    #saves the initial objective value
    if !isnothing(md_obj_val)
        res_dict = Dict("instance"=> instance.config_name,"iteration"=>0, "obj_val"=>md_obj_val, "time"=>0.0, "operator"=>"initial", "change_operator"=> "initial", "destroy_size"=>0.0)
        push!(obj_vals, res_dict)
        incumbent = md_obj_val
    end

    #main loop of algorithm
    iter_no_improve = 0
    for i in 1: lns_conf.n_iterations
        println("mip repair kwargs: ", lns_conf.rep.kwargs)
        set_optimizer_attribute(m, "MIPGap", lns_conf.rep.kwargs[:mip_gap])
        lns_conf.rep.kwargs[:mip_gap] *= lns_conf.rep.kwargs[:mip_gap_decay]
        #calls the destroy operator
        step_start = time()
        lns_conf.des.destroy!(m, instance, seed=seed ; lns_conf.des.kwargs...)
        #repairs using MILP TODO: add other repair operators
        optimize!(m)
       
        #saves the results
        iteration_time = time() - step_start
        old_incumbent = incumbent
        obj_val_delta = incumbent - objective_value(m)
        res_dict = Dict("instance"=> instance.config_name,
                        "iteration"=>i, 
                        "obj_val"=>objective_value(m), 
                        "obj_val_delta"=>obj_val_delta,
                        "time"=>iteration_time, 
                        "operator"=>lns_conf.des.name,
                        "change_operator"=>string(lns_conf.change.change!),
                        "destroy_size"=>lns_conf.des.kwargs[:percent_destroy])
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
        elseif lns_conf.des.kwargs[:percent_destroy] >= 1.0 && iter_no_improve >= 5
            @info "solved full problem, no improvement"
            break
        end
        if i < lns_conf.n_iterations
            x = all_variables(m)
            solution = value.(x)
            unfix_vars!(m, instance)
            set_start_value.(x, solution)
            lns_conf.adaptation!(iter_no_improve, lns_conf, m; 
                                    obj_val_delta=obj_val_delta,
                                    iteration=i, 
                                    iteration_time=iteration_time, 
                                    des_weight_update = lns_conf.des.weight_update,
                                    change_weight_update = lns_conf.change.weight_update,
                                    prev_best = old_incumbent,
                                    current_best = incumbent)
            #Don't change the shaking operator in the first few iterations
            lns_conf.change.change!(iter_no_improve, lns_conf, m; iteration=i, iteration_time=iteration_time, lns_conf.change.kwargs... )
            @info "iter_no_improve: $iter_no_improve , iteration: $i, operator: $(lns_conf.des.name), change_operator: $(lns_conf.change.change!), 
            destroy size $(lns_conf.des.kwargs)"
        end

    end
    #writes the results to a csv
    obj_df = DataFrame(obj_vals)
    CSV.write(lns_res_fp  * "lns_results.csv", obj_df)
    println("conf_fp", lns_conf.conf_fp)
    cp(lns_conf.conf_fp, lns_res_fp* "lns_conf.yaml")
    println("best obj val: ", incumbent_dict)
    return incumbent_dict, incumbent
end
