




#function to unfix all fixed variables in the model and load their 
function unfix_vars!(m::Model , instance::MALBP_W_instance)
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
    u_se = m[:u_se]
    for u in u_se
        if is_fixed(u)
            unfix(u)
        end
    end

    #unfixes total recourse worker assignment
    y_w = m[:y_w]
    for y_recorse in y_w
        if is_fixed(y_recorse)
            unfix(y_recorse)
            set_lower_bound(y_recorse, 0)
        end
    end

    if haskey(m, :y_wts)
        #unfixes worker_assignment, reassigns max workers bounds
        y_wts = m[:y_wts]
    for worker_assignment in y_wts
        if is_fixed(worker_assignment)
            unfix(worker_assignment)
            set_upper_bound(worker_assignment, instance.max_workers)
            set_lower_bound(worker_assignment, 0)
        end
    end
    
    elseif haskey(m, :y_lwts)
        y_lwts = m[:y_lwts]
        #unfixes worker_assignment, reassigns max workers bounds
        for worker_assignment in y_lwts
            if is_fixed(worker_assignment)
                unfix(worker_assignment)
            end
        end
    end
    

end

function save_lns_conf(lns_conf::LNSConf, output_fp::String)
    open(output_fp, "w") do io
        println(io, "#LNS Configuration")
        println(io, "seed: ", lns_conf.seed)
        println(io, "n_iterations: ", lns_conf.n_iterations)
        println(io, "time_limit: ", lns_conf.time_limit)
        println(io, "des: ", string(lns_conf.des.destroy!))
        println(io, "change: ", lns_conf.change.change!)
        println(io, "rep: ", lns_conf.rep.kwargs)
        println(io, "adaptation: ", lns_conf.adaptation!)
        println(io, "conf_fp: ", lns_conf.conf_fp)
    end
end


function large_neighborhood_search!(m::Model, instance::MALBP_W_instance, lns_conf::LNSConf; lns_res_fp::String="", md_obj_val::Union{Nothing, Float64}=nothing, run_time::Real=600.0, rng=Xoshiro())
    set_destroy_size!(lns_conf.des, instance)
    seed = lns_conf.seed
    #sets the time limit for the model
    
    #write_to_file(m, "my_model.mps")
    start_time = time()
    obj_vals = []
    incumbent = Inf
    size_issue = false #keeps track of algorithm performance and makes sure it doesn't stall on too big of an instance
    incumbent_dict = Dict()
    #best_m = copy(m)
    #saves the initial objective value
    if !isnothing(md_obj_val)
        res_dict = Dict("instance"=> instance.config_name,"iteration"=>0, "obj_val"=>md_obj_val, "time"=>0.0, "operator"=>"initial", "change_operator"=> "initial", "destroy_size"=>0.0)
        push!(obj_vals, res_dict)
        incumbent = md_obj_val
    end
    #Save the last feasible solution
    previous_vars = all_variables(m)
    #main loop of algorithm
    iter_no_improve = 0
    for i in 1: lns_conf.n_iterations
        #For the last iteration, the time limit is the remaining time (if over time, we have 0 remaining time)
        last_iter_time = min(max(0.0,run_time - (time() - start_time)), lns_conf.rep.kwargs[:time_limit])
        set_optimizer_attribute(m, "TimeLimit", last_iter_time)
        gap = max(1e-4, lns_conf.rep.kwargs[:mip_gap])
        set_optimizer_attribute(m, "MIPGap", gap)
        lns_conf.rep.kwargs[:mip_gap] *= lns_conf.rep.kwargs[:mip_gap_decay]
        #calls the destroy operator
        step_start = time()
        lns_conf.des.destroy!(m, instance, seed=seed ; lns_conf.des.kwargs...)
        #repairs using MILP TODO: add other repair operators
        optimize!(m)
       
        #saves the results
        iteration_time = time() - step_start
        old_incumbent = incumbent
        #Sometimes the solver does not return a feasible solution
        if is_solved_and_feasible(m)
            previous_vars = all_variables(m)
            obj_val_delta = incumbent - objective_value(m)
            obj_val = objective_value(m)
        else
            #if not feasible, we just keep the same solution
            obj_val_delta = 0.0
            obj_val = incumbent
        end
        res_dict = Dict("instance"=> instance.config_name,
                        "iteration"=>i, 
                        "obj_val"=>obj_val, 
                        "obj_val_delta"=>obj_val_delta,
                        "time"=>iteration_time, 
                        "operator"=>string(lns_conf.des.destroy!),
                        "change_operator"=>string(lns_conf.change.change!),
                        "destroy_size"=>lns_conf.des.kwargs[:percent_destroy])
        push!(obj_vals, res_dict)
        if obj_val < incumbent
            incumbent = obj_val
            incumbent_dict = res_dict
            #best_m = copy(m)
            iter_no_improve = 0
        else
            iter_no_improve += 1
        end
        @info "The time is $(time() - start_time) and the iteration is $i, the incumbent is $incumbent, the obj_val_delta is $obj_val_delta, the iteration time is $iteration_time, the destroy size is $(lns_conf.des.kwargs[:percent_destroy])"
        if time() - start_time > lns_conf.time_limit
            @info "Time limit reached, stopping LNS at iteration $i"
            break
        elseif lns_conf.des.kwargs[:percent_destroy] >= 1.0 && iter_no_improve >= 3
            @info "solved full problem, no improvement"
            break
        end
        if i < lns_conf.n_iterations
            #If there is no lower bound, then we got stuck in the lp phase
            if relative_gap(m) >= 1.0
                println("TRIGGERED!!")
                size_issue= true
            end
            if primal_status(m) == MOI.FEASIBLE_POINT
                x = all_variables(m)
                solution = value.(x)
            else
                x = previous_vars
                solution = value.(x)
            end
            unfix_vars!(m, instance)
            set_start_value.(x, solution)
            lns_conf.adaptation!(iter_no_improve, lns_conf, m; 
                                    obj_val_delta=obj_val_delta,
                                    iteration=i, 
                                    iteration_time=iteration_time, 
                                    des_weight_update = lns_conf.des.weight_update,
                                    change_weight_update = lns_conf.change.weight_update,
                                    prev_best = old_incumbent,
                                    current_best = incumbent,
                                    )
            lns_conf.change.change!(iter_no_improve, lns_conf, m; iteration=i, iteration_time=iteration_time, rng=rng, size_issue=size_issue, lns_conf.change.kwargs... )
            @info "iter_no_improve: $iter_no_improve , iteration: $i, operator: $(lns_conf.des.destroy!), change_operator: $(lns_conf.change.change!), 
            destroy size $(lns_conf.des.kwargs)"
        end
        size_issue = false
    end
    #writes the results to a csv
    obj_df = DataFrame(obj_vals)
    CSV.write(lns_res_fp  * "lns_results.csv", obj_df)
    println("conf_fp", lns_conf.conf_fp)
    save_lns_conf(lns_conf, lns_res_fp* "lns_conf.yaml")
    #cp(lns_conf.conf_fp, lns_res_fp* "lns_conf.yaml", force=true)
    println("best obj val: ", incumbent)
    return incumbent_dict, incumbent
end
