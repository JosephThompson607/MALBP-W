function MMALBP_W_model_dependent(instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing,md_heuristic=task_equip_heuristic )
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "md/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "md/"* instance.name * "/"
    end
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_md_linear!(m, instance; preprocess=preprocessing, start_heuristic=md_heuristic)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    if save_lp
        @info "Writing LP to file $(output_filepath * "model.lp")"
        write_to_file(m, output_filepath * "model.lp")
    end
    save_results(output_filepath, m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_model_dependent_nonlinear(instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, md_heuristic=task_equip_heuristic)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "md/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "md/"* instance.name * "/"
    end
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_md_nonlinear!(m, instance; preprocess=preprocessing, start_heuristic=md_heuristic)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    save_results(output_filepath, m, run_time, instance, output_filepath, "model_dependent_problem_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "model_dependent_problem_nonlinear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_fixed(instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real;preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "fixed/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "fixed/"* instance.name * "/"
    end
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_fixed_linear!(m, instance; preprocess=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_fixed(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    save_results(output_filepath, m, run_time, instance, output_filepath, "fixed_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "fixed/", m, run_time, instance, output_filepath, "fixed_problem_linear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_fixed_nonlinear(instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real;preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "fixed/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "fixed/"* instance.name * "/"
    end
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_fixed_nonlinear!(m, instance; preprocess=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_fixed(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    save_results(output_filepath, m, run_time, instance, output_filepath, "fixed_problem_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "fixed/", m, run_time, instance, output_filepath, "fixed_problem_nonlinear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_dynamic( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false, )
    
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_dynamic_linear!(m, instance, preprocessing=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv")
    return m
end

function MMALBP_W_dynamic_nonlinear( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false, )
    
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    println("instance productivity", instance.productivity_per_worker)
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_dynamic_nonlinear!(m, instance, preprocessing=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "dynamic_problem_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_nonlinear_labor_recourse.csv")
    return m
end

function MMALBP_W_dynamic_ws( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_dynamic_linear!(m, instance, warmstart_vars, preprocessing=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val)
    return m
end

function MMALBP_W_dynamic_nonlinear_ws( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_dynamic_nonlinear!(m, instance, warmstart_vars, preprocessing=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "dynamic_problem_problem_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_nonlinear_labor_recourse.csv"; prev_obj_val=md_obj_val)
    return m
end

function MMALBP_W_dynamic_nonlinear_lns( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real, search_strategy_fp::String; rng,
        save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the  dyanmic model parameters
    define_dynamic_nonlinear!(m, instance, warmstart_vars, preprocessing=preprocessing)
    #solves the model in a lns loop
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=false)
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time, rng=rng)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "dynamic_problem_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_nonlinear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
    return m
end

function MMALBP_W_dynamic_lns( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real, search_strategy_fp::String; rng,
                                save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the  dyanmic model parameters
    define_dynamic_linear!(m, instance, warmstart_vars, preprocessing=preprocessing)
    #solves the model in a lns loop
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=false)
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time, rng=rng)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
    return m
end



# function MMALBP_W_dynamic_lns_dict( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real, search_strategy::Dict; 
#     save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
# #if directory is not made yet, make it
# if !isnothing(slurm_array_ind)
# output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
# else
# output_filepath = original_filepath * "dynamic/"* instance.name * "/"
# end
# if !isdir(output_filepath )
# mkpath(output_filepath)
# end
# #creates the model
# m = Model(optimizer)
# set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
# #defines the  dyanmic model parameters
# define_dynamic_linear!(m, instance, warmstart_vars, preprocessing=preprocessing)
# #solves the model in a lns loop
# obj_dict, best_obj = large_neighborhood_search!(m, instance, search_strategy; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time)
# if save_variables
# write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
# end
# #writes the model to a file
# if save_lp
# write_to_file(m, output_filepath * "model.lp")
# end
# #saves the objective function, relative gap, run time, and instance_name to a file
# save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
# return m
# end

function MMALBP_W_md_lns( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real, search_strategy_fp::String; rng, 
                                save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false, md_heuristic=task_equip_heuristic)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "md/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "md/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    set_optimizer_attribute(m, "TimeLimit", 60)
    #defines the  dyanmic model parameters
    start_value = define_md_linear!(m, instance; preprocess=preprocessing, start_heuristic=md_heuristic)
    #Does a quick initial solve to make sure we have a feasible solution
    optimize!(m)
    #If the initial solve is not feasible, we try to find an inital solution
    if primal_status(m) != MOI.FEASIBLE_POINT
        @info "looking for feasible initial solution"
        set_optimizer_attribute(m, "TimeLimit", 1200)
        optimize!(m)
    end
    x = all_variables(m)
    solution = value.(x)
    set_start_value.(x, solution)
    set_optimizer_attribute(m, "TimeLimit", run_time)
    #solves the model in a lns loop
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=true)
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=start_value, run_time=run_time, rng=rng)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "md_lns_results.csv"; best_obj_val = best_obj)
    return m
end


function MMALBP_W_md_nonlinear_lns( instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real, search_strategy_fp::String; rng, 
    save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false, md_heuristic=task_equip_heuristic)
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "md/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
    output_filepath = original_filepath * "md/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    set_optimizer_attribute(m, "TimeLimit", 60)
    #defines the  dyanmic model parameters
    start_value = define_md_nonlinear!(m, instance; preprocess=preprocessing, start_heuristic=md_heuristic)
    #Does a quick initial solve to make sure we have a feasible solution
    optimize!(m)
    #If the initial solve is not feasible, we try to find an inital solution
    if primal_status(m) != MOI.FEASIBLE_POINT
        @info "looking for feasible initial solution"
        set_optimizer_attribute(m, "TimeLimit", 1200)
        optimize!(m)
    end
    x = all_variables(m)
    solution = value.(x)
    set_start_value.(x, solution)
    set_optimizer_attribute(m, "TimeLimit", run_time)
    #solves the model in a lns loop
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=true)
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=start_value, run_time=run_time, rng=rng)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath, "md_lns_nonlinear_labor_recourse.csv")
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "md_lns_nonlinear_labor_recourse.csv"; best_obj_val = best_obj)
    return m
end

function MMALBP_from_yaml(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false, grb_threads::Int=1, rng=Xoshiro(),md_heuristic=task_equip_heuristic)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-ddTHH:MM")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for instance in instances
        for milp in config_file["milp_models"]
            @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, md_heuristic=md_heuristic)
            elseif milp == "model_dependent_problem_nonlinear_labor_recourse"
                m = MMALBP_W_model_dependent_nonlinear(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "dynamic_problem_linear_labor_recourse"
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "dynamic_problem_nonlinear_labor_recourse"
                m = MMALBP_W_dynamic_nonlinear(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp)
            end
        end
    end
end

function MMALBP_md_lns_from_yaml(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads::Int=1)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for instance in instances
        @info "Running instance $(instance.name), \n Output will be saved to $(output_filepath)"
        m = MMALBP_W_md_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, rng=rng)

    end
end

function MMALBP_md_lns_from_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, md_heuristic::Function=task_equip_heuristic_task_only_combined_precedence,rng=Xoshiro(), grb_threads=1)
    config_file, instance = read_slurm_csv(config_filepath,slurm_array_ind, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), \n Output will be saved to $(output_filepath), on slurm array index $(slurm_array_ind)"
    m = MMALBP_W_md_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, slurm_array_ind=slurm_array_ind, rng=rng, md_heuristic=md_heuristic)

end

function MMALBP_from_csv_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int ; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic::Union{Function, Nothing}=task_equip_heuristic)
    
    config_file, instance = read_slurm_csv(config_filepath, slurm_array_ind, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for milp in config_file["milp_models"]
            @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind,  preprocessing=preprocessing, md_heuristic=md_heuristic)
            elseif milp == "model_dependent_problem_nonlinear_labor_recourse"
                m = MMALBP_W_model_dependent_nonlinear(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "dynamic_problem_linear_labor_recourse"
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            elseif milp == "fixed_problem_nonlinear_labor_recourse"
                m = MMALBP_W_fixed_nonlinear(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            end
    end
end

function warmstart_dynamic(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false, grb_threads::Int=1)
    instances = read_md_results(config_filepath)
optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, preprocessing=true)
    end
end

function warmstart_dynamic_nonlinear_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, grb_threads::Int=1)
    instance, warmstart_vars_fp, md_obj_val = read_md_result(config_filepath, slurm_array_ind)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    MMALBP_W_dynamic_nonlinear_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= warmstart_vars_fp, md_obj_val= md_obj_val, preprocessing=preprocessing)
   
end


function oos_dynamic_test(config_filepath::String, original_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int;rng=Xoshiro(), xp_folder::String="model_runs", preprocessing::Bool=false, grb_threads::Int=1)
    instance, warmstart_vars_fp, prev_obj_val = read_md_result(config_filepath, slurm_array_ind)
    generate_new_sequences!(instance, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    original_filepath = xp_folder * "/" * now * "_" * original_filepath 
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(original_filepath)"
    #if directory is not made yet, make it
    if !isnothing(slurm_array_ind)
        output_filepath = original_filepath * "dynamic/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    else
        output_filepath = original_filepath * "dynamic/"* instance.name * "/"
    end
    if !isdir(output_filepath )
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_dynamic_linear_oos!(m, instance, warmstart_vars_fp, preprocessing=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=prev_obj_val)
    return m
end

function warmstart_dynamic_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, grb_threads::Int=1)
    instance, warmstart_vars_fp, md_obj_val = read_md_result(config_filepath, slurm_array_ind)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    MMALBP_W_dynamic_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= warmstart_vars_fp, md_obj_val= md_obj_val, preprocessing=preprocessing)
   
end

function MMALBP_W_LNS(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic=task_equip_heuristic_task_only)
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, preprocessing=false, rng=rng, md_heuristic=md_heuristic)
    end
end

# function MMALBP_from_csv_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int ; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic::Union{Function, Nothing}=task_equip_heuristic)
    
#     config_file, instance = read_slurm_csv(config_filepath, slurm_array_ind, rng=rng)
#     optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
#     now = Dates.now()
#     now = Dates.format(now, "yyyy-mm-dd")
#     output_filepath = xp_folder * "/" * now * "_" * output_filepath 
#     for milp in config_file["milp_models"]
#             @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
#             if milp== "model_dependent_problem_linear_labor_recourse"
#                 m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind,  preprocessing=preprocessing, md_heuristic=md_heuristic)
#             elseif milp == "model_dependent_problem_nonlinear_labor_recourse"
#                 m = MMALBP_W_model_dependent_nonlinear(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
#             elseif milp == "dynamic_problem_linear_labor_recourse"
#                 m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing)
#             elseif milp == "fixed_problem_linear_labor_recourse"
#                 m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
#             end
#     end
# end

function MMALBP_md_then_dynamic_lns(config_filepath::String, original_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int ; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic::Union{Function, Nothing}=task_equip_heuristic, md_run_time::Real=3600)
   
    config_file, instance = read_slurm_csv(config_filepath, slurm_array_ind, rng=rng)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => md_run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * original_filepath * "/slurm_" * string(slurm_array_ind) * "/"
    @info "Running model dependent with $md_run_time seconds and saving results to $output_filepath"
    m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, md_heuristic=md_heuristic)
    md_obj_val = objective_value(m)
    warmstart_vars_fp =  output_filepath * "md/"* instance.name *"/"
    @info "Running dynamic lns"
    MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables=save_variables, save_lp=save_lp, warmstart_vars=warmstart_vars_fp, md_obj_val=md_obj_val, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing, rng=rng )
end

#Runs the LNS model on a list of instances using a slurm array index
function MMALBP_W_LNS_md_nonlinear_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic=task_equip_heuristic_task_only, md=true)
    config_file, instance = read_slurm_csv(config_filepath, slurm_array_ind, rng=rng)

    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    m = MMALBP_W_md_nonlinear_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, slurm_array_ind=slurm_array_ind, rng=rng, md_heuristic=md_heuristic)

    #runner_function(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= warmstart_vars_fp, md_obj_val= md_obj_val, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing, rng=rng, md_heuristic=md_heuristic)
end

#Runs the LNS model on a list of instances using a slurm array index
function MMALBP_W_LNS_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), runner_function = MMALBP_W_dynamic_lns, grb_threads=1)
    instance, warmstart_vars_fp, md_obj_val = read_md_result(config_filepath, slurm_array_ind)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    runner_function(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= warmstart_vars_fp, md_obj_val= md_obj_val, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing, rng=rng)
end



#irace run for dynamic model LNS
function irace_LNS(md_results_fp::String, md_res_index::Int, lns_conf::LNSConf, output_filepath::String, run_time::Float64; 
                    xp_folder::String="model_runs", preprocessing::Bool=true, rng=Xoshiro(), grb_threads::Int64=1)
        #We are assuming each irace instance is a single instance
    instance, warmstart_vars_fp, md_obj_val = read_md_result(md_results_fp, md_res_index)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)   
    @info "Running instance $(instance.name) \n Output will be saved to $(output_filepath)"
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the  dyanmic model parameters
    define_dynamic_linear!(m, instance, warmstart_vars_fp, preprocessing=preprocessing)
    #solves the model in a lns loop
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time, rng=rng)
    #saves the solution variables
    write_MALBP_W_solution_dynamic(output_filepath , instance, m, false)
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath , "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
    println("Finished running instance $(instance.name). best_obj is $best_obj")
    return best_obj
end