function MMALBP_W_model_dependent(instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real; preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
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
    define_md_linear!(m, instance; preprocess=preprocessing)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_fixed(instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real;preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
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
    save_results(original_filepath * "fixed/", m, run_time, instance, output_filepath, "fixed_problem_linear_labor_recourse.csv")
    
    return m
end

function MMALBP_W_dynamic( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false, )
    
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
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv")
    return m
end

function MMALBP_W_dynamic_ws( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=false)
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
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val)
    return m
end

function MMALBP_W_dynamic_lns( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real, search_strategy_fp::String; 
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
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time)
    if save_variables
        write_MALBP_W_solution_dynamic(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(original_filepath * "dynamic/", m, run_time, instance, output_filepath, "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
    return m
end



# function MMALBP_W_dynamic_lns_dict( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real, search_strategy::Dict; 
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

function MMALBP_W_md_lns( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real, search_strategy::String; 
                                save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing, preprocessing::Bool=true)
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
    #defines the  dyanmic model parameters
    start_value = define_md_linear!(m, instance; preprocess=true)
    #solves the model in a lns loop
    lns_conf = read_search_strategy_YAML(search_strategy_fp, run_time, model_dependent=true)
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=start_value, run_time=run_time)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    #writes the model to a file
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(original_filepath * "md/", m, run_time, instance, output_filepath, "md_lns_results.csv"; best_obj_val = best_obj)
    return m
end

function MMALBP_from_yaml(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-ddTHH:MM")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for instance in instances
        for milp in config_file["milp_models"]
            @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "dynamic_problem_linear_labor_recourse"
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp)
            end
        end
    end
end

function MMALBP_md_lns_from_yaml(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String; xp_folder::String="model_runs", preprocessing::Bool=false)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for instance in instances
        @info "Running instance $(instance.name), \n Output will be saved to $(output_filepath)"
        m = MMALBP_W_md_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing)

    end
end

function MMALBP_md_lns_from_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false)
    config_file = get_instance_YAML(config_filepath)
    instance, config_file = read_slurm_csv(config_filepath,slurm_array_ind)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    @info "Running instance $(instance.name), \n Output will be saved to $(output_filepath), on slurm array index $(slurm_array_ind)"
    m = MMALBP_W_md_lns(instance, optimizer, output_filepath, run_time, search_strategy_fp; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, slurm_array_ind=slurm_array_ind)

end

function MMALBP_from_csv_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int ; xp_folder::String="model_runs", preprocessing::Bool=false)
    
    config_file, instance = read_slurm_csv(config_filepath, slurm_array_ind)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for milp in config_file["milp_models"]
            @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind,  preprocessing=preprocessing)
            elseif milp == "dynamic_problem_linear_labor_recourse"
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            end
    end
end

function warmstart_dynamic(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false)
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, preprocessing=false)
    end
end

function warmstart_dynamic_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false)
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    instance, var_folder, md_obj_val = instances[slurm_array_ind]
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    MMALBP_W_dynamic_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, preprocessing=false)
   
end

function MMALBP_W_LNS(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String; xp_folder::String="model_runs", preprocessing::Bool=false)
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, preprocessing=false)
    end
end

#Runs the LNS model on a list of instances using a slurm array index
function MMALBP_W_LNS(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false)
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    instance, var_folder, md_obj_val = instances[slurm_array_ind]
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, slurm_array_ind=slurm_array_ind, preprocessing=preprocessing)
end



#irace run for dynamic model LNS
function irace_LNS(md_results_fp::String, md_res_index::Int, lns_conf::LNSConf, output_filepath::String, run_time::Float64; xp_folder::String="model_runs", preprocessing::Bool=true)

    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/"  * now * output_filepath * "/" * "$md_res_index" * "/"
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
        #We are assuming each irace instance is a single instance
    instance, warmstart_vars_fp, md_obj_val = read_md_result(md_results_fp, md_res_index)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time)
    @info "Running instance $(instance.name) \n Output will be saved to $(output_filepath)"
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the  dyanmic model parameters
    define_dynamic_linear!(m, instance, warmstart_vars_fp, preprocessing=preprocessing)
    #solves the model in a lns loop
    obj_dict, best_obj = large_neighborhood_search!(m, instance, lns_conf; lns_res_fp= output_filepath  , md_obj_val=md_obj_val, run_time=run_time)
    #saves the solution variables
    write_MALBP_W_solution_dynamic(output_filepath , instance, m, false)
    #saves the objective function, relative gap, run time, and instance_name to a file
    save_results(output_filepath, m, run_time, instance, output_filepath , "dynamic_problem_linear_labor_recourse.csv"; prev_obj_val=md_obj_val, best_obj_val = best_obj)
    println("Finished running instance $(instance.name). best_obj is $best_obj")
    return best_obj
end