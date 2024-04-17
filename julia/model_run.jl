using ArgParse
using JuMP
using YAML
using CSV
using DataFrames
using Gurobi
using Dates
using Random
using StatsBase
const GRB_ENV = Gurobi.Env()
#user defined modules
include("scenario_generators.jl")
include("read_MALBP_W.jl") 
include("output.jl")
include("models/model_dependent.jl")
include("models/dynamic.jl")
include("models/fixed.jl")
include("lns/lns.jl")
include("heuristics/preprocessing.jl")
include("heuristics/constructive.jl")




function MMALBP_W_model_dependent(instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real;preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
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

function MMALBP_W_dynamic( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing)
    
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
    define_dynamic_linear!(m, instance)
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

function MMALBP_W_dynamic_ws( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real; save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing)
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
    define_dynamic_linear!(m, instance, warmstart_vars)
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

function MMALBP_W_dynamic_lns( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, original_filepath::String, run_time::Real, search_strategy::String; 
                                save_variables::Bool=true, save_lp::Bool=false, warmstart_vars::String="", md_obj_val::Real=0.0, slurm_array_ind::Union{Int, Nothing}=nothing)
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
    define_dynamic_linear!(m, instance, warmstart_vars)
    #solves the model in a lns loop
    obj_dict, best_obj = large_neighborhood_search!(m, instance, search_strategy; lns_res_fp= output_filepath  * "lns_results.csv", md_obj_val=md_obj_val, run_time=run_time)
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

function MMALBP_from_yaml(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
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
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp)
            end
        end
    end
end

function MMALBP_from_csv_slurm(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, slurm_array_ind::Int ; xp_folder::String="model_runs")
    
    instances = read_slurm_csv(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    (instance, config_file) = instances[slurm_array_ind]
    for milp in config_file["milp_models"]
            @info "Running instance $(instance.name), of model $(milp). \n Output will be saved to $(output_filepath)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            elseif milp == "dynamic_problem_linear_labor_recourse"
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            elseif milp == "fixed_problem_linear_labor_recourse"
                m = MMALBP_W_fixed(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, slurm_array_ind=slurm_array_ind)
            end
    end
end

function warmstart_dynamic(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs")
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-ddTHH:MM")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_ws(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val)
    end
end

function MMALBP_W_LNS(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String; xp_folder::String="model_runs")
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-ddTHH:MM")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    for (instance, var_folder, md_obj_val) in instances
        @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
        MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val)
    end
end

#Runs the LNS model on a list of instances using a slurm array index
function MMALBP_W_LNS(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool, search_strategy_fp::String, slurm_array_ind::Int; xp_folder::String="model_runs")
    instances = read_md_results(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath 
    instance, var_folder, md_obj_val = instances[slurm_array_ind]
    @info "Running instance $(instance.name), from $(config_filepath). \n Output will be saved to $(output_filepath)"
    MMALBP_W_dynamic_lns(instance, optimizer, output_filepath, run_time,  search_strategy_fp; save_variables= save_variables, save_lp=save_lp, warmstart_vars= var_folder, md_obj_val= md_obj_val, slurm_array_ind=slurm_array_ind)
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--xp_type", "-x"
            help = "Type of xp to run. Options are: 'config_yaml' "
            default = "config_yaml"
        "--config_file", "-c"
            help = "Filepath of main config file"
            arg_type = String
            required = true
        "--LNS_config", "-l"
            help = "filepath of LNS configuration file"
            arg_type = String
            required = false
        "--save_variables"
            help = "Save the solution variables"
            action = :store_true
        "--preprocessing"
            help = "preprocess model constraints"
            action = :store_true
        "--save_lp"
            help = "Save the model as a .lp file"
            action = :store_true
        "--run_time"
            help = "Maximum run time for the model"
            arg_type = Float64
            default = 3600.0
        "--output_file", "-o"
            help = "Name of the experiment"
            arg_type = String
            default = "xp"
        "--slurm_array_ind"
            help = "Index of the slurm array job"
            arg_type = Int
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    output_file =  parsed_args["output_file"] * "/"
    if parsed_args["xp_type"] == "config_yaml"
        MMALBP_from_yaml(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"]; preprocessing=parsed_args["preprocessing"] )
    elseif parsed_args["xp_type"] == "warmstart"
        warmstart_dynamic(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"] )
    elseif parsed_args["xp_type"] == "csv_slurm"
        MMALBP_from_csv_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_array_ind"] )
        if isnothing(parsed_args["slurm_array_ind"])
            error("Slurm array index is required for slurm experiments")
        end
    elseif parsed_args["xp_type"] == "lns"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        end
        MMALBP_W_LNS(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"] )
    elseif parsed_args["xp_type"] == "slurm_array_lns"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_array_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_W_LNS(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_array_ind"] )
    end
end
main()

