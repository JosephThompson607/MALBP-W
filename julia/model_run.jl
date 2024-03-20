using ArgParse
using JuMP
using YAML
using CSV
using DataFrames
using Gurobi
const GRB_ENV = Gurobi.Env()
#user defined modules
include("scenario_generators.jl")
include("read_MALBP_W.jl") 
include("output.jl")
include("models/model_dependent.jl")
include("models/dynamic.jl")





function MMALBP_W_model_dependent(instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, output_filepath::String=""; save_variables::Bool=true, save_lp::Bool=false)
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_md_linear!(m, instance)
    #writes the model to a file
    optimize!(m)
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    if save_lp
        write_to_file(m, output_filepath * "model.lp")
    end
    return m
end

function MMALBP_W_dynamic( instance::MALBP_W_instance, optimizer::Gurobi.MathOptInterface.OptimizerWithAttributes, output_filepath::String=""; save_variables::Bool=true, save_lp::Bool=false)
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
    return m
end



function MMALBP_from_yaml(config_filepath::String, original_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool)
    config_file = get_instance_YAML(config_filepath)
    instances = read_MALBP_W_instances(config_filepath)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)
    for milp in config_file["milp_models"]
        for instance in instances
            name = instance.name
            @info "Running instance $(name), of model $(milp)"
            if milp== "model_dependent_problem_linear_labor_recourse"
                #if directory is not made yet, make it
                output_filepath = original_filepath * "md/"* name * "/"
                if !isdir(output_filepath)
                    mkpath(output_filepath)
                end
                m = MMALBP_W_model_dependent(instance, optimizer, output_filepath; save_variables= save_variables, save_lp=save_lp)
                #saves the objective function, relative gap, run time, and instance_name to a file
                save_results(original_filepath * "md/", m, run_time, instance, "model_dependent_problem_linear_labor_recourse.csv")
            elseif milp == "dynamic_problem_linear_labor_recourse"
                #if directory is not made yet, make it
                output_filepath = original_filepath * "dynamic/"* name * "/"
                if !isdir(output_filepath )
                    mkpath(output_filepath)
                end
                m = MMALBP_W_dynamic(instance, optimizer, output_filepath; save_variables= save_variables, save_lp=save_lp)
                save_results(original_filepath * "dynamic/", m, run_time, instance, "dynamic_problem_linear_labor_recourse.csv")
            end
        end
    end
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
        "--save_variables"
            help = "Save the solution variables"
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
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    output_file = "model_runs/" * parsed_args["output_file"] * "/"
    if parsed_args["xp_type"] == "config_yaml"
        MMALBP_from_yaml(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"] )
    end
end
main()

