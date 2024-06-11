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
include("heuristics/md_warmstart.jl")
include("runner_functions.jl")



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
        warmstart_dynamic(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"];  preprocessing=parsed_args["preprocessing"] )
    elseif parsed_args["xp_type"] == "warmstart_slurm"
        warmstart_dynamic_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_array_ind"] ;  preprocessing=parsed_args["preprocessing"] )
    elseif parsed_args["xp_type"] == "csv_slurm"
        MMALBP_from_csv_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_array_ind"];  preprocessing=parsed_args["preprocessing"] )
        if isnothing(parsed_args["slurm_array_ind"])
            error("Slurm array index is required for slurm experiments")
        end
    elseif parsed_args["xp_type"] == "lns_md"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        end
        MMALBP_md_lns_from_yaml(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"]; preprocessing=parsed_args["preprocessing"] )
    elseif parsed_args["xp_type"] == "lns"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        end
        MMALBP_W_LNS(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], preprocessing=parsed_args["preprocessing"] )
    elseif parsed_args["xp_type"] == "slurm_array_lns_md"
        if isnothing(parsed_args["LNS_config"]) 
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_array_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_md_lns_from_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_array_ind"] )
    elseif parsed_args["xp_type"] == "slurm_array_lns"
        if isnothing(parsed_args["LNS_config"]) 
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_array_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_W_LNS(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_array_ind"] )
    else
        error("Invalid xp_type")
    end
end
main()

