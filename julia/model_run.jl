module ModelRun
using Suppressor
using ArgParse
using JuMP
using YAML
using CSV
using DataFrames
using Gurobi
using Dates
using Random
using StatsBase
const GRB_ENV_REF = Ref{Gurobi.Env}()


function __init__()
    global GRB_ENV_REF
    @suppress begin
    GRB_ENV_REF[] = Gurobi.Env()
    end
    return
end
#user defined modules
include("scenario_generators.jl")
include("read_MALBP_W.jl") 
include("output.jl")
include("models/nonlinear/dynamic_nonlinear.jl")
include("models/nonlinear/model_dependent_nonlinear.jl")
include("models/model_dependent.jl")
include("models/dynamic.jl")
include("models/fixed.jl")
include("lns/lns_config.jl")
include("lns/destroy_ops.jl")
include("lns/change_ops.jl")
include("lns/adapt_strategies.jl")
include("lns/lns.jl")
include("heuristics/preprocessing.jl")
include("heuristics/constructive.jl")
include("heuristics/md_warmstart.jl")
include("heuristics/fixed_constructive.jl")
include("heuristics/fixed_improvement.jl")
include("runner_functions.jl")


include("sample_tester.jl")


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
        "--md_heuristic"
            help = "heuristic for model dependent warmstart"
            arg_type = String
            default = "task_equip_heuristic"
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
        "--slurm_ind", "--slurm_array_ind"
            help = "Index of the slurm array job"
            arg_type = Int
        "--seed"
            help = "Seed for random number generator"
            required = false
            arg_type = Int
        "--grb_threads"
            help = "Number of threads for Gurobi"
            arg_type = Int
            default = 1
        "--scenario_generator"
            help = "how to generate production sequences. options are sobold_limit and monte_carlo_limit"
            arg_type = String
            default = "monte_carlo_limit"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    #If it has a seed, set the seed,
    if !isnothing(parsed_args["seed"])
        rng = Xoshiro(parsed_args["seed"])
    else
        rng = Xoshiro()
    end
    if parsed_args["md_heuristic"] == "none"
        parsed_args["md_heuristic"] = nothing
    else
        parsed_args["md_heuristic"] = getfield(ModelRun, Symbol(parsed_args["md_heuristic"]))
    end
    output_file =  parsed_args["output_file"] * "/"
    if parsed_args["xp_type"] == "config_yaml"
        MMALBP_from_yaml(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"]; preprocessing=parsed_args["preprocessing"], grb_threads=parsed_args["grb_threads"], md_heuristic = parsed_args["md_heuristic"])
    elseif parsed_args["xp_type"] == "warmstart"
        warmstart_dynamic(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"];  preprocessing=parsed_args["preprocessing"], grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "warmstart_slurm"
        if isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm experiments")
        end
        warmstart_dynamic_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_ind"] ;  preprocessing=parsed_args["preprocessing"], grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "warmstart_slurm_nonlinear"
        if isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm experiments")
        end
        warmstart_dynamic_nonlinear_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_ind"] ;  preprocessing=parsed_args["preprocessing"], grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "csv_slurm"
        MMALBP_from_csv_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_ind"];  preprocessing=parsed_args["preprocessing"], grb_threads=parsed_args["grb_threads"],md_heuristic=parsed_args["md_heuristic"] )
        if isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm experiments")
        end
    elseif parsed_args["xp_type"] == "lns_md"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        end
        MMALBP_md_lns_from_yaml(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"]; preprocessing=parsed_args["preprocessing"], rng= rng , grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "lns"
        if isnothing(parsed_args["LNS_config"]) && parsed_args["xp_type"] == "lns"
            error("LNS config file is required for LNS experiments")
        end
        MMALBP_W_LNS(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], preprocessing=parsed_args["preprocessing"], rng=rng, grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "slurm_lns_md" || parsed_args["xp_type"] == "slurm_array_lns_md"
        if isnothing(parsed_args["LNS_config"]) 
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_md_lns_from_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_ind"] ,rng=rng , grb_threads=parsed_args["grb_threads"], md_heuristic = parsed_args["md_heuristic"])
    elseif parsed_args["xp_type"] == "slurm_lns" || parsed_args["xp_type"] == "slurm_array_lns"
        if isnothing(parsed_args["LNS_config"]) 
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_W_LNS_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_ind"], rng=rng , grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "slurm_lns_nonlinear" || parsed_args["xp_type"] == "slurm_array_lns_nonlinear"
        if isnothing(parsed_args["LNS_config"]) 
            error("LNS config file is required for LNS experiments")
        elseif isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        MMALBP_W_LNS_slurm(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["LNS_config"], parsed_args["slurm_ind"], rng=rng, runner_function = MMALBP_W_dynamic_nonlinear_lns , grb_threads=parsed_args["grb_threads"])
    elseif parsed_args["xp_type"] == "md_sample_test_slurm"
        if isnothing(parsed_args["slurm_ind"])
            error("Slurm array index is required for slurm LNS experiments")
        end
        slurm_md_sample_test(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], parsed_args["slurm_ind"], rng=rng , grb_threads=parsed_args["grb_threads"], scenario_generator = parsed_args["scenario_generator"])
    elseif parsed_args["xp_type"] == "md_sample_test"
        md_sample_test(parsed_args["config_file"], output_file,parsed_args["run_time"], parsed_args["save_variables"], parsed_args["save_lp"], rng=rng , grb_threads=parsed_args["grb_threads"], scenario_generator = parsed_args["scenario_generator"])
    else
        error("Invalid xp_type")
    end
end

export main
end

using .ModelRun
main()