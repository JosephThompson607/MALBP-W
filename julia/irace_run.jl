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
#user defined modules
include("scenario_generators.jl")
include("read_MALBP_W.jl") 
include("output.jl")
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
include("runner_functions.jl")
const GRB_ENV_REF = Ref{Gurobi.Env}()

function __init__()
    @suppress begin
    const GRB_ENV_REF[] = Gurobi.Env()
    end
    return
end

function irace_lns_config(;time_limit::Float64, 
                            name::String, 
                            destroy_list::Vector{String}, 
                            destroy::String, 
                            adaptive::String, 
                            percent_destroy::Float64, 
                            change_operator::String,  
                            weight_update::String, 
                            destroy_weights::Union{Nothing, 
                            Dict{String, Float64}}=nothing, 
                            repair_time::Float64, 
                            seed::Union{Nothing, Int}=nothing,
                            size_period::Int=1,
                            reward::Float64 = 1.0)
    
    #Destroy op config
    if isnothing(destroy_weights)
        destroy_weights = Dict{String, Float64}()
        for operator in destroy_list
            destroy_weights[operator] = 1
        end
    end

    destroy_list = parse_destroy_list(destroy_list)
    #We need to select from the destroy list if they specify random
    if destroy == "random_destroy" || destroy == "random"
        destroy = rand(destroy_list)
    else
        destroy = getfield(ModelRun, Symbol(destroy))
    end
    weight_update = getfield(ModelRun, Symbol(weight_update))
    destroy_kwargs = Dict(:percent_destroy => percent_destroy,
                         :des_decay => 0.9
                         )
    destroy_op = DestroyOp(destroy_list,
                            destroy,
                            destroy_kwargs,
                            destroy_kwargs,
                            destroy_weights,
                            weight_update)
    
    #Change op config
    change_operator = getfield(ModelRun, Symbol(change_operator))
    change_kwargs = Dict(:change_freq => 1, 
                        :filter_out_current => false,
                        :change_decay => 0.9,
                        :size_period => size_period,
                        :reward => reward)
    weight_update = getfield(ModelRun, Symbol(weight_update))
    change_weights =  Dict("no_change!"=>1.0, 
    "increase_size!"=>1.0, 
        "decrement_y!"=>1.0, 
        "change_destroy!"=>1.0, 
        "increase_repair_time!"=>1.0)
    change_op = ChangeOp(change_operator,
                        change_kwargs,
                        change_weights,
                        weight_update)
                        
    #Repair op config
    repair_operator = optimize!
    repair_kwargs = Dict(:time_limit => repair_time,
                        :mip_gap => 1e-2,
                        :mip_gap_decay => 0.95)
    repair_op = RepairOp(repair_operator, repair_kwargs)

    #LNS config
    adaptive = getfield(ModelRun, Symbol(adaptive))
    lns_conf = LNSConf(name,
                        2000,
                        100,
                        time_limit,
                        repair_op,
                        destroy_op,
                        change_op,
                        adaptive,
                        seed)
    return lns_conf

end



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "irace_config_id"
            help = "ID of the irace configuration"
            arg_type = String
        "irace_instance_id"
            help = "ID of the irace instance"
            arg_type = String
        "seed"
            help = "Seed for the random number generator"
            arg_type = Int
        "config_file" 
            help = "Filepath of main config file"
            arg_type = String
        "--run_time", "-t"
            help = "Maximum run time for the model"
            arg_type = Float64
            default = 4200.0
        "--output_file", "-o"
            help = "Name of the experiment"
            arg_type = String
            default = "xp"
        "--destroy_list", "-d"
            help = "what destroy operators to use"
            arg_type = String
            action = :store_arg
            nargs = '*'
        "--repair_time_limit"
            help = "Time limit for the repair heuristic"
            arg_type = Float64
            default = 600.0
        "--adaptive"
            help = "type of adaptive step to use"
            arg_type = String
            default = "no_adapt!"
        "--percent_destroy"
            help = "percent of the solution to destroy"
            arg_type = Float64
            default = 0.25
        "--change_operator"
            help = "change operator"
            arg_type = String
            default = "change_destroy!"
        "--weight_update"
            help = "how the destroy and change operator weights are adjusted"
            arg_type = String
            default = "basic_update"
        "--seed"
            help = "seed for the random number generator"
            arg_type = Int
            default = nothing
        "--preprocessing"
            help = "preprocess model constraints"
            arg_type = Bool
            default = true
        "--destroy"
            help = "destroy operator"
            arg_type = String
            default = "random_destroy"
        "--des_station_prob"
            help = "probability of destroy operator station"
            arg_type = Float64
            default = 1/3
        "--des_subtree_prob"
            help = "probability of destroy operator subtree"
            arg_type = Float64
            default = 1/3
        # "--des_model_prob", "-r"
        #     help = "probability of destroy operator model"
        #     arg_type = Float64
        #     default = 1/3
        "--reward"
            help = "Reward for ALNS"
            arg_type = Float64
            default = 1.0
        "--index", "-i"
            help = "index of the slurm array job"
            arg_type = Int
            required = true
        "--size_period"
            help = "period between increase of the size of the destroy block"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    des_model_prob =  max(0, 1 - args["des_station_prob"] - args["des_subtree_prob"])
    destroy_weights = Dict{String, Float64}("random_station_destroy!" => args["des_station_prob"], "random_subtree_destroy!" => args["des_subtree_prob"] , "random_model_destroy" =>des_model_prob )
    lns_conf = irace_lns_config(
        time_limit = args["run_time"],
        name = args["output_file"],
        destroy_list = args["destroy_list"],
        destroy = args["destroy"],
        adaptive = args["adaptive"],
        percent_destroy = args["percent_destroy"],
        change_operator = args["change_operator"],
        weight_update = args["weight_update"],
        repair_time = args["repair_time_limit"],
        seed = args["seed"],
        size_period = args["size_period"],
        reward = args["reward"]
    )
    
    result = Inf
    output = ""
    rng=Xoshiro(args["seed"])
    #Irace gets cranky if anything besides the obj value escapes
    err = @capture_err begin
    output = @capture_out begin
    result = irace_LNS(args["config_file"], args["index"], lns_conf, args["output_file"], args["run_time"]; preprocessing= args["preprocessing"], rng=rng)
    end
    end
    #writes output to file
    open("output.txt", "w") do io
        println(io, output)
    end
    #writes error to file
    open("error.txt", "w") do io
        println(io, err)
    end
    #Irace is too cool for return statements
    println(result)
    return result
end

export main
end

using .ModelRun
main()