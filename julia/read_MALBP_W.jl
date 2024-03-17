using CSV
using DataFrames
import YAML
include("scenario_generators.jl")

#reads scenario tree file from csv accepts a dictionary of scenario tree info and model mixtures(optional)
function read_scenario_tree(scenario_info::Dict, model_mixtures::Dict{String, Float64} )
    if scenario_info["generator"] == "read_csv"
        return CSV.read(scenario_info["filepath"], DataFrame)
    elseif scenario_info["generator"] == "full"
        return generate_scenario_tree(scenario_info["sequence_length"], model_mixtures)
    else
        error("unrecognized generator, currently we only support read_csv for scenario tree generation.")
    end
end


struct ModelInstance
    name::String
    probability::Real
    no_tasks::Int
    order_strength:: Real
    precendence_relations::Vector{Vector{String}}
    task_times::Dict{Int, Dict{String,Float64}}
end

struct ModelsInstance
    name::String
    no_models::Int
    cycle_time:: Int
    models::Vector{ModelInstance}
end

struct EquipmentInstance
    name::String
    no_stations::Int
    no_equipment::Int
    no_tasks::Int
    c_se :: Vector{Vector{Float64}}
    r_oe :: Vector{Vector{Int64}}
end



struct MALBP_W_instance
    config_name::String
    models::ModelsInstance
    scenarios::DataFrame
    no_scenarios::Int
    equipment::EquipmentInstance
    no_stations:: Int
    max_workers:: Int
    worker_cost:: Int
    recourse_cost:: Int
    sequence_length:: Int
    no_cycles:: Int
    MILP_models::Array{String}
end

function calculate_scenarions(scenarios::DataFrame)
    return nrow(scenarios)
end

function MALBP_W_instance(config_name::String, models::ModelsInstance, scenarios::DataFrame, equipment::EquipmentInstance, no_stations:: Int, max_workers:: Int, worker_cost:: Int, recourse_cost:: Int, sequence_length:: Int, no_cycles:: Int, MILP_models::Array{String})
    no_scenarios = calculate_scenarions(scenarios)
    return MALBP_W_instance(config_name, models, scenarios,no_scenarios, equipment, no_stations, max_workers, worker_cost, recourse_cost, sequence_length, no_cycles, MILP_models)
end



#reads MALBP-W instance file from yaml
function get_instance_YAML(file_name::String)
    return YAML.load(open(file_name))
end

#Reads equipment instance YAML and returns an equipment object
function read_equipment_instance(file_name::String)
    equip_yaml  = YAML.load(open(file_name))
    name = equip_yaml["name"]
    no_stations= equip_yaml["no_stations"]
    no_equipment = equip_yaml["no_equipment"]
    no_tasks = equip_yaml["no_tasks"]
    c_se = equip_yaml["c_se"]
    r_oe = equip_yaml["r_oe"]
    equip_instance = EquipmentInstance(
        name,
        no_stations,
        no_equipment,
        no_tasks,
        c_se,
        r_oe
    )
end



#Reads models instance
function read_models_instance(file_name :: String)
    models_yaml = YAML.load(open(file_name))
    models = []
    instance_name = models_yaml["name"]
    #if the cycle time is defined as "takt_time" in the yaml file, read instance
    if haskey(models_yaml, "takt_time")
        cycle_time = models_yaml["takt_time"]
    else
        cycle_time = models_yaml["cycle_time"]
    end
    #reads the model instances
    for (key, value) in models_yaml["model_data"]
        name = key
        probability = value["probability"]
        no_tasks  = value["num_tasks"]
        order_strength  = value["order_strength"]
        precedence_relations = value["precedence_relations"]
        task_times  = value["task_times"]
        #println(task_times)
        model_instance = ModelInstance(name, probability, no_tasks, order_strength, precedence_relations, task_times)
        push!(models, model_instance)
    end
    no_models = length(models)
    models_instance = ModelsInstance(instance_name, no_models, cycle_time, models)
    return models_instance
end

#Gets the model mixture from the models instance. The model mixture is a dictionary with model name as
# key and probability as value
function get_model_mixture(models_instance::ModelsInstance)
    model_mixture = Dict{String, Float64}()
    for model in models_instance.models
        model_mixture[model.name] = model.probability
    end
    return model_mixture
end

#checks the instance for consistency
function check_instance(config_file, models_instance, equipment_instance)
    if config_file["no_stations"] != equipment_instance.no_stations
        error("number of stations in the config file does not match the number of stations in the equipment instance")
    end
    #If the tasks in the models instance are not in the equipment instance, throw an error
    for model in models_instance.models
        for task in model.task_times
            if task[1] > equipment_instance.no_tasks
                error("task $(task[1]) in model $(model.name) is not in the equipment instance")
            end
        end
    end
end
#takes an instance filepath as an input, and returns an array of MALBP_W_instance struct
function read_MALBP_W_instances(file_name::String)
    config_file = get_instance_YAML(file_name)
    instances = []
    for model in config_file["model_files"]
        for equip in config_file["equipment_files"]
            models_instance = read_models_instance(model)
            equipment_instance = read_equipment_instance(equip)
            check_instance(config_file,models_instance, equipment_instance)
            scenarios = read_scenario_tree(config_file["scenario"], get_model_mixture(models_instance))
            no_cycles = config_file["scenario"]["sequence_length"] + config_file["no_stations"] - 1
            current_instance =MALBP_W_instance(config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equipment_instance, 
                        config_file["no_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        config_file["scenario"]["sequence_length"],
                        no_cycles, 
                        config_file["milp_models"])
            push!(instances, current_instance)
        end
    end
    return instances
end


#prints each line of scenario tree
function print_scenario_tree(scenario_tree)
    for row in eachrow(scenario_tree)
        println(row)
    end
end

# tree = read_scenario_tree("SALBP_benchmark/MM_instances/scenario_trees/5_takts_5_samples_3_models.csv")
# print_scenario_tree(tree)
#instances = read_MALBP_W_instances("SALBP_benchmark/MM_instances/julia_debug.yaml")
