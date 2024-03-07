using CSV
using DataFrames
import YAML

#reads scenario tree file from csv
function read_scenario_tree(scenario_info::Dict)
    if scenario_info["generator"] == "read_csv"
        return CSV.read(scenario_info["filepath"], DataFrame)
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
    equipment::EquipmentInstance
    no_stations:: Int
    max_workers:: Int
    worker_cost:: Int
    recourse_cost:: Int
    sequence_length:: Int
    no_cycles:: Int
    MILP_models::Array{String}
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
    cycle_time = models_yaml["takt_time"]
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


#takes an instance filepath as an input, and returns an array of MALBP_W_instance struct
function read_MALBP_W_instances(file_name::String)
    config_file = get_instance_YAML(file_name)
    instances = []
    println(config_file)
    for model in config_file["model_files"]
        for equip in config_file["equipment_files"]
            models_instance = read_models_instance(model)
            equipment_instance = read_equipment_instance(equip)
            scenarios = read_scenario_tree(config_file["scenario_generator"])
            no_cycles = config_file["sequence_length"] + config_file["no_stations"] - 1
            current_instance =MALBP_W_instance(config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equipment_instance, 
                        config_file["no_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        config_file["sequence_length"],
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
instances = read_MALBP_W_instances("SALBP_benchmark/MM_instances/julia_debug.yaml")
