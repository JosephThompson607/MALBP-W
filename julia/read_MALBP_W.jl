using CSV
using DataFrames
import YAML

#reads scenario tree file from csv
function read_scenario_tree(file_name)
    return CSV.read(file_name, DataFrame)
end


struct ModelInstance
    name::String
    probability::Real
    no_tasks::Int
    order_strength:: Real
    precendence_relations::Array{Int, 2}
    task_times::Array{Dict{String,Int}}
end

struct ModelsInstance
    Models::Array{ModelInstance}
    name::String
end

struct EquipmentInstance
    name::String
    no_stations::Int
    no_equipment::Int
    no_tasks::Int
    c_se :: Array{Int, 2}
    r_oe :: Array{Int, 2}
end



struct MALBP_W_instance
    name::String
    models::ModelsInstance
    scenarios::DataFrame
    equipment::EquipmentInstance
    no_stations:: Int
    no_models:: Int
    max_workers:: Int
    cycle_time :: Int
    worker_cost:: Int
    recourse_cost:: Int
    sequence_length:: Int
    MILP_models::Array{String}
end



#reads MALBP-W instance file from yaml
function read_instance(file_name::String)
    return YAML.load(open(file_name))
end

#takes an instance filepath as an input, and returns an array of MALBP_W_instance struct
function read_MALBP_W_instances(file_name::String)
    config_file = read_instance(file_name)
    instances = []
    for model in config_file["model_files"]
        models_instance = get_models_instance(model)
        equipment_instance = get_equipment_instance(config_file["equipment"])
        scenarios = read_scenario_tree(config_file["scenario_generator"])
        current_instance =MALBP_W_instance(config_file["name"], 
                    models_instance, 
                    scenarios, 
                    equipment_instance, 
                    config_file["no_stations"], 
                    config_file["no_models"], 
                    config_file["max_workers"], 
                    config_file["cycle_time"], 
                    config_file["worker_cost"], 
                    config_file["recourse_cost"], 
                    config_file["sequence_length"], 
                    config_file["MILP_models"])
        push!(instances, current_instance)
    end
    return instances
end
#Gets scenario tree, prints each line of scenario tree
function print_scenario_tree(scenario_tree)
    for row in eachrow(scenario_tree)
        println(row)
    end
end

# tree = read_scenario_tree("SALBP_benchmark/MM_instances/scenario_trees/5_takts_5_samples_3_models.csv")
# print_scenario_tree(tree)
MMALBP_yaml = read_instance("SALBP_benchmark/MM_instances/small_instance_debug.yaml")

