
include("scenario_generators.jl")
using YAML
using CSV
using DataFrames

struct ModelInstance
    name::String
    probability::Real
    no_tasks::Int
    order_strength:: Real
    precendence_relations::Vector{Vector{String}}
    task_times::Dict{Int, Dict{String,Float64}}
end

struct ModelsInstance
    filepath::String
    name::String
    no_models::Int
    cycle_time:: Int
    models::Dict{String, ModelInstance}
end

struct EquipmentInstance
    filepath::String
    name::String
    no_stations::Int
    no_equipment::Int
    no_tasks::Int
    c_se :: Vector{Vector{Float64}}
    r_oe :: Vector{Vector{Int64}}
end



struct MALBP_W_instance
    filepath::String
    name::String
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




function calculate_scenarios(scenarios::DataFrame)
    return nrow(scenarios)
end

function MALBP_W_instance(filepath::String,config_name::String, models::ModelsInstance, scenarios::DataFrame, equipment::EquipmentInstance, no_stations:: Int, max_workers:: Int, worker_cost:: Int, recourse_cost:: Int, sequence_length:: Int, no_cycles:: Int, MILP_models::Array{String})
    no_scenarios = calculate_scenarios(scenarios)
    name =  models.name  * "_" *equipment.name
    return MALBP_W_instance(filepath,name,config_name, models, scenarios,no_scenarios, equipment, no_stations, max_workers, worker_cost, recourse_cost, sequence_length, no_cycles, MILP_models)
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
        file_name,
        name,
        no_stations,
        no_equipment,
        no_tasks,
        c_se,
        r_oe
    )
    return equip_instance
end



#Reads models instance
function read_models_instance(file_name :: String)
    models_yaml = YAML.load(open(file_name))
    models = Dict{String, ModelInstance}()
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
        models[name] = model_instance
    end
    no_models = length(models)
    models_instance = ModelsInstance(file_name,instance_name, no_models, cycle_time, models)
    return models_instance
end

#Gets the model mixture from the models instance. The model mixture is a dictionary with model name as
# key and probability as value
function get_model_mixture(models_instance::ModelsInstance)
    model_mixture = Dict{String, Float64}()
    for (model, model_dict) in models_instance.models
        model_mixture[model] = model_dict.probability
    end
    return model_mixture
end

#checks the instance for consistency
function check_instance(config_file, models_instance, equipment_instance)
    if config_file["no_stations"] != equipment_instance.no_stations
        error("number of stations in the config file does not match the number of stations in the equipment instance")
    end
    #If the tasks in the models instance are not in the equipment instance, throw an error
    for (model, model_dict) in models_instance.models
        for task in model_dict.task_times
            if task[1] > equipment_instance.no_tasks
                error("task $(task[1]) in model $(model) is not in the equipment instance")
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
            current_instance =MALBP_W_instance(file_name,
                        config_file["config_name"], 
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

#reads the scenario tree from a csv file
function read_scenario_csv(file_name::String)
    scenarios = CSV.read(file_name, DataFrame)
    #For each row in the scenario tree, the sequence is a vector of the sequence of models
    #Makes a dataframe to add rows to
    new_scenarios = []
    for row in eachrow(scenarios)
        #Removes everything outside of brackets in the sequence column
        new_seq = split(row.sequence, "]")[1]
        new_seq = split(new_seq, "[")[2]
        new_seq = split(string(new_seq), ",")
        new_seq = [replace(strip(x), "\"" => "") for x in new_seq]
        push!(new_scenarios, (sequence=new_seq, probability=row.probability))

    end
    new_scenarios = DataFrame(new_scenarios)
    return new_scenarios
end

#reads the instances from the results file of a model dependent run
function read_md_results(file_name::String; sequence_csv_name::String="sequences.csv")
    results = CSV.read(file_name, DataFrame)
    instances = []
    for row in eachrow(results)
        models_instance = read_models_instance(row.model_fp)
        equip_instance = read_equipment_instance(row.equip_fp)
        config_file = get_instance_YAML(row.instance_fp)
        scenarios_fp = row.output_folder * sequence_csv_name
        scenarios = read_scenario_csv(scenarios_fp)
        no_cycles = config_file["scenario"]["sequence_length"] + config_file["no_stations"] - 1
        current_instance = MALBP_W_instance(row.instance_fp,
                        config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equip_instance, 
                        config_file["no_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        config_file["scenario"]["sequence_length"],
                        no_cycles, 
                        config_file["milp_models"])
        push!(instances, (instance=current_instance, vars= row.output_folder, objective_value = row.objective_value))

    end
    return instances
end

#reads the instances from a csv file
function read_slurm_csv(file_name::String)
    results = CSV.read(file_name, DataFrame)
    instances = []
    for row in eachrow(results)
        models_instance = read_models_instance(row.model_yaml)
        equip_instance = read_equipment_instance(row.equipment_yaml)
        config_file = get_instance_YAML(row.config_yaml)
        if row.scenario_tree_yaml != "" && row.scenario_tree_yaml != "No Tree"
            scenarios = read_scenario_csv(row.scenario_tree_yaml)
        else
            scenarios = read_scenario_tree(config_file["scenario"], get_model_mixture(models_instance))
        end
        no_cycles = config_file["scenario"]["sequence_length"] + config_file["no_stations"] - 1
        current_instance = MALBP_W_instance(file_name,
                        config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equip_instance, 
                        config_file["no_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        config_file["scenario"]["sequence_length"],
                        no_cycles, 
                        config_file["milp_models"])
        push!(instances, (current_instance, config_file))

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
