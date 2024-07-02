
include("scenario_generators.jl")
using YAML
using CSV
using DataFrames

mutable struct ProdSequences
    sequences::DataFrame
    n_scenarios::Int
    generator::String
    sequence_length::Int
end

mutable struct ModelInstance
    name::String
    probability::Real
    n_tasks::Int
    order_strength:: Real
    precendence_relations::Vector{Vector{String}}
    task_times::Dict{Int, Dict{String,Float64}}
end

mutable struct ModelsInstance
    filepath::String
    name::String
    n_models::Int
    cycle_time:: Int
    models::Dict{String, ModelInstance}
end

mutable struct EquipmentInstance
    filepath::String
    name::String
    n_stations::Int
    n_equipment::Int
    n_tasks::Int
    c_se :: Vector{Vector{Float64}}
    r_oe :: Vector{Vector{Int64}}
end



struct MALBP_W_instance
    filepath::String
    name::String
    config_name::String
    models::ModelsInstance
    sequences::ProdSequences
    equipment::EquipmentInstance
    n_stations:: Int
    max_workers:: Int
    worker_cost:: Int
    recourse_cost:: Int
    num_cycles:: Int
    MILP_models::Array{String}
end




function calculate_scenarios(scenarios::DataFrame)
    return nrow(scenarios)
end

function MALBP_W_instance(filepath::String,config_name::String, models::ModelsInstance, sequences::ProdSequences, equipment::EquipmentInstance, n_stations:: Int, max_workers:: Int, worker_cost:: Int, recourse_cost:: Int, num_cycles:: Int, MILP_models::Array{String})
    name =  models.name  * "_" *equipment.name
    return MALBP_W_instance(filepath,name,config_name, models, sequences, equipment, n_stations, max_workers, worker_cost, recourse_cost,  num_cycles, MILP_models)
end



#reads MALBP-W instance file from yaml
function get_instance_YAML(file_name::String)
    return YAML.load(open(file_name))
end

#Reads equipment instance YAML and returns an equipment object
function read_equipment_instance(file_name::String)
    equip_yaml  = YAML.load(open(file_name))

    name = equip_yaml["name"]
    n_stations= equip_yaml["n_stations"]
    n_equipment = equip_yaml["n_equipment"]
    n_tasks = equip_yaml["n_tasks"]
    c_se = equip_yaml["c_se"]
    r_oe = equip_yaml["r_oe"]
    equip_instance = EquipmentInstance(
        file_name,
        name,
        n_stations,
        n_equipment,
        n_tasks,
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
        n_tasks  = value["num_tasks"]
        order_strength  = models_yaml["order_strength"][key]
        precedence_relations = value["precedence_relations"]
        #converts precedence relations to string
        precedence_relations = [string.(x) for x in precedence_relations]
        task_times  = value["task_times"]
        #converts the keys of the task times to strings
        for (key, value) in task_times
            task_times[key] = Dict{String, Float64}(string.(keys(value)) .=> values(value))
        end
        model_instance = ModelInstance(name, probability, n_tasks, order_strength, precedence_relations, task_times)
        models[name] = model_instance
    end
    n_models = length(models)
    models_instance = ModelsInstance(file_name,instance_name, n_models, cycle_time, models)
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
    if config_file["n_stations"] != equipment_instance.n_stations
        error("number of stations in the config file does not match the number of stations in the equipment instance")
    end
    #If the tasks in the models instance are not in the equipment instance, throw an error
    for (model, model_dict) in models_instance.models
        for task in model_dict.task_times
            if task[1] > equipment_instance.n_tasks
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
            num_cycles = config_file["scenario"]["sequence_length"] + config_file["n_stations"] - 1
            current_instance =MALBP_W_instance(file_name,
                        config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equipment_instance, 
                        config_file["n_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        num_cycles, 
                        config_file["milp_models"])
            push!(instances, current_instance)
        end
    end
    return instances
end


#reads the instances from the results file of a model dependent run
function read_md_result(file_name::String, res_index::Int; sequence_csv_name::String="sequences.csv")
    results = CSV.read(file_name, DataFrame)
    row = results[res_index, :]
    obj_val = row.objective_value
    models_instance = read_models_instance(row.model_fp)
    equip_instance = read_equipment_instance(row.equip_fp)
    config_file = get_instance_YAML(row.instance_fp)
    config_file = overwrite_config_settings(row, config_file, models_instance, equip_instance)
    scenarios_fp = row.output_folder * sequence_csv_name
    scenarios = read_scenario_csv(scenarios_fp)
    num_cycles = config_file["scenario"]["sequence_length"] + config_file["n_stations"] - 1
    current_instance = MALBP_W_instance(row.instance_fp,
                    config_file["config_name"], 
                    models_instance, 
                    scenarios, 
                    equip_instance, 
                    config_file["n_stations"], 
                    config_file["max_workers"], 
                    config_file["worker_cost"], 
                    config_file["recourse_cost"], 
                    num_cycles, 
                    config_file["milp_models"])
    return current_instance, row.output_folder, obj_val
end


#reads the instances from the results file of a model dependent run
function read_md_results(file_name::String; sequence_csv_name::String="sequences.csv")
    results = CSV.read(file_name, DataFrame)
    instances = []
    for row in eachrow(results)
        models_instance = read_models_instance(row.model_fp)
        equip_instance = read_equipment_instance(row.equip_fp)
        config_file = get_instance_YAML(row.instance_fp)
        config_file = overwrite_config_settings(row, config_file, models_instance, equip_instance)
        scenarios_fp = row.output_folder * sequence_csv_name
        scenarios = read_scenario_csv(scenarios_fp)
        num_cycles = config_file["scenario"]["sequence_length"] + config_file["n_stations"] - 1
        current_instance = MALBP_W_instance(row.instance_fp,
                        config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equip_instance, 
                        config_file["n_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        num_cycles, 
                        config_file["milp_models"])
        push!(instances, (instance=current_instance, vars= row.output_folder, objective_value = row.objective_value))

    end
    return instances
end

function overwrite_config_settings(row, config_file, models_instance, equip_instance)
    #If the row has n_stations, use the n_stations from the row
    if hasproperty(row, :n_stations) && row.n_stations != ""
        @info "Using number of stations from csv file: $(row.n_stations)" 
        config_file["n_stations"] = row.n_stations
    end
    #If the row has max_workers, use the max_workers from the row
    if hasproperty(row, :max_workers) && row.max_workers != ""
        @info "Using max workers from csv file : $(row.max_workers)"
        config_file["max_workers"] = row.max_workers
    end
    #If the row has worker_cost, use the worker_cost from the row
    if hasproperty(row, :worker_cost) && row.worker_cost != ""
        @info "Using worker cost from csv file : $(row.worker_cost)"
        config_file["worker_cost"] = row.worker_cost
    end
    #If the row has recourse_cost, use the recourse_cost from the row
    if hasproperty(row, :recourse_cost) && row.recourse_cost != ""
        @info "Using recourse cost from csv file : $(row.recourse_cost)"
        config_file["recourse_cost"] = row.recourse_cost
    end
    #If the row has sequence_length, use the sequence_length from the row
    if hasproperty(row, :sequence_length) && row.sequence_length != ""
        @info "Using sequence length from csv file : $(row.sequence_length)"
        config_file["scenario"]["sequence_length"] = row.sequence_length
    end
    if hasproperty(row, :n_scenarios) && row.n_scenarios != ""
        @info "Using number of scenarios from csv file : $(row.n_scenarios)"
        config_file["scenario"]["n_samples"] = row.n_scenarios
    end
    if hasproperty(row, :cycle_time) && row.cycle_time !=""
        @info "Using cycle time from csv file : $(row.cycle_time)"
        models_instance.cycle_time = row.cycle_time
    end
    return config_file

end



#reads the instances from a csv file
function read_slurm_csv(file_name::String, slurm_ind::Int)
    results = CSV.read(file_name, DataFrame)
    row = results[slurm_ind, :]
    models_instance = read_models_instance(row.model_yaml)
    equip_instance = read_equipment_instance(row.equipment_yaml)
    config_file = get_instance_YAML(row.config_yaml)
    config_file = overwrite_config_settings(row, config_file, models_instance, equip_instance)
    if hasproperty(row, :scenario_tree_yaml) && row.scenario_tree_yaml != "" && row.scenario_tree_yaml != "No Tree"
        scenarios = read_scenario_csv(row.scenario_tree_yaml)
    else
        scenarios = read_scenario_tree(config_file["scenario"], get_model_mixture(models_instance))
    end
    num_cycles = config_file["scenario"]["sequence_length"] + config_file["n_stations"] - 1
    current_instance = MALBP_W_instance(row.config_yaml,
                    config_file["config_name"], 
                    models_instance, 
                    scenarios, 
                    equip_instance, 
                    config_file["n_stations"], 
                    config_file["max_workers"], 
                    config_file["worker_cost"], 
                    config_file["recourse_cost"], 
                    num_cycles, 
                    config_file["milp_models"])
    return config_file, current_instance
end

#reads the instances from a csv file
function read_csv(file_name::String)
    results = CSV.read(file_name, DataFrame)
    instances = []
    for row in eachrow(results)
        models_instance = read_models_instance(row.model_yaml)
        equip_instance = read_equipment_instance(row.equipment_yaml)
        config_file = get_instance_YAML(row.config_yaml)
        config_file = overwrite_config_settings(row, config_file, models_instance, equip_instance)
        if row.scenario_tree_yaml != "" && row.scenario_tree_yaml != "No Tree"
            scenarios = read_scenario_csv(row.scenario_tree_yaml)
        else
            scenarios = read_scenario_tree(config_file["scenario"], get_model_mixture(models_instance))
        end
        num_cycles = config_file["scenario"]["sequence_length"] + config_file["n_stations"] - 1
        current_instance = MALBP_W_instance(row.config_yaml,
                        config_file["config_name"], 
                        models_instance, 
                        scenarios, 
                        equip_instance, 
                        config_file["n_stations"], 
                        config_file["max_workers"], 
                        config_file["worker_cost"], 
                        config_file["recourse_cost"], 
                        config_file["scenario"]["sequence_length"],
                        num_cycles, 
                        config_file["milp_models"])
        push!(instances, (current_instance, config_file))

    end
    return instance
end



#prints each line of scenario tree
function print_scenario_tree(scenario_tree)
    for row in eachrow(scenario_tree)
        println(row)
    end
end

# tree = read_scenario_tree("SALBP_benchmark/MM_instances/scenario_trees/5_takts_5_samples_3_models.csv")
# print_scenario_tree(tree)
