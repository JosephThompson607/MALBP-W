
function get_u_se_values!(m::Model, instance::MALBP_W_instance; old_u_se::Union{Nothing, Dict{Int, Array}}=nothing)
    u_se = m[:u_se]
    c_se = instance.equipment.c_se
    u_se_values = Dict{Int, Array}()
    for s in 1:instance.equipment.n_stations
        if !isnothing(old_u_se)
            u_se_values[s] = old_u_se[s]
        else
            u_se_values[s] = []
        end
        for e in 1:instance.equipment.n_equipment
            if value(u_se[s, e]) > 0
                push!(u_se_values[s], value(u_se[s, e]))
                c_se[s][e] = 0
            end
        end
    end
    equip_inst = EquipmentInstance(instance.equipment.filepath, instance.equipment.name, instance.equipment.n_stations, instance.equipment.n_equipment, instance.equipment.n_tasks, c_se, instance.equipment.r_oe)
    return u_se_values, equip_inst
end



function get_task_assignments(m::Model, instance::MALBP_W_instance)
        #writes the solution to a file
        x = m[:x_soi]
        x_soi_solution = []
        for s in 1:instance.equipment.n_stations
            for o in 1:instance.equipment.n_tasks
                for i in 1:instance.models.n_models
                    x_soi_dict = Dict("station"=>s, "task"=>o, "model"=>i, "value"=>value(x[s, o, i]))
                    push!(x_soi_solution, x_soi_dict)
                end
            end
        end
        return x_soi_solution
    end

#Creates a MALBP_W_instance with only one model
function mixed_to_single(model::ModelInstance, instance::MALBP_W_instance; equipment::EquipmentInstance=instance.equipment)
    model_dict = Dict{String, ModelInstance}(model.name => model)
    models_inst = ModelsInstance(instance.models.filepath,
                                instance.models.name,
                                1,
                                instance.models.cycle_time,
                                model_dict
                                )
    scenario_df = DataFrame(sequence = [repeat([model.name], instance.sequences.sequence_length)], probability = [model.probability])
    new_instance = MALBP_W_instance(instance.filepath, 
                                    instance.name,
                                    instance.config_name,
                                    models_inst,
                                    scenario_df,
                                    1,
                                    equipment,
                                    instance.n_stations,
                                    instance.max_workers,
                                    instance.worker_cost,
                                    instance.recourse_cost,
                                    instance.sequences.sequence_length,
                                    instance.n_cycles,
                                    instance.MILP_models)
    return new_instance
end

function MMALBP_W_model_dependent_decomposition_solve(instance::MALBP_W_instance; preprocessing::Bool=false, run_time::Int=1000)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV), "TimeLimit" => run_time)

    m = Model(optimizer)
    #defines the model dependent parameters
    models_list = [model for (model_name, model) in instance.models.models]
    #sorts the models by their probability
    sort!(models_list, by = x -> x.probability, rev=true)
    assignments_dict = Dict{}()
    model = popfirst!(models_list)
    new_instance = mixed_to_single(model, instance)                           
    define_md_linear!(m, new_instance; preprocess=preprocessing)
    optimize!(m)
    u_se, new_equip = get_u_se_values!(m, new_instance)
    task_assignments = get_task_assignments(m, new_instance)
    assignments_dict[model.name] = task_assignments
    for model in models_list
        m = Model(optimizer)
        new_instance = mixed_to_single(model, instance; equipment=new_equip)                                 
        define_md_linear!(m, new_instance; preprocess=preprocessing)
        optimize!(m)
        u_se, new_equip = get_u_se_values!(m, new_instance; old_u_se=u_se)
        task_assignments = get_task_assignments(m, new_instance)
        assignments_dict[model.name] = task_assignments
    end
    return assignments_dict, u_se
end