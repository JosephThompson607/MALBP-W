

#defines the decision variables for the model dependent MALBP-W model
function define_md_linear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_soi[1:instance.equipment.no_stations, 1:instance.equipment.no_tasks, 1:instance.models.no_models], Bin, base_name="x_soi")
    @variable(m, u_se[1:instance.equipment.no_stations, 1:instance.equipment.no_equipment], Bin, base_name="u_se")
    @variable(m, instance.max_workers>=y_wts[1:instance.no_scenarios, 1:instance.no_cycles, 1:instance.no_stations] >=0, Int, base_name="y_wts")
    @variable(m, y_w[1:instance.no_scenarios]>=0, Int, base_name="y_w")
    @variable(m, y>=0, Int, base_name="y")

end
#defines the objective function of the model dependent MALBP-W model
function define_md_linear_obj!(m::Model, instance::MALBP_W_instance)
    #defines the objective function of the model
    y = m[:y]
    y_w = m[:y_w]
    u_se = m[:u_se]
    
    @objective(m, 
            Min, 
            instance.worker_cost * y + 
            instance.recourse_cost * sum(y_w[w] * instance.scenarios[w, "probability"] for w in 1:instance.no_scenarios) + 
            sum(instance.equipment.c_se[s][e] * u_se[s, e] for s in 1:instance.equipment.no_stations, e in 1:instance.equipment.no_equipment)
            )
end

#defines the constraints of the model dependent MALBP-W model
function define_md_linear_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_soi = m[:x_soi]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    #constraint 1: y_w and y must sum to the sum accross all stations of y_wts for each scenario and cycle
    for w in 1:instance.no_scenarios
        for t in 1:instance.no_cycles
        @constraint(m, y +  y_w[w] >= sum(y_wts[w, t, s] for s in 1:instance.equipment.no_stations))
        end
    end
    #constraint 2: each task is assigned to exactly one station
    for o in 1:instance.equipment.no_tasks
        for i in 1:instance.models.no_models
        #If the task is not a task of the model, then it is not assigned to any station
            if string(o) ∉ keys(instance.models.models[model_indexes[i]].task_times[1])
                continue
            end
            @constraint(m, sum(x_soi[s, o, i] for s in 1:instance.equipment.no_stations) == 1)
        end
        
    end
    # #constraint 3: sum of task times of each assigned task for each model must be less than the cycle time times the number of workers y_wts
    for w in eachrow(instance.scenarios)
        w_index = rownumber(w)
        for t in 1:instance.no_cycles
            for s in 1:instance.equipment.no_stations
                if 1 <= t - s +1<= instance.sequence_length 
                    j = t-s + 1
                    model = w["sequence"][j]
                    i = findfirst( ==(model), model_indexes) 
                    task_times = instance.models.models[model].task_times[1]
                    @constraint(m, sum(task_time * x_soi[s, parse(Int,o), i] for (o, task_time) in task_times) <= instance.models.cycle_time * y_wts[w_index, t, s])
                end
            end
        end
    end
    #constraint 4: tasks can only be assigned to stations that have the correct equipment
    for s in 1:instance.equipment.no_stations
        for o in 1:instance.equipment.no_tasks
            for i in 1:instance.models.no_models
                if string(o) ∉ keys(instance.models.models[model_indexes[i]].task_times[1])
                continue
            end
                @constraint(m, x_soi[s, o, i] <= sum(instance.equipment.r_oe[o][e] * u_se[s, e] for e in 1:instance.equipment.no_equipment))
            end
        end
    end
    #constraint 5: precedence precedence_relations
    for i in 1:instance.models.no_models
        model = model_indexes[i]
        for k in 1:instance.equipment.no_stations
            for (prec, suc) in instance.models.models[model].precendence_relations
                    @constraint(m, sum( x_soi[s, parse(Int,prec), i] for s in 1:k) >= sum( x_soi[s, parse(Int,suc), i] for s in 1:k))
            end
        end
    end
    #reduntant precedence constraints
    # for i in 1:instance.models.no_models
    #     model = model_indexes[i]
    #     for (prec, suc) in instance.models.models[model].precendence_relations
    #             @constraint(m, sum( s * x_soi[s, parse(Int,prec), i] for s in 1:instance.equipment.no_stations) <= sum( s * x_soi[s, parse(Int,suc), i] for s in 1:instance.equipment.no_stations))
    #     end
    # end
end

#defines the redundant constraints of the model dependent MALBP-W model
function define_md_linear_redundant_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_soi = m[:x_soi]
    #PREPROCESSING 1: TASKS cannot be to early or too late
    #calculates the infeasible task assignments
    infeasible_tasks_forward, infeasible_tasks_backwards = get_infeasible_task_assignments(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
    #prohibits the infeasible task assignments
    for (model, tasks) in infeasible_tasks_forward
        i = findfirst( ==(model), model_indexes) 
        for (task, stations) in tasks
            for station in stations
                @constraint(m, x_soi[station, parse(Int,task), i] == 0)
            end
        end
    end
    for (model, tasks) in infeasible_tasks_backwards
        i = findfirst( ==(model), model_indexes) 
        for (task,stations) in tasks
            for station in stations
                @constraint(m, x_soi[station,parse(Int,task), i] == 0)
            end
        end
    end
    #PREPROCESSING 2: Pairs of tasks that take up a large amount of time cannot both be started too late or too early
    #calculates the infeasible task pairs
    infeasible_pairs_forward, infeasible_pairs_backwards = get_infeasible_assignment_pairs(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.)) 
    #prohibits the infeasible task pairs
    for (model, pairs) in infeasible_pairs_forward
        i = findfirst( ==(model), model_indexes) 
        for ((task1, task2), stations) in pairs
            for station in stations
                @constraint(m, x_soi[station, parse(Int,task1), i] + x_soi[station, parse(Int,task2), i] <= 1)
            end
        end
    end
    for (model, pairs) in infeasible_pairs_backwards
        i = findfirst( ==(model), model_indexes) 
        for ((task1, task2), stations) in pairs
            for station in stations
                @constraint(m, x_soi[station, parse(Int,task1), i] + x_soi[station, parse(Int,task2), i] <= 1)
            end
        end
    end
end


#This function is used to set the initial values of the model dependent MALBP-W model
function set_initial_values!(m::Model, instance::MALBP_W_instance; 
                                    x_soi_start::Array{Int,3} = zeros(Int, instance.equipment.no_stations, instance.equipment.no_tasks, instance.models.no_models),
                                    y_start::Int = 0,
                                    y_w_start::Array{Int,1} = zeros(Int, instance.no_scenarios),
                                    y_wts_start::Array{Int,3} = zeros(Int, instance.no_scenarios, instance.no_cycles, instance.equipment.no_stations),
                                    equipment_assignments:: Dict{Int64, Vector{Int64}} = zeros(Int, instance.equipment.no_stations, instance.equipment.no_equipment)) 
    
    model_indexes = [i for (i, model_dict) in instance.models.models]
    x_soi = m[:x_soi]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    u_se = m[:u_se]

    #assigns tasks to stations
    for s in 1:instance.equipment.no_stations
        for o in 1:instance.equipment.no_tasks
            for i in 1:instance.models.no_models
                set_start_value(x_soi[s, o, i], x_soi_start[s, o, i])
            end
        end
    end

    #assigns workers to stations
    for w in 1:instance.no_scenarios
        set_start_value(y_w[w], y_w_start[w])
        for t in 1:instance.no_cycles
            for s in 1:instance.equipment.no_stations
                    set_start_value(y_wts[w, t, s], y_wts_start[w, t, s])
            end
        end
    end
    set_start_value(y, y_start)
    #assigns equipment to stations
    for s in 1:instance.equipment.no_stations
        for e in 1:instance.equipment.no_equipment
            set_start_value(u_se[s, e], 0)
        end
    end
    for (station, equipment) in equipment_assignments
        for e in equipment
            set_start_value(u_se[station, e], 1.)
        end
    end
end



function define_md_linear!(m::Model, instance::MALBP_W_instance; preprocess = false, start_heuristic::Function = task_equip_heuristic)
    define_md_linear_vars!(m, instance)
    define_md_linear_obj!(m, instance)
    define_md_linear_constraints!(m, instance)   
    if preprocess
        @info "Preprocessing: adding redundant constraints to the model"
        define_md_linear_redundant_constraints!(m, instance)
        @info "using heuristic $(start_heuristic) for initial task and worker assignments"
        time_start = time()
        #task_assignments, equip_assignments = MMALBP_W_model_dependent_decomposition_solve(instance; preprocessing=false)
        model_task_assignments, y_start, y_w_start, y_wts_start, equipment_assignments  = start_heuristic(instance)
        total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y_start, y_w_start, instance)
        set_initial_values!(m, 
                            instance, 
                            x_soi_start = model_task_assignments, 
                            y_start = y_start, 
                            y_w_start = y_w_start, 
                            y_wts_start = y_wts_start, 
                            equipment_assignments = equipment_assignments)
        @info "Heuristic start time: ", time() - time_start
        return total_cost
    end
    return nothing
end


