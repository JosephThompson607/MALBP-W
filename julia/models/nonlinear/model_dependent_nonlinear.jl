

#defines the decision variables for the model dependent MALBP-W model
function define_md_nonlinear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_soi[1:instance.equipment.n_stations, 1:instance.equipment.n_tasks, 1:instance.models.n_models], Bin, base_name="x_soi")
    @variable(m, u_se[1:instance.equipment.n_stations, 1:instance.equipment.n_equipment], Bin, base_name="u_se")
    @variable(m, y_lwts[1:instance.max_workers, 1:instance.sequences.n_scenarios, 1:instance.n_cycles, 1:instance.n_stations], Bin, base_name="y_lwts")
    @variable(m, y_w[1:instance.sequences.n_scenarios]>=0, Int, base_name="y_w")
    @variable(m, y>=0, Int, base_name="y")

end
#defines the objective function of the model dependent MALBP-W model
function define_md_nonlinear_obj!(m::Model, instance::MALBP_W_instance; same_e_cost=true)
    #defines the objective function of the model
    y = m[:y]
    y_w = m[:y_w]
    u_se = m[:u_se]
    if same_e_cost#default, equipment costs the same at every station
        @objective(m, 
                Min, 
                instance.worker_cost * y + 
                instance.recourse_cost * sum(y_w[w] * instance.sequences.sequences[w, "probability"] for w in 1:instance.sequences.n_scenarios) + 
                sum(instance.equipment.c_se[1][e] * u_se[s, e] for s in 1:instance.equipment.n_stations, e in 1:instance.equipment.n_equipment)
                )
    else
        @objective(m, 
                Min, 
                instance.worker_cost * y + 
                instance.recourse_cost * sum(y_w[w] * instance.sequences.sequences[w, "probability"] for w in 1:instance.sequences.n_scenarios) + 
                sum(instance.equipment.c_se[s][e] * u_se[s, e] for s in 1:instance.equipment.n_stations, e in 1:instance.equipment.n_equipment)
                )
    end
end

#defines the constraints of the model dependent MALBP-W model
function define_md_nonlinear_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_soi = m[:x_soi]
    u_se = m[:u_se]
    y_lwts = m[:y_lwts]
    y_w = m[:y_w]
    y = m[:y]
    #constraint 1: y_w and y must sum to the sum accross all stations of y_lwts for each scenario and cycle
    for w in 1:instance.sequences.n_scenarios
        for t in 1:instance.n_cycles
        @constraint(m, y +  y_w[w] >= sum(y_lwts[l, w, t, s] for s in 1:instance.equipment.n_stations for l in 1:instance.max_workers))
        end
    end
    #constraint 2: each task is assigned to exactly one station
    for o in 1:instance.equipment.n_tasks
        for i in 1:instance.models.n_models
        #If the task is not a task of the model, then it is not assigned to any station
            if string(o) ∉ keys(instance.models.models[model_indexes[i]].task_times[1])
                continue
            end
            @constraint(m, sum(x_soi[s, o, i] for s in 1:instance.equipment.n_stations) == 1)
        end
        
    end
   #constraint 3: sum of task times of each assigned task for each model must be less than the cycle time times the number of workers y_lwts
   for w in eachrow(instance.sequences.sequences)
    w_index = rownumber(w)
        for t in 1:instance.n_cycles
            for s in 1:instance.equipment.n_stations
                if 1 <= t - s +1<= instance.sequences.sequence_length 
                    j = t-s + 1
                    model = w["sequence"][j]
                    i = findfirst( ==(model), model_indexes) 
                    task_times = instance.models.models[model].task_times[1]
                    @constraint(m, sum(task_time * x_soi[s, parse(Int,o), i] for (o, task_time) in task_times) <= sum(instance.models.cycle_time * y_lwts[l, w_index, t, s] * instance.productivity_per_worker[l] for l in 1:instance.max_workers))
                end
            end
        end
    end
    #constraint 4: tasks can only be assigned to stations that have the correct equipment
    for s in 1:instance.equipment.n_stations
        for o in 1:instance.equipment.n_tasks
            for i in 1:instance.models.n_models
                if string(o) ∉ keys(instance.models.models[model_indexes[i]].task_times[1])
                continue
            end
                @constraint(m, x_soi[s, o, i] <= sum(instance.equipment.r_oe[o][e] * u_se[s, e] for e in 1:instance.equipment.n_equipment))
            end
        end
    end
    #constraint 5: precedence precedence_relations
    for i in 1:instance.models.n_models
        model = model_indexes[i]
        for k in 1:instance.equipment.n_stations
            for (prec, suc) in instance.models.models[model].precendence_relations
                    @constraint(m, sum( x_soi[s, parse(Int,prec), i] for s in 1:k) >= sum( x_soi[s, parse(Int,suc), i] for s in 1:k))
            end
        end
    end
    #Constraint 6: cannot assign another worker if the previous worker is not assigned
    for w in 1:instance.sequences.n_scenarios
        for t in 1:instance.n_cycles
            for s in 1:instance.equipment.n_stations
                for l in 2:instance.max_workers
                    @constraint(m, y_lwts[l,w, t, s] <= y_lwts[l-1,w, t, s])
                end
            end
        end
    end
    #reduntant precedence constraints
    # for i in 1:instance.models.n_models
    #     model = model_indexes[i]
    #     for (prec, suc) in instance.models.models[model].precendence_relations
    #             @constraint(m, sum( s * x_soi[s, parse(Int,prec), i] for s in 1:instance.equipment.n_stations) <= sum( s * x_soi[s, parse(Int,suc), i] for s in 1:instance.equipment.n_stations))
    #     end
    # end
end

#This function fixes the task assignments so the same task is at the same station accross models
function define_fixed_linear_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_soi = m[:x_soi]
    #constraint fix: each task is assigned to the same station for all models
    for i in 1:(instance.models.n_models-1)
        for i_prime in (i+1):instance.models.n_models
            @constraint(m, x_soi[:, :, i] .== x_soi[:, :, i_prime])
        end
    end

end

#defines the redundant constraints of the model dependent MALBP-W model
function define_md_nonlinear_redundant_constraints!(m::Model, instance::MALBP_W_instance;  productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_soi = m[:x_soi]
    #PREPROCESSING 1: TASKS cannot be to early or too late
    #calculates the infeasible task assignments
    infeasible_tasks_forward, infeasible_tasks_backwards = get_infeasible_task_assignments(instance;productivity_per_worker = productivity_per_worker)
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
    infeasible_pairs_forward, infeasible_pairs_backwards = get_infeasible_assignment_pairs(instance; productivity_per_worker = productivity_per_worker) 
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
function set_nonlinear_initial_values!(m::Model, instance::MALBP_W_instance; 
                                    x_soi_start::Array{Int,3} = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models),
                                    y_start::Int = 0,
                                    y_w_start::Array{Int,1} = zeros(Int, instance.sequences.n_scenarios),
                                    y_lwts_start::Array{Int,4} = zeros(Int, instance.max_workers, instance.sequences.n_scenarios, instance.n_cycles, instance.equipment.n_stations),
                                    equipment_assignments:: Dict{Int64, Vector{Int64}} = zeros(Int, instance.equipment.n_stations, instance.equipment.n_equipment)) 
    
    model_indexes = [i for (i, model_dict) in instance.models.models]
    x_soi = m[:x_soi]
    y_lwts = m[:y_lwts]
    y_w = m[:y_w]
    y = m[:y]
    u_se = m[:u_se]

    #assigns tasks to stations
    for s in 1:instance.equipment.n_stations
        for o in 1:instance.equipment.n_tasks
            for i in 1:instance.models.n_models
                set_start_value(x_soi[s, o, i], x_soi_start[s, o, i])
            end
        end
    end

    #assigns workers to stations

    for w in 1:instance.sequences.n_scenarios
        set_start_value(y_w[w], y_w_start[w])
        for t in 1:instance.n_cycles
            for l in 1:instance.max_workers
                for s in 1:instance.equipment.n_stations
                        set_start_value(y_lwts[l, w, t, s], y_lwts_start[l, w, t, s])
                end
            end
        end
    end

    set_start_value(y, y_start)
    #assigns equipment to stations
    for s in 1:instance.equipment.n_stations
        for e in 1:instance.equipment.n_equipment
            set_start_value(u_se[s, e], 0)
        end
    end
    for (station, equipment) in equipment_assignments
        for e in equipment
            set_start_value(u_se[station, e], 1.)
        end
    end
end

#Turns the integer heuristic solution into a binary solutions for number of workers
function get_lwts_start(y_wts_start, instance)
    y_lwts_start = zeros(Int, instance.max_workers, instance.sequences.n_scenarios, instance.n_cycles, instance.equipment.n_stations)
    for w in 1:instance.sequences.n_scenarios
        for t in 1:instance.n_cycles
            for s in 1:instance.equipment.n_stations
                for l in 1:y_wts_start[ w, t, s]
                y_lwts_start[l, w, t, s] = 1
            end
        end
    end
    end
    return y_lwts_start
end

function define_md_nonlinear!(m::Model, instance::MALBP_W_instance; preprocess = false, start_heuristic::Function = task_equip_heuristic)
    define_md_nonlinear_vars!(m, instance)
    define_md_nonlinear_obj!(m, instance)
    define_md_nonlinear_constraints!(m, instance)   
    if preprocess
        @info "Preprocessing: adding redundant constraints to the model"
        define_md_nonlinear_redundant_constraints!(m, instance, productivity_per_worker=instance.productivity_per_worker)
        @info "using heuristic $(start_heuristic) for initial task and worker assignments"
        time_start = time()

        model_task_assignments, y_start, y_w_start, y_wts_start, equipment_assignments  = start_heuristic(instance, productivity_per_worker=collect(values(instance.productivity_per_worker)))
        y_lwts_start = get_lwts_start(y_wts_start, instance)
        total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y_start, y_w_start, instance)
        set_nonlinear_initial_values!(m, 
                            instance, 
                            x_soi_start = model_task_assignments, 
                            y_start = y_start, 
                            y_w_start = y_w_start, 
                            y_lwts_start = y_lwts_start, 
                            equipment_assignments = equipment_assignments)
        @info "Heuristic start time: ", time() - time_start
        return total_cost
    end
    return nothing
end


function define_fixed_linear_from_md!(m::Model, instance::MALBP_W_instance; preprocess = false, start_heuristic::Function = task_equip_heuristic)
    define_md_nonlinear_vars!(m, instance)
    define_md_nonlinear_obj!(m, instance)
    define_md_nonlinear_constraints!(m, instance)   
    define_fixed_constraint!(m, instance)
end