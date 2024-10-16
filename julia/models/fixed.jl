

#defines the decision variables for the model dependent MALBP-W model
function define_fixed_linear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_so[1:instance.equipment.n_stations, 1:instance.equipment.n_tasks], Bin, base_name="x_so")
    @variable(m, u_se[1:instance.equipment.n_stations, 1:instance.equipment.n_equipment], Bin, base_name="u_se")
    @variable(m, instance.max_workers>=y_wts[1:instance.sequences.n_scenarios, 1:instance.n_cycles, 1:instance.n_stations] >=0, Int, base_name="y_wts")
    @variable(m, y_w[1:instance.sequences.n_scenarios]>=0, Int, base_name="y_w")
    @variable(m, y>=0, Int, base_name="y")

end
#defines the objective function of the model dependent MALBP-W model
function define_fixed_linear_obj!(m::Model, instance::MALBP_W_instance; same_e_cost=true)
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
function define_fixed_linear_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_so = m[:x_so]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    #creates the combined precedence relations and combined tasks
    combined_precendent = []
    combined_task_times = Dict{String, Float64}()
    for i in 1:instance.models.n_models
        model = model_indexes[i]
        for (prec, suc) in instance.models.models[model].precendence_relations
            if (prec, suc) in combined_precendent
                continue
            end
            push!(combined_precendent, (prec, suc))
        end
        for (task, time) in instance.models.models[model].task_times[1]
            if task in keys(combined_task_times)
                #Here I am being kind of harsh (normally you take the average), but this would provide a feasible solution to the model dependent
                combined_task_times[task] = max(time, combined_task_times[task])
            else
                combined_task_times[task] = time
            end
        end
    end
    #constraint 1: y_w and y must sum to the sum accross all stations of y_wts for each scenario and cycle
    for w in 1:instance.sequences.n_scenarios
        for t in 1:instance.n_cycles
        @constraint(m, y +  y_w[w] >= sum(y_wts[w, t, s] for s in 1:instance.equipment.n_stations))
        end
    end
    #constraint 2: each task is assigned to exactly one station
    for o in 1:instance.equipment.n_tasks
        for i in 1:instance.models.n_models
        #If the task is not a task of the model, then it is not assigned to any station
            if string(o) ∉ keys(combined_task_times)
                continue
            end
            @constraint(m, sum(x_so[s, o] for s in 1:instance.equipment.n_stations) == 1)
        end
        
    end
    # #constraint 3: sum of task times of each assigned task for each model must be less than the cycle time times the number of workers y_wts
    for w in eachrow(instance.sequences.sequences)
        w_index = rownumber(w)
        for t in 1:instance.n_cycles
            for s in 1:instance.equipment.n_stations
                @constraint(m, sum(task_time * x_so[s, parse(Int,o)] for (o, task_time) in combined_task_times) <= instance.models.cycle_time * y_wts[w_index, t, s])
            end
        end
    end
    #constraint 4: tasks can only be assigned to stations that have the correct equipment
    for s in 1:instance.equipment.n_stations
        for o in 1:instance.equipment.n_tasks
                if string(o) ∉ keys(combined_task_times)
                    continue
                end
                @constraint(m, x_so[s, o] <= sum(instance.equipment.r_oe[o][e] * u_se[s, e] for e in 1:instance.equipment.n_equipment))

        end
    end
    #constraint 5: precedence precedence_relations
    for k in 1:instance.equipment.n_stations
        for (prec, suc) in combined_precendent
                @constraint(m, sum( x_so[s, parse(Int,prec)] for s in 1:k) >= sum( x_so[s, parse(Int,suc)] for s in 1:k))
        end
    end

    #reduntant precedence constraints
    # for i in 1:instance.models.n_models
    #     model = model_indexes[i]
    #     for (prec, suc) in instance.models.models[model].precendence_relations
    #             @constraint(m, sum( s * x_so[s, parse(Int,prec), i] for s in 1:instance.equipment.n_stations) <= sum( s * x_so[s, parse(Int,suc), i] for s in 1:instance.equipment.n_stations))
    #     end
    # end
end


function set_initial_values_fixed!(m::Model, instance::MALBP_W_instance; 
                                    x_soi_start::Union{Array{Int,3},Nothing} = zeros(Int, instance.equipment.n_stations, instance.equipment.n_tasks, instance.models.n_models),
                                    y_start::Union{Int,Nothing} = 0,
                                    y_w_start::Union{Array{Int,1},Nothing} = zeros(Int, instance.sequences.n_scenarios),
                                    y_wts_start::Union{Array{Int,3},Nothing} = zeros(Int, instance.sequences.n_scenarios, instance.n_cycles, instance.equipment.n_stations),
                                    equipment_assignments:: Union{Dict{Int64, Vector{Int64}}, Nothing} = zeros(Int, instance.equipment.n_stations, instance.equipment.n_equipment)) 
    
    model_indexes = [i for (i, model_dict) in instance.models.models]
    x_so = m[:x_so]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    u_se = m[:u_se]
    if !isnothing(x_soi_start)
    #assigns tasks to stations
        for s in 1:instance.equipment.n_stations
            for o in 1:instance.equipment.n_tasks
                set_start_value(x_so[s, o,], x_soi_start[s, o, 1])
            end
        end
    end
    #assigns workers to stations
    if !isnothing(y_w_start)
        for w in 1:instance.sequences.n_scenarios
            set_start_value(y_w[w], y_w_start[w])
            for t in 1:instance.n_cycles
                for s in 1:instance.equipment.n_stations
                        set_start_value(y_wts[w, t, s], y_wts_start[w, t, s])
                end
            end
        end
        set_start_value(y, y_start)
    end
    if !isnothing(equipment_assignments)
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
end

function define_fixed_linear!(m::Model, instance::MALBP_W_instance; preprocess = false, start_heuristic::Union{Function,Nothing} = task_equip_heuristic_task_only_combined_precedence)
    define_fixed_linear_vars!(m, instance)
    define_fixed_linear_obj!(m, instance)
    define_fixed_linear_constraints!(m, instance)   
    if preprocess
        @info "TODO Preprocessing: adding redundant constraints to the model"
        #define_fixed_linear_redundant_constraints!(m, instance)
       
    end
    if !isnothing(start_heuristic)
          
        @info "using heuristic $(start_heuristic) for initial task and worker assignments"
        time_start = time()
        #task_assignments, equip_assignments = MMALBP_W_model_dependent_decomposition_solve(instance; preprocessing=false)
        model_task_assignments, y_start, y_w_start, y_wts_start, equipment_assignments  = start_heuristic(instance)
        
        set_initial_values_fixed!(m, 
                            instance, 
                            x_soi_start = model_task_assignments, 
                            y_start = y_start, 
                            y_w_start = y_w_start, 
                            y_wts_start = y_wts_start, 
                            equipment_assignments = equipment_assignments)
        @info "Heuristic start time: ", time() - time_start
        if !isnothing(equipment_assignments) && !isnothing(y_wts_start)
            total_cost = calculate_equip_cost(equipment_assignments, instance) + calculate_worker_cost(y_start, y_w_start, instance)
        else
            @info "Cannot calculate initial cost,Only have task assignments, returning large value to start"
            return 1e12
        end
        return total_cost
    end
    return nothing
end


