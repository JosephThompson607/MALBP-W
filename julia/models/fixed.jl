

#defines the decision variables for the model dependent MALBP-W model
function define_fixed_linear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_so[1:instance.equipment.n_stations, 1:instance.equipment.n_tasks], Bin, base_name="x_so")
    @variable(m, u_se[1:instance.equipment.n_stations, 1:instance.equipment.n_equipment], Bin, base_name="u_se")
    @variable(m, instance.max_workers>=y_wts[1:instance.sequences.n_scenarios, 1:instance.num_cycles, 1:instance.n_stations] >=0, Int, base_name="y_wts")
    @variable(m, y_w[1:instance.sequences.n_scenarios]>=0, Int, base_name="y_w")
    @variable(m, y>=0, Int, base_name="y")

end
#defines the objective function of the model dependent MALBP-W model
function define_fixed_linear_obj!(m::Model, instance::MALBP_W_instance)
    #defines the objective function of the model
    y = m[:y]
    y_w = m[:y_w]
    u_se = m[:u_se]
    
    @objective(m, 
            Min, 
            instance.worker_cost * y + 
            instance.recourse_cost * sum(y_w[w] * instance.sequences.sequences[w, "probability"] for w in 1:instance.sequences.n_scenarios) + 
            sum(instance.equipment.c_se[s][e] * u_se[s, e] for s in 1:instance.equipment.n_stations, e in 1:instance.equipment.n_equipment)
            )
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
        for t in 1:instance.num_cycles
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
        for t in 1:instance.num_cycles
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

#defines the redundant constraints of the model dependent MALBP-W model
# function define_fixed_linear_redundant_constraints!(m::Model, instance::MALBP_W_instance)
#     # usesful variables
#     model_indexes = [i for (i, model_dict) in instance.models.models]
#     #model variables
#     x_so = m[:x_so]
#     #PREPROCESSING 1: TASKS cannot be to early or too late
#     #calculates the infeasible task assignments
#     infeasible_tasks_forward, infeasible_tasks_backwards = get_infeasible_task_assignments(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
#     #prohibits the infeasible task assignments
#     for (model, tasks) in infeasible_tasks_forward
#         i = findfirst( ==(model), model_indexes) 
#         for (task, stations) in tasks
#             for station in stations
#                 @constraint(m, x_so[station, parse(Int,task), i] == 0)
#             end
#         end
#     end
#     for (model, tasks) in infeasible_tasks_backwards
#         i = findfirst( ==(model), model_indexes) 
#         for (task,stations) in tasks
#             for station in stations
#                 @constraint(m, x_so[station,parse(Int,task), i] == 0)
#             end
#         end
#     end
#     #PREPROCESSING 2: Pairs of tasks that take up a large amount of time cannot both be started too late or too early
#     #calculates the infeasible task pairs
#     infeasible_pairs_forward, infeasible_pairs_backwards = get_infeasible_assignment_pairs(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.)) 
#     #prohibits the infeasible task pairs
#     for (model, pairs) in infeasible_pairs_forward
#         i = findfirst( ==(model), model_indexes) 
#         for ((task1, task2), stations) in pairs
#             for station in stations
#                 @constraint(m, x_so[station, parse(Int,task1), i] + x_so[station, parse(Int,task2), i] <= 1)
#             end
#         end
#     end
#     for (model, pairs) in infeasible_pairs_backwards
#         i = findfirst( ==(model), model_indexes) 
#         for ((task1, task2), stations) in pairs
#             for station in stations
#                 @constraint(m, x_so[station, parse(Int,task1), i] + x_so[station, parse(Int,task2), i] <= 1)
#             end
#         end
#     end
# end

function heuristic_start_fixed!(m::Model, instance::MALBP_W_instance; 
                                    task_assign_func::Function = ehsans_heuristic, 
                                    worker_assign_func::Function = base_worker_assign_func, 
                                    equipment_assign_func::Function = base_equipment_assign_func)
    #assigns tasks to stations
    model_indexes = [i for (i, model_dict) in instance.models.models]
    x_so = m[:x_so]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    model_task_assignments = task_assign_func(instance)
    println("MODEL TASK ASSIGNMENTS: ", model_task_assignments)
    for (model, station_assignments) in model_task_assignments
        i = findfirst( ==(model), model_indexes) 
        for (station, tasks) in station_assignments
            for task in tasks
                set_start_value( x_so[station, parse(Int,task), i], 1)
            end
        end
    end
    #assigns workers to stations
    y_start, y_w_start, y_wts_start = worker_assign_func(instance, model_task_assignments)
    println("y_start: ", y_start)
    for w in 1:instance.sequences.n_scenarios
        set_start_value(y_w[w], y_w_start[w])
        for t in 1:instance.num_cycles
            for s in 1:instance.equipment.n_stations
                    set_start_value(y_wts[w, t, s], y_wts_start[w, t, s])
            end
        end
    end
    set_start_value(y, y_start)
    #assigns equipment to stations
    u_se = m[:u_se]
    equipment_assignments = equipment_assign_func(instance, model_task_assignments)
    for (station, equipment) in equipment_assignments
        for e in equipment
            set_start_value(u_se[station, e], 1.)
        end
    end


end

function define_fixed_linear!(m::Model, instance::MALBP_W_instance; preprocess = false)
    define_fixed_linear_vars!(m, instance)
    define_fixed_linear_obj!(m, instance)
    define_fixed_linear_constraints!(m, instance)   
    if preprocess
        @info "TODO Preprocessing: adding redundant constraints to the model"
        #define_fixed_linear_redundant_constraints!(m, instance)
        # @info "using heuristic for initial task and worker assignments"
        # time_start = time()
        # heuristic_start!(m, 
        #                 instance, 
        #                 task_assign_func = ehsans_heuristic, 
        #                 worker_assign_func = worker_assignment_heuristic, 
        #                 equipment_assign_func = greedy_equipment_assignment_heuristic)
        # @info "Heuristic start time: ", time() - time_start
    end
end


