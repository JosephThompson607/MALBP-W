

#defines the decision variables for the dynamic MALBP-W model
function define_dynamic_linear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_wsoj[1:instance.sequences.n_scenarios, 1:instance.equipment.n_stations, 1:instance.equipment.n_tasks, 1:instance.sequences.sequence_length], Bin, base_name="x_wsoj")
    @variable(m, u_se[1:instance.equipment.n_stations, 1:instance.equipment.n_equipment], Bin, base_name="u_se")
    @variable(m, instance.max_workers>=y_wts[1:instance.sequences.n_scenarios, 1:instance.num_cycles, 1:instance.n_stations] >=0, Int, base_name="y_wts")
    @variable(m, y_w[1:instance.sequences.n_scenarios]>=0, Int, base_name="y_w")
    @variable(m, y>=0, Int, base_name="y")

end

#defines the objective function of the model dependent MALBP-W model
function define_dynamic_linear_obj!(m::Model, instance::MALBP_W_instance)
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

function add_non_anticipativity_constraints!(m::Model, instance::MALBP_W_instance, w::Int, w_prime::Int)
    #usesful variables
    x_wsoj = m[:x_wsoj]
    #constraint 1:  x_wsoj must be equal for shared sequence segments
    for t in instance.sequences.sequence_length:-1:1
        #if the sequence is the same up to time t, then x_wsoj must be equal up to time t
        if instance.sequences.sequences[w, "sequence"][1:t] == instance.sequences.sequences[w_prime, "sequence"][1:t]
            for j in 1:t
                max_station = min(t-j+1, instance.equipment.n_stations)
                for s in 1:max_station
                    for o in 1:instance.equipment.n_tasks
                        @constraint(m, x_wsoj[w, s, o, j] == x_wsoj[w_prime, s, o, j])
                    end
                end
            end
            return
        end
    end

end

#defines the constraints of the model dependent MALBP-W model
function define_dynamic_linear_constraints!(m::Model, instance::MALBP_W_instance)
    
    #model variables
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    #constraint 1: y_w and y must sum to the sum accross all stations of y_wts for each scenario and cycle
    for w in 1:instance.sequences.n_scenarios
        for t in 1:instance.num_cycles
        @constraint(m, y +  y_w[w] >= sum(y_wts[w, t, s] for s in 1:instance.equipment.n_stations))
        end
    end
    #constraint 2: each task is assigned to exactly one station
    for w in 1:instance.sequences.n_scenarios
            for o in 1:instance.equipment.n_tasks
                for j in 1:instance.sequences.sequence_length
                    #Do not need constraint if the task is not a task of the model
                    if string(o) ∉ keys(instance.models.models[string(instance.sequences.sequences[w,"sequence"][j])].task_times[1])
                        continue
                    end
                    @constraint(m, sum(x_wsoj[w, s, o, j] for s in 1:instance.n_stations) == 1)
                end
            end
    end
    #constraint 3: sum of task times of each assigned task for each model must be less than the cycle time times the number of workers y_wts
    for w in eachrow(instance.sequences.sequences)
        w_index = rownumber(w)
        for t in 1:instance.num_cycles
            for s in 1:instance.equipment.n_stations
                if 1 <= t - s +1<= instance.sequences.sequence_length 
                    j = t-s + 1
                    model = w["sequence"][j]
                    task_times = instance.models.models[model].task_times[1]
                    @constraint(m, sum(task_time * x_wsoj[w_index,s, parse(Int,o), j] for (o, task_time) in task_times) <= instance.models.cycle_time * y_wts[w_index, t, s])
                end
            end
        end
    end
    #constraint 4: tasks can only be assigned to stations that have the correct equipment
    for w in 1:instance.sequences.n_scenarios
        for s in 1:instance.equipment.n_stations
            for o in 1:instance.equipment.n_tasks
                for j in 1:instance.sequences.sequence_length
                    if string(o) ∉ keys(instance.models.models[instance.sequences.sequences[w,"sequence"][j]].task_times[1])
                    continue
                end
                    @constraint(m, x_wsoj[w,s, o, j] <= sum(instance.equipment.r_oe[o][e] * u_se[s, e] for e in 1:instance.equipment.n_equipment))
                end
            end
        end
    end
    #constraint 5: precedence relations
    for w in 1:instance.sequences.n_scenarios
        for j in 1:instance.sequences.sequence_length
            for k in 1:instance.equipment.n_stations
                model = instance.sequences.sequences[w,"sequence"][j]
                for (prec, suc) in instance.models.models[model].precendence_relations
                    @constraint(m, sum(x_wsoj[w,s, parse(Int,prec), j] for s in 1:k) >= sum( x_wsoj[w,s, parse(Int,suc), j] for s in 1:k))
                end
            end
        end
    end
    #Constraint 6: non-anticipativity
    for w in 1:instance.sequences.n_scenarios
        for w_prime in (w+1):instance.sequences.n_scenarios
            add_non_anticipativity_constraints!(m, instance, w, w_prime)
        end
    end
end

function read_MALBP_W_solution(results_folder::String)
    #Opens the x_soi file
    x_soi_fp = results_folder * "x_soi_solution.csv"
    x_soi_df = CSV.read(x_soi_fp, DataFrame)
    #Opens the u_se file
    u_se_fp = results_folder * "u_se_solution.csv"
    u_se_df = CSV.read(u_se_fp, DataFrame)
    #Opens the y_wts file
    y_wts_fp = results_folder * "y_wts_solution.csv"
    y_wts_df = CSV.read(y_wts_fp, DataFrame)
    #Opens the y_w file
    y_w_fp = results_folder * "y_solution.csv"
    y_w_df = CSV.read(y_w_fp, DataFrame)
    return x_soi_df, u_se_df, y_wts_df, y_w_df
end


function read_1st_stage_solution(results_folder::String)
    
    #Opens the u_se file
    u_se_fp = results_folder * "u_se_solution.csv"
    u_se_df = CSV.read(u_se_fp, DataFrame)

    #Opens the y_w file
    y_w_fp = results_folder * "y_solution.csv"
    y_w_df = CSV.read(y_w_fp, DataFrame)
    return u_se_df, y_w_df
end


function warmstart_dynamic_from_md_setup!(m::Model, vars_fp::String, instance::MALBP_W_instance)
    #usesful variables
    x_wsoj = m[:x_wsoj]
    u_se = m[:u_se]
    y_wts = m[:y_wts]
    y_w = m[:y_w]
    y = m[:y]
    #Gets the model depedent variables from the given folder
    x_soi_df, u_se_md_df, y_wts_md_df, y_w_md_df = read_MALBP_W_solution(vars_fp)

    #Sets the start value of the equipment variables if they are in the model dependent solution
    if !isempty(u_se_md_df)
        for row in eachrow(u_se_md_df)
            s = row.station
            e = row.equipment
            set_start_value(u_se[s, e], row.value)
        end
    end

    #Sets the start value of the worker y_w variables, if the scenario == fixed, then it is the y variable
    if !isempty(y_w_md_df)
        for row in eachrow(y_w_md_df)
            w = row.scenario
            if w == "fixed"
                set_start_value(y, row.value)
            else
                w = parse(Int, w)
                set_start_value(y_w[w], row.value)
            end
        end
    end

    #sets the start value of the y_wts variables
    if !isempty(y_wts_md_df)
        for row in eachrow(y_wts_md_df)
            w = row.scenario
            t = row.cycle
            s = row.station
            value = row.value
            set_start_value(y_wts[w, t, s], value)
        end
    end

    #Sets the start value of the task variables
    if !isempty(x_soi_df)
        model_indexes = [i for (i, model_dict) in instance.models.models]
        for row in eachrow(x_soi_df)
            s = row.station
            o = row.task
            i = row.model
            value = row.value
            for w in 1:instance.sequences.n_scenarios
                for j in 1:instance.sequences.sequence_length
                    if instance.sequences.sequences[w, "sequence"][j] == model_indexes[i]
                        set_start_value(x_wsoj[w, s, o, j], value)
                    end
                end
            end
        end
    end
end


function warmstart_dynamic_from_dynamic_and_fix!(m::Model, vars_fp::String)
    #usesful variables
    u_se = m[:u_se]
    y = m[:y]
    #Gets the relevant values from the given folder
    u_se_md_df, y_w_md_df = read_1st_stage_solution(vars_fp)

    #Sets the start value of the equipment variables if they are in the model dependent solution
    if !isempty(u_se_md_df)
        for row in eachrow(u_se_md_df)
            s = row.station
            e = row.equipment
            fix(u_se[s, e], row.value, force=true)
        end
    end

    #Sets the start value of the worker y_w variables, if the scenario == fixed, then it is the y variable
    if !isempty(y_w_md_df)
        for row in eachrow(y_w_md_df)
            w = row.scenario
            if w == "fixed"
                fix(y, row.value, force=true)
            end
        end
    end

end


function define_dynamic_linear_redundant_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_wsoj = m[:x_wsoj]
    #PREPROCESSING 1: TASKS cannot be to early or too late
    #calculates the infeasible task assignments
    infeasible_tasks_forward, infeasible_tasks_backwards = get_infeasible_task_assignments(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.))
    #prohibits the infeasible task assignments
    for (w,scenario) in enumerate(eachrow(instance.sequences.sequences))
        for j in 1:instance.sequences.sequence_length
            model = scenario.sequence[j]
            tasks = infeasible_tasks_forward[model]
            for (task, stations) in tasks
                for station in stations
                    @constraint(m, x_wsoj[w, station, parse(Int,task), j] == 0)
                end
            end
            model = scenario.sequence[j]
            tasks = infeasible_tasks_backwards[model]
            for (task, stations) in tasks
                for station in stations
                    @constraint(m, x_wsoj[w, station, parse(Int,task), j] == 0)
                end
            end
        end
    end
    #PREPROCESSING 2: Pairs of tasks that take up a large amount of time cannot both be started too late or too early
    #calculates the infeasible task pairs
    infeasible_pairs_forward, infeasible_pairs_backwards = get_infeasible_assignment_pairs(instance; productivity_per_worker = Dict(1=>1., 2=>1., 3=>1., 4=>1.)) 
    #prohibits the infeasible task pairs
    for (w,scenario) in enumerate(eachrow(instance.sequences.sequences))
        for j in 1:instance.sequences.sequence_length
            model = scenario.sequence[j]
            pairs = infeasible_pairs_forward[model]
            for ((task1, task2), stations) in pairs
                for station in stations
                    @constraint(m, x_wsoj[w, station, parse(Int,task1), j] + x_wsoj[w, station, parse(Int,task2), j] <= 1)
                end
            end
            pairs = infeasible_pairs_backwards[model]
            for ((task1, task2), stations) in pairs
                for station in stations
                    @constraint(m, x_wsoj[w, station, parse(Int,task1), j] + x_wsoj[w, station, parse(Int,task2), j] <= 1)
                end
            end
        end
    end
end


function define_dynamic_linear!(m::Model, instance::MALBP_W_instance; preprocessing = true)
    define_dynamic_linear_vars!(m, instance)
    define_dynamic_linear_obj!(m, instance)
    define_dynamic_linear_constraints!(m, instance)
    if preprocessing   
        @info "adding in redundant constraints"
        define_dynamic_linear_redundant_constraints!(m, instance)
    end

end

function define_dynamic_linear!(m::Model, instance::MALBP_W_instance, warmstart_vars_fp::String; preprocessing = true)
    define_dynamic_linear_vars!(m, instance)
    warmstart_dynamic_from_md_setup!(m, warmstart_vars_fp, instance)
    define_dynamic_linear_obj!(m, instance)
    define_dynamic_linear_constraints!(m, instance)   
    if preprocessing   
        @info "adding in preprocessing vars"
        define_dynamic_linear_redundant_constraints!(m, instance)
    end
end

function define_dynamic_linear_oos!(m::Model, instance::MALBP_W_instance, warmstart_vars_fp::String; preprocessing = true)
    define_dynamic_linear_vars!(m, instance)
    warmstart_dynamic_from_dynamic_and_fix!(m, warmstart_vars_fp)
    define_dynamic_linear_obj!(m, instance)
    define_dynamic_linear_constraints!(m, instance)   
    if preprocessing   
        @info "adding in preprocessing vars"
        define_dynamic_linear_redundant_constraints!(m, instance)
    end
end
