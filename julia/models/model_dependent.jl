

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
        @constraint(m, y +  y_w[w] == sum(y_wts[w, t, s] for s in 1:instance.equipment.no_stations))
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
end


function define_md_linear!(m::Model, instance::MALBP_W_instance)
    define_md_linear_vars!(m, instance)
    define_md_linear_obj!(m, instance)
    define_md_linear_constraints!(m, instance)   
end


