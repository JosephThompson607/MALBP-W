using JuMP
using Gurobi
include("../read_MALBP_W.jl") 
include("../output.jl")


#defines the decision variables for the dynamic MALBP-W model
function define_dynamic_linear_vars!(m::Model, instance::MALBP_W_instance)
    #defines the variables
    @variable(m, x_wsoj[1:instance.no_scenarios, 1:instance.equipment.no_stations, 1:instance.equipment.no_tasks, 1:instance.sequence_length], Bin, base_name="x_soj")
    @variable(m, u_se[1:instance.equipment.no_stations, 1:instance.equipment.no_equipment], Bin, base_name="u_se")
    @variable(m, y_wts[1:instance.no_scenarios, 1:instance.no_cycles, 1:instance.no_stations] >=0, Int, base_name="y_wts")
    @variable(m, y_w[1:instance.no_scenarios]>=0, Int, base_name="y_w")
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
            instance.recourse_cost * sum(y_w[w] * instance.scenarios[w, "probability"] for w in 1:instance.no_scenarios) + 
            sum(instance.equipment.c_se[s][e] * u_se[s, e] for s in 1:instance.equipment.no_stations, e in 1:instance.equipment.no_equipment)
            )
end

function add_non_anticipativity_constraints!(m::Model, instance::MALBP_W_instance, w::Int, w_prime::Int)
    #usesful variables
    x_wsoj = m[:x_wsoj]
    #constraint 1:  x_wsoj must be equal for shared sequence segments
    for t in instance.sequence_length:1
        #if the sequence is the same up to time t, then x_wsoj must be equal
        if instance.scenarios[w, "sequence"][1:t] == instance.scenarios[w_prime, "sequence"][1:t]
            for j in 1:instance.sequence_length
                @constraint(m, x_wsoj[w, s, o, j] == x_wsoj[w_prime, s, o, j])
            end
        return
        end
    end

end

#defines the constraints of the model dependent MALBP-W model
function define_dynamic_linear_constraints!(m::Model, instance::MALBP_W_instance)
    # usesful variables
    model_indexes = [i for (i, model_dict) in instance.models.models]
    #model variables
    x_wsoj = m[:x_wsoj]
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
    for w in 1:instance.no_scenarios
        for s in 1:instance.equipment.no_stations
            for o in 1:instance.equipment.no_tasks
                for j in 1:instance.sequence_length
                    #Do not need constraint if the task is not a task of the model
                    if string(o) ∉ keys(instance.models.models[instance.scenarios[w,"sequence"][j]].task_times[1])
                        continue
                    end
                    @constraint(m, sum(x_wsoj[w, s, o, j] for j in 1:instance.sequence_length) == 1)
                end
            end
        end
    end
    #constraint 3: sum of task times of each assigned task for each model must be less than the cycle time times the number of workers y_wts
    for w in eachrow(instance.scenarios)
        w_index = rownumber(w)
        for t in 1:instance.no_cycles
            for s in 1:instance.equipment.no_stations
                if 1 <= t - s +1<= instance.sequence_length 
                    j = t-s + 1
                    model = w["sequence"][j]
                    task_times = instance.models.models[model].task_times[1]
                    @constraint(m, sum(task_time * x_wsoj[w_index,s, parse(Int,o), j] for (o, task_time) in task_times) <= instance.models.cycle_time * y_wts[w_index, t, s])
                end
            end
        end
    end
    #constraint 4: tasks can only be assigned to stations that have the correct equipment
    for w in 1:instance.no_scenarios
        for s in 1:instance.equipment.no_stations
            for o in 1:instance.equipment.no_tasks
                for j in 1:instance.sequence_length
                    if string(o) ∉ keys(instance.models.models[instance.scenarios[w,"sequence"][j]].task_times[1])
                    continue
                end
                    @constraint(m, x_wsoj[w,s, o, j] <= sum(instance.equipment.r_oe[o][e] * u_se[s, e] for e in 1:instance.equipment.no_equipment))
                end
            end
        end
    end
    #constraint 5: precedence precedence_relations
    for w in 1:instance.no_scenarios
        for j in 1:instance.sequence_length
            model = instance.scenarios[w,"sequence"][j]
            for (prec, suc) in instance.models.models[model].precendence_relations
                @constraint(m, sum( s * x_wsoj[w,s, parse(Int,prec), j] for s in 1:instance.equipment.no_stations) <= sum( s * x_wsoj[w,s, parse(Int,suc), j] for s in 1:instance.equipment.no_stations))
            end
        end
    end
    #Constraint 6: non-anticipativity
    for w in 1:instance.no_scenarios
        for w_prime in w:instance.no_scenarios
            add_non_anticipativity_constraints!(m, instance, w, w_prime)
        end
    end
end


function define_dynamic_linear!(m::Model, instance::MALBP_W_instance)
    define_dynamic_linear_vars!(m, instance)
    define_dynamic_linear_obj!(m, instance)
    define_dynamic_linear_constraints!(m, instance)   
end


function MMALBP_W_dynamic(config_filepath::String, output_filepath::String="")
    #reads the instance file
    instances = read_MALBP_W_instances(config_filepath)
    #creates the model
    m = Model(Gurobi.Optimizer)
    instance = instances[1]
    #defines the model dependent parameters
    define_dynamic_linear!(m, instance)
    #writes the model to a file
    optimize!(m)
    write_MALBP_W_solution_dynamic(output_filepath, instance, m, true)
    write_to_file(m, output_filepath * "model.lp")
    return m
end

malbp_w_model = MMALBP_W_dynamic("SALBP_benchmark/MM_instances/julia_debug.yaml", "model_runs/juliaaa/dynamic/")