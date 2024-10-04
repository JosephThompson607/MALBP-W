include("constructive.jl")
using(Test)

config_filepath = "SALBP_benchmark/MM_instances/testing_yaml/julia_debug.yaml"
instance = read_MALBP_W_instances(config_filepath)[2]
task_assignments = ehsans_heuristic(instance)
#Tests to make sure there are two model dictionaries in task assignments
@test length(task_assignments) == 2
#Tests to make sure there are 4 stations
@test length(task_assignments["A"]) == 4
#Sums the length of each station's task list
total_tasks = sum([length(task_assignments["A"][station]) for station in keys(task_assignments["A"])])
@test total_tasks == 20

y_wts = base_worker_assign_func(instance, task_assignments)
@test size(y_wts) == (2^5, instance.n_cycles, instance.equipment.n_stations)

instance = read_MALBP_W_instances(config_filepath)[1]
model = instance.models.models["A"]
new_cycle_time = calculate_new_cycle_time(model, instance.equipment.n_stations, instance.models.cycle_time)
@test new_cycle_time == 208.75


prec_matrix = create_precedence_matrix(instance; order_function= positional_weight_order)
@test size(prec_matrix["A"]["precedence_matrix"]) == (5, 4)
@test prec_matrix["A"]["index_to_task"][1] == "6"
@test prec_matrix["A"]["precedence_matrix"][end, :] == [0, 1, 2, 2]

equipment_assignments = greedy_equipment_assignment_heuristic(instance, task_assignments)


r_oe = transpose(stack(instance.equipment.r_oe))
@test size(r_oe) == (20, 4)
for (model, station_assignments) in task_assignments
    for (station, tasks) in station_assignments
        for task in tasks
            task_covered = false
            for equip in equipment_assignments[station]
                if instance.equipment.r_oe[parse(Int,task)][equip] == 1
                    task_covered = true
                end
            end
            @test task_covered == true
        end
        
    end
end
