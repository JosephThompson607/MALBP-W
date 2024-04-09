using CSV
using DataFrames
using Dates

function write_x_soi_solution(output_filepath::String, instance::MALBP_W_instance, x::Array, only_nonzero::Bool=false)
    #writes the solution to a file
    x_soi_solution = []
    for s in 1:instance.equipment.no_stations
        for o in 1:instance.equipment.no_tasks
            for i in 1:instance.models.no_models
                if only_nonzero && value(x[s, o, i]) == 0
                    continue
                end
                x_soi_dict = Dict("station"=>s, "task"=>o, "model"=>i, "value"=>value(x[s, o, i]))
                push!(x_soi_solution, x_soi_dict)
            end
        end
    end
    #writes the x_soi_solution as a csv
    x_soi_solution_df = DataFrame(x_soi_solution)
    CSV.write(output_filepath * "x_soi_solution.csv", x_soi_solution_df)
end

function write_x_wsoj_solution(output_filepath::String, instance::MALBP_W_instance, x::Array, only_nonzero::Bool=false)
    #writes the solution to a file
    x_wsoj_solution = []
    for w in 1:instance.no_scenarios
        for s in 1:instance.equipment.no_stations
            for o in 1:instance.equipment.no_tasks
                for j in 1:instance.sequence_length
                    if only_nonzero && value(x[w, s, o, j]) == 0
                        continue
                    end
                    
                    x_wsoj_dict = Dict("scenario"=>w, "station"=>s, "task"=>o, "item_idx"=>j, "item"=>instance.scenarios[w,"sequence"][j], "value"=>value(x[w, s, o, j]))
                    push!(x_wsoj_solution, x_wsoj_dict)
                end
            end
        end
    end
    #writes the x_wsoj_solution as a csv
    x_wsoj_solution_df = DataFrame(x_wsoj_solution)
    CSV.write(output_filepath * "x_wsoj_solution.csv", x_wsoj_solution_df)
end

function write_u_se_solution(output_filepath::String, instance::MALBP_W_instance, u::Array, only_nonzero::Bool=false)
    #writes the solution to a file
    u_se_solution = []
    for s in 1:instance.equipment.no_stations
        for e in 1:instance.equipment.no_equipment
            if only_nonzero && value(u[s, e]) == 0
                continue
            end
            u_se_dict = Dict("station"=>s, "equipment"=>e, "value"=>value(u[s, e]))
            push!(u_se_solution, u_se_dict)
        end
    end
    #writes the u_se_solution as a csv
    u_se_solution_df = DataFrame(u_se_solution)
    CSV.write(output_filepath * "u_se_solution.csv", u_se_solution_df)
end

function write_y_wts_solution(output_filepath::String, instance::MALBP_W_instance, y_wts, only_nonzero::Bool=false)
    #writes the solution to a file
    y_wts_solution = []
    for w in 1:instance.no_scenarios
        for t in 1:instance.no_cycles
            for s in 1:instance.equipment.no_stations
                if only_nonzero && value(y_wts[w, t, s]) == 0
                    continue
                end
                y_wts_dict = Dict("scenario"=>w, "cycle"=>t, "station"=>s, "value"=>value(y_wts[w, t, s]))
                push!(y_wts_solution, y_wts_dict)
            end
        end
    end
    #writes the y_wts_solution as a csv
    y_wts_solution_df = DataFrame(y_wts_solution)
    CSV.write(output_filepath * "y_wts_solution.csv", y_wts_solution_df)
end

function write_y_w_solution(output_filepath::String, instance::MALBP_W_instance, y_w, y; only_nonzero::Bool=false)
    #writes the solution to a file
    y_solution = []
    for w in 1:instance.no_scenarios
        if only_nonzero && value(y_w[w]) == 0
            continue
        end
        y_dict = Dict("scenario"=>w, "value"=>value(y_w[w]))
        push!(y_solution, y_dict)
    end
    y_dict = Dict("scenario"=>"fixed", "value"=>value(y))
    push!(y_solution, y_dict)
    #writes the y_solution as a csv
    y_solution_df = DataFrame(y_solution)
    CSV.write(output_filepath * "y_solution.csv", y_solution_df)
end

function write_MALBP_W_solution_md(output_filepath::String, instance::MALBP_W_instance, m::Model, only_nonzero::Bool=false)
    #If the output_filepath does not exist, make in
    if !isdir(output_filepath)
        mkdir(output_filepath)
    end
    x = m[:x_soi]
    u = m[:u_se]
    y_w = m[:y_w]
    y = m[:y]
    if is_solved_and_feasible(m) || termination_status(m) == MOI.TIME_LIMIT
        write_x_soi_solution(output_filepath, instance, x, only_nonzero)
        write_u_se_solution(output_filepath, instance, u, only_nonzero)
        write_y_w_solution(output_filepath, instance, y_w, y; only_nonzero = only_nonzero)
        write_y_wts_solution(output_filepath, instance, m[:y_wts], only_nonzero)
    else
        @info("Model is not solved or feasible, no solution written")
    end
    #writes the sequences to a file
    CSV.write(output_filepath * "sequences.csv", instance.scenarios)
end

function write_MALBP_W_solution_dynamic(output_filepath::String, instance::MALBP_W_instance, m::Model, only_nonzero::Bool=false)
    #If the output_filepath does not exist, make in
    if !isdir(output_filepath)
        mkdir(output_filepath)
    end
    x = m[:x_wsoj]
    u = m[:u_se]
    y_w = m[:y_w]
    y = m[:y]

    if is_solved_and_feasible(m) || termination_status(m) == MOI.TIME_LIMIT
        write_x_wsoj_solution(output_filepath, instance, x, only_nonzero)
        write_u_se_solution(output_filepath, instance, u, only_nonzero)
        write_y_w_solution(output_filepath, instance, y_w, y; only_nonzero = only_nonzero)
        write_y_wts_solution(output_filepath, instance, m[:y_wts], only_nonzero)
    else
        @info("Model is not solved or feasible, no solution written")
    end
    #writes the sequences to a file
    CSV.write(output_filepath * "sequences.csv", instance.scenarios)

    #writes the sequences to a yaml file
    
    
end


function save_results(output_filepath::String, m::Model, run_time::Real, instance::MALBP_W_instance, var_fp::String, output_csv::String; prev_obj_val::Union{Real, Nothing}=nothing)
    #saves the objective function, relative gap, run time, and instance_name to a file
    if is_solved_and_feasible(m)  || termination_status(m) == MOI.TIME_LIMIT
        obj_val = objective_value(m)
        rel_gap = relative_gap(m)
        solution_time = solve_time(m)
    else
        obj_val = "NA"
        rel_gap = "NA"
        solution_time = "NA"
    end
    if !isnothing(prev_obj_val)
    results = DataFrame(instance_name=instance.name, 
                        prev_obj_val=prev_obj_val,
                        objective_value=obj_val, 
                        relative_gap=rel_gap, 
                        solve_time=solution_time,
                        run_time=run_time, 
                        date=Dates.now(),
                        equip_fp= instance.equipment.filepath,
                        model_fp= instance.models.filepath,
                        instance_fp= instance.filepath, 
                        output_folder=var_fp)
    else
        results = DataFrame(instance_name=instance.name, 
                        objective_value=obj_val, 
                        relative_gap=rel_gap, 
                        solve_time=solution_time,
                        run_time=run_time, 
                        date=Dates.now(),
                        equip_fp= instance.equipment.filepath,
                        model_fp= instance.models.filepath,
                        instance_fp= instance.filepath, 
                        output_folder=var_fp)
    end
    #If the file does not exist, create it
    if !isfile(output_filepath * output_csv)
        CSV.write(output_filepath * output_csv, results)
    else
        CSV.write(output_filepath * output_csv, results, append=true)
    end
end