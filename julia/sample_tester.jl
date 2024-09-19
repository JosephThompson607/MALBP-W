
function md_sample_test(config_filepath::String, base_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1, md_heuristic::Union{Function, Nothing}=task_equip_heuristic, scenario_generator::String = "monte_carlo_limit")
    instances = read_MALBP_W_instances(config_filepath, scenario_generator=scenario_generator)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-ddTHH:MM")
    
    for instance in instances
        output_filepath = xp_folder * "/" * now * "_" * base_filepath * "md/"* instance.name * "/"
        if !isdir(output_filepath)
            mkpath(output_filepath)
        end
        md_iterative_sampling(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, md_heuristic=md_heuristic, scenario_generator=scenario_generator)
    end
end


function slurm_md_sample_test(config_filepath::String, output_filepath::String, run_time::Float64, save_variables::Bool, save_lp::Bool,slurm_array_ind::Int; xp_folder::String="model_runs", preprocessing::Bool=false, rng=Xoshiro(), grb_threads=1,  md_heuristic::Union{Function, Nothing}=task_equip_heuristic, scenario_generator::String = "monte_carlo_limit")
    _, instance = read_slurm_csv(config_filepath, slurm_array_ind, scenario_generator=scenario_generator)
    optimizer = optimizer_with_attributes(() -> Gurobi.Optimizer(GRB_ENV_REF[]), "TimeLimit" => run_time, "Threads" => grb_threads)    #adds the date and time to the output file path
    now = Dates.now()
    now = Dates.format(now, "yyyy-mm-dd")
    output_filepath = xp_folder * "/" * now * "_" * output_filepath *  "md/"* instance.name * "/slurm_" * string(slurm_array_ind) * "/"
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    md_iterative_sampling(instance, optimizer, output_filepath, run_time; save_variables= save_variables, save_lp=save_lp, preprocessing=preprocessing, slurm_array_ind=slurm_array_ind, md_heuristic=md_heuristic, scenario_generator=scenario_generator)

end

function md_iterative_sampling(instance::MALBP_W_instance, optimizer, original_filepath::String, run_time::Real; scenario_generator::String, preprocessing::Bool=false, save_variables::Bool=true, save_lp::Bool=false, slurm_array_ind::Union{Int, Nothing}=nothing,md_heuristic=task_equip_heuristic, max_iterations=20,  )
    #if directory is not made yet, make it
    iter = 0

    output_filepath = original_filepath * "/samples_$(instance.sequences.n_scenarios)/"
    if !isdir(output_filepath)
        mkpath(output_filepath)
    end
    #creates the model
    m = Model(optimizer)
    set_optimizer_attribute(m, "LogFile", output_filepath * "gurobi.log")
    #defines the model dependent parameters
    define_md_linear!(m, instance; preprocess=preprocessing, start_heuristic=md_heuristic)
    #writes the model to a file
    optimize!(m)

    if save_lp
        @info "Writing LP to file $(output_filepath * "model.lp")"
        write_to_file(m, output_filepath * "model.lp")
    end
    if save_variables
        write_MALBP_W_solution_md(output_filepath, instance, m, false)
    end
    save_results(original_filepath , m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
    while iter < max_iterations
        add_more_samples!(instance, 1; generator=scenario_generator)
        output_filepath = original_filepath * "/samples_$(instance.sequences.n_scenarios)/"
        if !isdir(output_filepath)
            mkpath(output_filepath)
        end
        new_m =  Model(optimizer)
        set_optimizer_attribute(new_m, "LogFile", output_filepath * "gurobi.log")
        #defines the model dependent parameters
        define_md_linear!(new_m, instance; preprocess=false, start_heuristic=nothing)
        sample_warmstart!(m, new_m)
        m =nothing #freeing up memory
        optimize!(new_m)
        if save_variables
            write_MALBP_W_solution_md(output_filepath, instance, new_m, false)
        end
        save_results(original_filepath , new_m, run_time, instance, output_filepath, "model_dependent_problem_linear_labor_recourse.csv")
        m = new_m
        iter +=1
    end
    return m
end