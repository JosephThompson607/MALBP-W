using Random
using StatsBase
using QuasiMonteCarlo, Distributions

function generate_scenario_tree(sequence_length::Int, model_mixtures::Dict{String, Float64} )
    #If the model mixtures do not sum to 1, throw an error
    if round(sum(values(model_mixtures)),digits = 6) != 1
        error("model mixtures do not sum to 1")
    #if the sequence length is less than 1, throw an error
    elseif sequence_length < 1
        error("sequence length must be at least 1")
    #If the model mixtures are not positive, throw an error
    elseif any(values(model_mixtures) .< 0)
        error("model mixtures must be positive")
    end
    function generate_sequences!(final_sequences, model_mixtures, sequence_length, current_sequence, seq_probability=1)
        if length(current_sequence) == sequence_length
            current_sequence = Dict("sequence" => current_sequence, "probability" => seq_probability)
            push!(final_sequences, current_sequence)
        else
            for (model, probability) in model_mixtures
                new_sequence = copy(current_sequence)
                push!(new_sequence, model)
                generate_sequences!(final_sequences, model_mixtures, sequence_length, new_sequence, seq_probability*probability)
            end
        end
    end
    #turns the final_sequences into a DataFrame
    final_sequences = []
    generate_sequences!(final_sequences, model_mixtures, sequence_length, [])
    #turns the final sequences into a DataFrame
    final_sequences = DataFrame(final_sequences)
    return final_sequences
end

#generates a monte carlo sampled scenario tree
function monte_carlo_tree_limit(sequence_length::Int, model_mixtures::Dict{String, Float64}, n_samples::Int; rng=Xoshiro())
    #If the model mixtures do not sum to 1, throw an error
    if round(sum(values(model_mixtures)),digits= 6) != 1
        error("model mixtures do not sum to 1. They sum to: $(sum(values(model_mixtures)))")
    #if the sequence length is less than 1, throw an error
    elseif sequence_length < 1
        error("sequence length must be at least 1, the sequence length is: $(sequence_length)")
    #If the model mixtures are not positive, throw an error
    elseif any(values(model_mixtures) .< 0)
        error("model mixtures must be positive")
    elseif (sequence_length<12) && (n_samples > length(model_mixtures)^sequence_length)
        @info "Number of samples is greater than the number of possible sequences, returning all possible sequences"
        return generate_scenario_tree(sequence_length, model_mixtures)
    end
    final_sequences = []
    for i in 1:n_samples
        current_sequence = []
        current_probability = 1
       for j in 1:sequence_length
            #samples from the model mixtures
            model = sample(rng,collect(keys(model_mixtures)), Weights(collect(values(model_mixtures))))
            push!(current_sequence, model)
        end
        push!(final_sequences, Dict("sequence" => current_sequence, "probability" => 1/n_samples))
        end
    final_sequences = DataFrame(final_sequences)
    return final_sequences
end


#generates a uniformly distributed tree
function uniform_limit(sequence_length::Int, model_mixtures::Dict{String, Float64}, n_samples::Int; rng=Xoshiro())
    Random.seed!(seed)
    #If the model mixtures do not sum to 1, throw an error
    if round(sum(values(model_mixtures)),digits= 6) != 1
        error("model mixtures do not sum to 1")
    #if the sequence length is less than 1, throw an error
    elseif sequence_length < 1
        error("sequence length must be at least 1")
    #If the model mixtures are not positive, throw an error
    elseif any(values(model_mixtures) .< 0)
        error("model mixtures must be positive")
    elseif n_samples > length(model_mixtures)^sequence_length
        @info "Number of samples is greater than the number of possible sequences, returning all possible sequences"
        return generate_scenario_tree(sequence_length, model_mixtures)
    end
    #uniformly generates n points between zero and 1
    points = [1/(n_samples*sequence_length) * i for i in 1:(n_samples*sequence_length)]
    #shuffles the points
    points = shuffle(points)
    final_sequences = []
    #calculates the range of the model mixtures for instance if prob A = 0.5, prob B = 0.3, prob C = 0.2, then the ranges are [0, 0.5, 0.8, 1]
    model_ranges = cumsum(values(model_mixtures))
    #makes sure the last element of the model ranges is 1 to avoid floating point errors
    model_ranges[end] = 1.0
    model_keys = collect(keys(model_mixtures))
  
    for i in 1:n_samples
        current_sequence = []
       for j in 1:sequence_length
            #finds the model that the point falls into
            model_index = findfirst(x -> x >= points[(i-1)*sequence_length + j], model_ranges)
            model = model_keys[model_index]
            push!(current_sequence, model)
        end
        push!(final_sequences, Dict("sequence" => current_sequence, "probability" => 1/n_samples))
        end
    final_sequences = DataFrame(final_sequences)
    return final_sequences
end


#generates a uniformly distributed tree
function sobold_limit(sequence_length::Int, model_mixtures::Dict{String, Float64}, n_samples::Int; rng=Xoshiro())
    Random.seed!(seed)
    #If the model mixtures do not sum to 1, throw an error
    if round(sum(values(model_mixtures)),digits= 6) != 1
        error("model mixtures do not sum to 1")
    #if the sequence length is less than 1, throw an error
    elseif sequence_length < 1
        error("sequence length must be at least 1")
    #If the model mixtures are not positive, throw an error
    elseif any(values(model_mixtures) .< 0)
        error("model mixtures must be positive")
    elseif n_samples > length(model_mixtures)^sequence_length
        @info "Number of samples is greater than the number of possible sequences, returning all possible sequences"
        return generate_scenario_tree(sequence_length, model_mixtures)
    end
    #uniformly generates n points between zero and 1
 
    final_sequences = []
    #calculates the range of the model mixtures for instance if prob A = 0.5, prob B = 0.3, prob C = 0.2, then the ranges are [0, 0.5, 0.8, 1]
    model_ranges = cumsum(values(model_mixtures))
    #makes sure the last element of the model ranges is 1 to avoid floating point errors
    model_ranges[end] = 1.0
    model_keys = collect(keys(model_mixtures))
  
    for i in 1:n_samples
        current_sequence = []
       for j in 1:sequence_length
            points = (QuasiMonteCarlo.sample(rng,sequence_length, [0], [1], Sobolsample(rng,)).+ rand()).% 1
            #finds the model that the point falls into
            model_index = findfirst(x -> x >= points[j], model_ranges)
            model = model_keys[model_index]
            push!(current_sequence, model)
        end
        push!(final_sequences, Dict("sequence" => current_sequence, "probability" => 1/n_samples))
        end
    final_sequences = DataFrame(final_sequences)
    return final_sequences
end


#reads the scenario tree from a csv file
function read_scenario_csv(file_name::String)
    scenarios = CSV.read(file_name, DataFrame)
    #For each row in the scenario tree, the sequence is a vector of the sequence of models
    #Makes a dataframe to add rows to
    new_scenarios = []
    len = 0
    for row in eachrow(scenarios)
        #Removes everything outside of brackets in the sequence column
        new_seq = split(row.sequence, "]")[1]
        new_seq = split(new_seq, "[")[2]
        new_seq = split(string(new_seq), ",")
        new_seq = [replace(strip(x), "\"" => "") for x in new_seq]
        push!(new_scenarios, (sequence=new_seq, probability=row.probability))
        len = length(new_seq)

    end
    new_scenarios = DataFrame(new_scenarios)
    sequences = ProdSequences(new_scenarios, nrow(new_scenarios), "read_csv", len)
    return sequences
end


function generate_new_sequences!(instance; rng=Xoshiro(), generator::String="monte_carlo_limit", n_new_samples::Int64=40)
    n_samples = instance.sequences.n_scenarios
    delete!(instance.sequences.sequences, 1:n_samples)
    instance.sequences.n_scenarios = 0
    add_more_samples!(instance, n_new_samples, rng = rng, generator=generator)
end

function add_more_samples!(instance, n_samples::Int64;rng=Xoshiro(), generator::String="monte_carlo_limit", )
    model_mixtures = get_model_mixture(instance.models)
    if generator == "sobold_limit"
        @info "adding $n_samples scenarios with sobold_limit"
        new_scenarios_df = sobold_limit(instance.sequences.sequence_length, model_mixtures, n_samples, rng=rng)
    elseif generator == "monte_carlo_limit"
        @info "adding $n_samples scenarios with monte_carlo_limit"
        new_scenarios_df = monte_carlo_tree_limit(instance.sequences.sequence_length, model_mixtures, n_samples, rng=rng)
    else
        error("Cannot find generator $generator")
    end
    instance.sequences.n_scenarios += n_samples
    combined = vcat( instance.sequences.sequences, new_scenarios_df,)
    combined[!,"probability"] .= 1/instance.sequences.n_scenarios
    instance.sequences.sequences= combined
    
    
end

#reads scenario tree file from csv accepts a dictionary of scenario tree info and model mixtures(optional)
function read_scenario_tree(scenario_info::Dict, model_mixtures::Dict{String, Float64} ; rng=Xoshiro())
    if scenario_info["generator"] == "read_csv"
        return read_scenario_csv(scenario_info["file_name"])
    elseif scenario_info["generator"] == "full"
        scenario_df= generate_scenario_tree(scenario_info["sequence_length"], model_mixtures)
    elseif scenario_info["generator"] == "monte_carlo_tree_limit" || scenario_info["generator"] == "monte_carlo_limit"
        if !haskey(scenario_info, "n_samples")
            error("n_samples is required for monte_carlo_tree_limit, please provide it in the config file")
        end
        
        scenario_df = monte_carlo_tree_limit(scenario_info["sequence_length"], model_mixtures, scenario_info["n_samples"], rng=rng)
    elseif scenario_info["generator"] == "sobold_limit"
        if !haskey(scenario_info, "n_samples")
            error("n_samples is required for sobold_limit, please provide it in the config file")
        end
        if !haskey(scenario_info, "seed")
            scenario_info["seed"] = nothing
        end
        scenario_df = sobold_limit(scenario_info["sequence_length"], model_mixtures, scenario_info["n_samples"], rng=rng)
    elseif scenario_info["generator"] == "uniform_limit"
        if !haskey(scenario_info, "n_samples")
            error("n_samples is required for uniform_limit, please provide it in the config file")
        end
        if !haskey(scenario_info, "seed")
            scenario_info["seed"] = nothing
        end
        scenario_df = uniform_limit(scenario_info["sequence_length"], model_mixtures, scenario_info["n_samples"], rng=rng)
    else
        error("unrecognized generator, currently we only support read_csv, monte_carlo_tree_limit, uniform_limit, and full for scenario tree generation.")
    end
    n_scenarios = nrow(scenario_df)
    scenario = ProdSequences(
        scenario_df,
        n_scenarios,
        scenario_info["generator"],
        scenario_info["sequence_length"]
    )
    return scenario
end

