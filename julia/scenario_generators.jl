

function generate_scenario_tree(sequence_length::Int, model_mixtures::Dict{String, Float64})
    #If the model mixtures do not sum to 1, throw an error
    if sum(values(model_mixtures)) != 1
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

#reads scenario tree file from csv accepts a dictionary of scenario tree info and model mixtures(optional)
function read_scenario_tree(scenario_info::Dict, model_mixtures::Dict{String, Float64} )
    if scenario_info["generator"] == "read_csv"
        return CSV.read(scenario_info["filepath"], DataFrame)
    elseif scenario_info["generator"] == "full"
        return generate_scenario_tree(scenario_info["sequence_length"], model_mixtures)
    else
        error("unrecognized generator, currently we only support read_csv for scenario tree generation.")
    end
end

