#This file contains the adaptive strategies for the LNS algorithm

#Does not adapt the LNS algorithm, useful for testing
function no_adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64)
    return iter_no_improve, lns_obj, m
end



#adaptive lns for destroy operator selection
function adapt_lns_des!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64, weight_update::Function = no_weight_update)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    #decays the rewards of the destroy and change operator
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] += (1-decay) * 1 * iteration / (1 + (iteration_time/10))
    end
    return iter_no_improve, lns_obj, m
end

#Weighting functions for adaptive LNS
#does not change weights
function no_weight_update(weights::Float64; _...)
    return weights
end

#weights the update by the number of iterations and compute time
function iter_and_time_update(weights::Float64, iteration::Int, iteration_time::Float64, decay::Float64; _...)
    new_weights =  weights + (1-decay) * iteration / (1 + (iteration_time/10))
    return new_weights
end

#basic update operator
function basic_update(weights::Float64, decay::Float64; up_amount::Float64= 1., _...)
    return weights + (1-decay) * up_amount
end


#adaptive lns for destroy and change operator selection
function adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64, init_period::Int=6, des_weight_update::Function = no_weight_update, change_weight_update::Function = no_weight_update)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    change_decay = lns_obj.change.kwargs[:change_decay]
    #decays the rewards of the destroy and change operator
    lns_obj.des.destroy_weights[lns_obj.des.name] *= decay 
    lns_obj.change.change_weights[string(lns_obj.change.change!)] *= change_decay
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] += des_weight_update(lns_obj.des.destroy_weights[lns_obj.des.name], 
                                                                            iteration, 
                                                                            iteration_time, 
                                                                            decay)
        lns_obj.change.change_weights[string(lns_obj.change.change!)] += change_weight_update(lns_obj.change.change_weights[string(lns_obj.change.change!)], 
                                                                                                iteration,
                                                                                                iteration_time, 
                                                                                                change_decay)
    end
    #selects new change operator if we are passed the learning period
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        if iteration >init_period
        
            #randomly choses the destroy operator based on the rewards
            change_dict = lns_obj.change.change_weights
            change_names = collect(keys(change_dict))
            change_rewards = collect(values(change_dict))
            change_list = [no_change!, increase_destroy!, decrement_y!, change_destroy!, increase_repair_time!]
            println("functions", change_list)
            change_name = sample(change_names, Weights(change_rewards))
            println("change_name: ", change_name)
            change! = change_list[findfirst(x -> string(x) == change_name, change_list)]
            @info "change operator changed to: $change! at iteration $iteration"
            lns_obj.change.change! = change!
        else
            operator = rand([random_station_destroy!, random_subtree_destroy!,random_model_destroy!])
            @info "Exploration period at iteration $iteration, trying operator: $operator"
            update_destroy_operator!(lns_obj.des, operator)
        end

    end
    return iter_no_improve, lns_obj, m
end