#This file contains the adaptive strategies for the LNS algorithm

#Does not adapt the LNS algorithm, useful for testing
function no_adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    return iter_no_improve, lns_obj, m
end



#adaptive lns for destroy operator selection
function adapt_lns_des!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64, des_weight_update::Function = no_weight_update, obj_val_delta::Float64=0., _...)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    #rewards the destroy and change operator if there has been an improvement
    if iter_no_improve == 0
        lns_obj.des.destroy_weights[lns_obj.des.name] = des_weight_update(iter_no_improve,
        lns_obj.des.destroy_weights[lns_obj.des.name]; 
                                                                            obj_val_delta = obj_val_delta,
                                                                            iteration = iteration, 
                                                                            iteration_time = iteration_time, 
                                                                            decay = decay)
    end
    return iter_no_improve, lns_obj, m
end

#Weighting functions for adaptive LNS
#does not change weights
function no_weight_update(iter_no_improve::Int, weights::Float64; _...)
    return weights
end

#weights the update by the number of iterations and compute time
function iter_and_time_update(iter_no_improve::Int, weights::Float64; iteration::Int, iteration_time::Float64, decay::Float64, _...)
    if iter_no_improve == 0
        new_weights = decay *  weights + (1-decay) * iteration / (1 + (iteration_time/10))
        return new_weights
    else
        return weights * decay
    end
end

#basic update operator
function basic_update(iter_no_improve::Int, weights::Float64; decay::Float64, up_amount::Float64= 1., _...)
    if iter_no_improve == 0
        return decay * weights + (1-decay) * up_amount
    else
        return weights * decay
    end
end

#weight that rewards objective value improvement at later iterations and penalizes run time
function obj_val_update(iter_no_improve::Int, weights::Float64; current_best::Float64, prev_best::Float64, iteration::Int, iteration_time::Float64, decay::Float64, _...)
    if iter_no_improve == 0
        new_weights =  decay* weights + (1-decay) * iteration * (prev_best/current_best)/ (1 + (iteration_time/10))
        return new_weights
    else
        return weights * decay
    end
end


#adaptive lns for destroy and change operator selection
function adapt_lns!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; iteration::Int, iteration_time::Float64, current_best::Float64,prev_best::Float64, init_period::Int=1, des_weight_update::Function = no_weight_update, change_weight_update::Function = no_weight_update, obj_val_delta::Float64=0.0,  _...)
    #retrieves the decay and change decay parameters
    decay = lns_obj.des.kwargs[:des_decay]
    change_decay = lns_obj.change.kwargs[:change_decay]
    #rewards the destroy and change operator if there has been an improvement
    println("WEIGHTS BEFORE", lns_obj.des.destroy_weights)
    println("WEIGHTS BEFORE", lns_obj.change.change_weights)
    lns_obj.des.destroy_weights[lns_obj.des.name] = des_weight_update(iter_no_improve, 
                                                                            lns_obj.des.destroy_weights[lns_obj.des.name];
                                                                            obj_val_delta = obj_val_delta, 
                                                                            iteration = iteration, 
                                                                            iteration_time = iteration_time, 
                                                                            decay = decay, 
                                                                            current_best =  current_best,
                                                                            prev_best = prev_best)
    lns_obj.change.change_weights[string(lns_obj.change.change!)] = change_weight_update(iter_no_improve,
                                                                                            lns_obj.change.change_weights[string(lns_obj.change.change!)], 
                                                                                               obj_val_delta = obj_val_delta,
                                                                                                iteration = iteration,
                                                                                               iteration_time =  iteration_time, 
                                                                                               decay = change_decay,
                                                                                               current_best = current_best,
                                                                                               prev_best = prev_best)

    println("change_weight_update: ", change_weight_update)
    println("des_weight_update: ", des_weight_update)
    println("XXXXAAAAAdestroy_weights: ", lns_obj.des.destroy_weights)
    println("XXXXAAAAAchange_weights: ", lns_obj.change.change_weights)
    #selects new change operator if we are passed the learning period
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        if iteration >init_period
        
            #randomly choses the destroy operator based on the rewards
            change_dict = lns_obj.change.change_weights
            change_names = collect(keys(change_dict))
            change_rewards = collect(values(change_dict))
            change_list = [no_change!, increase_destroy!, decrement_y!, change_destroy!, increase_repair_time!]
            change_name = sample(change_names, Weights(change_rewards))
            change! = change_list[findfirst(x -> string(x) == change_name, change_list)]
            @info "change operator changed to: $change! at iteration $iteration"
            lns_obj.change.change! = change!
        else
            operator = rand(lns_obj.des.destroy_list)
            @info "Exploration period at iteration $iteration, trying operator: $operator"
            update_destroy_operator!(lns_obj.des, operator)
        end

    end
    return iter_no_improve, lns_obj, m
end
