

function no_change!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    return iter_no_improve, lns_obj, m
end

function decrement_y!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        y = m[:y]
        fix.(y, start_value(y)-1, force=true)

    elseif iter_no_improve % lns_obj.change.kwargs[:change_freq] in [1:lns_obj.change.kwargs[:fix_steps];];
        y = m[:y]
        fix.(y, start_value(y), force=true)
    end
    return iter_no_improve, lns_obj, m
end

function increase_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        lns_obj.des.kwargs[:n_destroy] += 1
    end
    return iter_no_improve, lns_obj, m
end

function change_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model;  
                            filter_out_current = true, _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        #randomly chooses from the destroy operators
        select_destroy!(lns_obj; filter_out_current=filter_out_current)
    end
    return iter_no_improve, lns_obj, m
end

#selects new destroy randomly from destroy operators using their weight
function select_destroy!(lns_obj::LNSConf; filter_out_current=true)
    operator_list = [random_station_destroy!, random_subtree_destroy!,random_model_destroy!]
    destroy_weights = copy(lns_obj.des.destroy_weights)
    #filters out the current operator
    if filter_out_current
        operator_list = filter(x -> x != lns_obj.des.destroy!, operator_list)
        delete!(destroy_weights, lns_obj.des.name)
    end
    weights = collect(values(destroy_weights))
    destroy_names = collect(keys(destroy_weights))
    destroy_choice = sample(destroy_names, Weights(weights))
    destroy = operator_list[findfirst(x -> string(x) == destroy_choice, operator_list)]
    update_destroy_operator!(lns_obj.des, destroy)
end

#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
function change_destroy_increase_size!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=true,  _...)
    if iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        if lns_obj.des.kwargs[:n_destroy] < lns_obj.des.kwargs[:destroy_limit] 
            lns_obj.des.kwargs[:n_destroy] += 1
        else
            #resets the size of block to destroy
            lns_obj.des.kwargs[:n_destroy]= lns_obj.des.old_kwargs[:n_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current)
        end
    end
    return iter_no_improve, lns_obj, m
end