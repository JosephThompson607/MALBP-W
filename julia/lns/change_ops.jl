import Random

function no_change!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    return iter_no_improve, lns_obj, m
end

function decrement_y!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        y = m[:y]
        fix.(y, start_value(y)-1, force=true)

    elseif iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] in [1:lns_obj.change.kwargs[:fix_steps];];
        y = m[:y]
        fix.(y, start_value(y), force=true)
    end
    return iter_no_improve, lns_obj, m
end

function increase_size!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if lns_obj.des.kwargs[:percent_destroy] >= 1.0
        lns_obj.des.kwargs[:percent_destroy] = lns_obj.des.old_kwargs[:percent_destroy]
    end 
    if iter_no_improve == 0
        lns_obj.des.kwargs[:percent_destroy] = lns_obj.des.old_kwargs[:percent_destroy]
    elseif iter_no_improve > 0 &&  iter_no_improve % lns_obj.change.kwargs[:size_period] == 0 
        #increase the size of the destroy block only so much as to have an impact on one of the operators
        lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
    end
    return iter_no_improve, lns_obj, m
end

function increase_destroy_old!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve > 0 &&  iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0 
        #increase the size of the destroy block only so much as to have an impact on one of the operators
        lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
    end
    return iter_no_improve, lns_obj, m
end

function change_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model;  
                            filter_out_current = false, rng= Xoshiro() , _...)
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        #randomly chooses from the destroy operators
        select_destroy!(lns_obj; filter_out_current=filter_out_current, rng = rng)
    end
    return iter_no_improve, lns_obj, m
end

#selects new destroy randomly from destroy operators using their weight
function select_destroy!(lns_obj::LNSConf; filter_out_current=false, rng = Xoshiro())
    operator_list = lns_obj.des.destroy_list
    destroy_weights = copy(lns_obj.des.destroy_weights)
    #filters out the current operator
    if filter_out_current
        operator_list = filter(x -> x != lns_obj.des.destroy!, operator_list)
        delete!(destroy_weights, string(lns_obj.des.destroy!))
    end
    weights = collect(values(destroy_weights))
    destroy_names = collect(keys(destroy_weights))
    destroy_choice = sample(rng, destroy_names, Weights(weights))
    destroy = operator_list[findfirst(x -> string(x) == destroy_choice, operator_list)]
    update_destroy_operator!(lns_obj.des, destroy)
end

#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
function increase_size_change_destroy!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=false, rng = Xoshiro(), _...)
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        if lns_obj.des.kwargs[:n_destroy] < lns_obj.des.kwargs[:destroy_limit] 
            lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
        else
            #resets the size of block to destroy
            lns_obj.des.kwargs[:percent_destroy] = lns_obj.des.old_kwargs[:percent_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current, rng=rng)
        end
    end
    return iter_no_improve, lns_obj, m
end

#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
function change_destroy_increase_size!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=false, rng= Xoshiro(), _...)
    #if size of destruction is already >1, then reset the destruction size to old value
    if lns_obj.des.kwargs[:percent_destroy] >= 1.0
        lns_obj.des.kwargs[:percent_destroy] = lns_obj.des.old_kwargs[:percent_destroy]
    end 
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0 
        #calculates number of periods
        n_periods = ceil(iter_no_improve / lns_obj.change.kwargs[:size_period])
        if iter_no_improve < lns_obj.change.kwargs[:size_period] * n_periods
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current,rng=rng)         
        else
            lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current, rng=rng)
        end
    end
    return iter_no_improve, lns_obj, m
end


#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
#If there is an improvement, resets the size of the destroy block
function change_destroy_increase_size_reset_improve!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=true, rng=Xoshiro(), _...)
    #if size of destruction is already >1, then reset the destruction size to old value
    if lns_obj.des.kwargs[:percent_destroy] >= 1.0 || iter_no_improve == 0
        lns_obj.des.kwargs[:percent_destroy] = lns_obj.des.old_kwargs[:percent_destroy]
    end 
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0 
        #calculates number of periods
        n_periods = ceil(iter_no_improve / lns_obj.change.kwargs[:size_period])
        if iter_no_improve < lns_obj.change.kwargs[:size_period] * n_periods
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current, rng=rng)         
        else
            lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current,rng=rng)
        end
    end
    return iter_no_improve, lns_obj, m
end


#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator
#If there is an improvement, makes it destroy less
function change_destroy_increase_size_reduce_improve!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=true, rng=Xoshiro(), _...)
    #if size of destruction is already >1, then reset the destruction size to old value
    if lns_obj.des.kwargs[:percent_destroy] >= 1.0 || iter_no_improve == 0
        lns_obj.des.kwargs[:percent_destroy] = max(lns_obj.des.kwargs[:percent_destroy]-lns_obj.des.kwargs[:min_destroy], lns_obj.des.kwargs[:min_destroy])
    end 
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0 
        #calculates number of periods
        n_periods = ceil(iter_no_improve / lns_obj.change.kwargs[:size_period])
        if iter_no_improve < lns_obj.change.kwargs[:size_period] * n_periods
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current, rng=rng)         
        else
            lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.kwargs[:min_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current,rng=rng)
        end
    end
    return iter_no_improve, lns_obj, m
end
#increases the size of destroy block if no improvement until it reaches a limit, then changes the destroy operator, does not reset after 1
function change_destroy_increase_size_old!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; filter_out_current=true,  _...)
    #if size of destruction is already >1, then reset the destruction size to old value
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0 
        #calculates number of periods
        n_periods = ceil(iter_no_improve / lns_obj.change.kwargs[:size_period])
        if iter_no_improve < lns_obj.change.kwargs[:size_period] * n_periods
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current)         
        else
            lns_obj.des.kwargs[:percent_destroy] += lns_obj.des.old_kwargs[:percent_destroy]
            #randomly chooses from the destroy operators
            select_destroy!(lns_obj; filter_out_current=filter_out_current)
        end
    end
    return iter_no_improve, lns_obj, m
end

#increases the runtime of the repair operator by 1 minute
function increase_repair_time!(iter_no_improve::Int, lns_obj::LNSConf, m::Model; _...)
    if iter_no_improve > 0 && iter_no_improve % lns_obj.change.kwargs[:change_freq] == 0
        lns_obj.rep.kwargs[:time_limit] += 60
        set_optimizer_attribute(m, "TimeLimit", lns_obj.rep.kwargs[:time_limit])
    end
    return iter_no_improve, lns_obj, m
end