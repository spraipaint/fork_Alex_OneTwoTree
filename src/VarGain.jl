using StatsBase

# Spliting criterion for regression based on the variance similar to infogain
function var_gain(parent_labels::AbstractVector, child_1_labels::AbstractVector, child_2_labels::AbstractVector) :: Float64
    if isempty(child_1_labels) || isempty(child_2_labels)
        return 0.0
    end

    total = length(parent_labels) 

    child_1_weight = length(child_1_labels) / total
    child_2_weight = length(child_2_labels) / total

    weighted_var_1 = child_1_weight * variance(child_1_labels) 
    weighted_var_2 = child_2_weight * variance(child_2_labels) 
    weighted_var = weighted_var_1 + weighted_var_2

    return variance(parent_labels) - weighted_var
end