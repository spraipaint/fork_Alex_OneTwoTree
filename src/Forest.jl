using StatsBase: mode #TODO richtig einbinden oder selbst schreiben

mutable struct RandomForest <: AbstractDecisionTree
    trees::Vector{AbstractDecisionTree}
    n_trees::Int
    n_features_per_tree::Int
    max_depth::Int
end

function RandomForest(n_trees::Int, n_features_per_tree::Int, max_depth::Int)
    if n_trees <= 0
        error("A RandomForest needs more than 0 trees.\n (Currently n_trees == $n_trees)")
    end

    if n_features_per_tree <= 0
        error("A RandomForest needs more than 0 features per Tree.\n (Currently n_features_per_tree == $n_features_per_tree)")
    end

    if max_depth <= 0
        error("A RandomForest needs more than 0 max depth per Tree.\n (Currently max_depth == $max_depth)")
    end

    RandomForest(Vector{AbstractDecisionTree}(), n_trees, n_features_per_tree, max_depth)
end

#Returns random features and their labels
function get_random_features(features::Matrix{Union{Real, String}}, labels::Vector{Union{Real, String}}, n_features::Int)
    random_indices = rand(1:size(features,1), n_features)
    random_features = features[random_indices, :]
    random_labels = labels[random_indices]
    return random_features, random_labels
end

function fit!(forest::RandomForest, features::Matrix{Union{Real, String}}, labels::Vector{Union{Real, String}})
    for _ in 1:forest.n_trees
        # get random dataset of size forest.n_features_per_tree
        current_tree_features, current_tree_labels = get_random_features(features, labels, n_features)

        # TODO Regression or Classification?
        tree = DecisionTreeClassifier(max_depth=forest.max_depth)

        fit!(tree,current_tree_features, current_tree_labels)
        push!(forest.trees, tree)
    end
end


#TODO automaticly evaluate if regression or not
function predict(forest::RandomForest, X::Matrix{Union{Real, String}}, is_regression::Bool = false)
    if isempty(forest.trees)
        error("Prediction failed because there are no trees. (Maybe you forgot to fit?)")
    end
    predictions = [predict(tree, X) for tree in forest.trees]
    return voting(predictions, is_regression)
end

#returns the most common value in a (predictions) vector for classification or the mean for regression 
function voting(vec::Vector{T}, is_regression::Bool = false) where T
    if is_regression
        return mean(vec)
    else
        return mode(vec)
    end
end