| Complete predictor | Route to a sound partial enclosure | Additional requirement |
|---|---|---|
| Random forest / extra trees | Bound each leaf's class-probability vector and average tree bounds | Preserve probability averaging and the frozen temperature transformation; a boosted-logit sum is not equivalent |
| XGBoost / LightGBM multiclass trees | Sum reachable-leaf bounds for each class score, then bound the link | Validate intercepts, class/tree indexing, missing and categorical splits, and libraries' exact output semantics |
| Larger numerical trees | Use multiple mask words or another candidate-leaf representation | Charge extra update and interval-extraction work; the one-word runtime does not transfer automatically |
| RBF SVC | Enclose kernels and decision values under partial features | Also enclose the fitted pairwise probabilities and multiclass coupling; the present softmax score formula cannot simply replace native SVC semantics |
| Neural network | Propagate sound feature bounds to logits, then probabilities | Justify bounds for uncomputed signal features and control relaxation cost; empirical training minima/maxima alone do not enclose unseen inputs |
