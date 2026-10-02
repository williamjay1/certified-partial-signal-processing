# Method derivation for the new manuscript

This document supplies exact-arithmetic results. Numerical implementation and measured performance are separate evidence.

## Loss and feasible outputs

Let f be a frozen K-class full-information predictor, q=f(x), and let the observable state contain only already computed feature blocks. A sound partial evaluation gives an outer set Q of possible complete probabilities, containing the actual q. Half-Brier loss is L(p,y)=.5||p-e_y||^2. Define the uniform relative-loss certificate

    B(p,Q) = max_{y=1,...,K; q in Q} [L(p,y)-L(q,y)].

The guarantee is relative to the same frozen predictor, not to the Bayes classifier. It is pointwise in x and y. If B<=delta, every empirical average of the same cases and arbitrary labels has relative half-Brier increase at most delta. No distributional calibration hypothesis is needed for this implication.

## Interval-simplex minimax theorem

For nonempty Q={q in simplex: l<=q<=u}, set d_y=min_Q L(q,y) and a_y=.5-d_y. Expanding the loss gives B(p,Q)=.5||p||^2+max_y(a_y-p_y). The unique minimizer is p*=projection_simplex(a), and its exact worst excess is .5||p*||^2+max_y(a_y-p*_y).

Proof: replace the maximum coordinate by max_{w in simplex} w dot (a-p). The objective is convex in p, affine in w and both domains are convex compact, so minimax interchange applies. For any w in the simplex, the inner minimum over p is attained at p=w and equals -.5||w||^2. Thus the dual maximizes a dot w-.5||w||^2, whose unique solution is projection_simplex(a). The projection KKT conditions imply a_y-p*_y equals a common value on positive coordinates and is no larger on zero coordinates; this gives a saddle point and establishes that p* is also the primal minimizer. Strict convexity of the quadratic ensures uniqueness.

The theorem is tight for the outer interval-simplex set. Independent tree bounds and probability intervals may admit outputs not jointly achievable by any input, so it is not asserted to be tight for the model's exact reachable-output set.

## Bounded projection subproblems

For label y, the minimizing q_y is min(u_y,1-sum_{j!=y} l_j). Increasing q_y by transferring positive feasible mass from another coordinate decreases the distance to e_y, until q_y reaches that maximum. Holding q_y fixed, the remaining coordinates solve minimum squared norm with their lower/upper bounds and fixed total mass, hence q_j=clip(v,l_j,u_j), with v chosen to satisfy the total. Sorting lower/upper breakpoints permits a finite piecewise-linear water-fill solution. These K small subproblems replace online generic constrained optimization.

A conservative floating-point certificate must not use a merely feasible approximate objective as d_y: it is an upper bound on d_y and could understate B. For the equality-constraint multiplier nu and t_i=1{i=y}, a valid exact-arithmetic dual lower value is

    dLB(nu)=.5-nu+sum_i min_{l_i<=z<=u_i} [.5*z*z+(nu-t_i)*z].

The minimizing z is clip(t_i-nu,l_i,u_i). Any nu provides a lower bound. Use dLB with outward numerical adjustments when constructing a and the final certificate. A lower dual value yields a conservative certificate, although its p need not be exactly the mathematical minimax optimum. Floating-point validation is reported separately from theorem validity.

## Tree output enclosure

Known split features choose only the branch consistent with their values. Unknown split features retain both branches. Taking the minimum and maximum over reachable leaves gives each tree's output interval. Summing intervals per class and adding the frozen baseline gives logit intervals [L_k,U_k]. The actual full score is enclosed even when repeated-feature or cross-tree correlations are relaxed. For a fixed softmax,

    lower_y = 1/(1+sum_{k!=y} exp(U_k-L_y)),
    upper_y = 1/(1+sum_{k!=y} exp(L_k-U_y)).

Monotonicity of the softmax in the own and other logits proves enclosure. A fixed positive temperature scales intervals before applying the formula. Use stable exponential evaluation and conservative numerical adjustments. Training-set value ranges are not certified bounds on unseen feature values.

## Consequences and limiting cases

When Q={q}, a=q-.5||q||^2*1, the projection is q and the certificate is zero. For the entire simplex, the output is uniform and the certificate is .5*(1-1/K). If new observations shrink Q, the exact optimum cannot increase, since every fixed output has a no-larger worst case.

The relative-loss guarantee also entails probability proximity: for the actual complete q, sum_y q_y[L(p,y)-L(q,y)]=.5||p-q||^2. Therefore a certificate at most delta implies ||p-q||_2<=sqrt(2*delta). This is an algebraic consequence, not an assumption that q is the actual label distribution. If p has a unique winning margin greater than 2*sqrt(delta), then q has the same winning label, by the Cauchy-Schwarz bound on coordinate-difference perturbations. Otherwise class agreement is not guaranteed; the principal output guarantee concerns probability loss.

The same exact theorem applies to any sound outer Q, but the interval-simplex structure is what makes the online minimax output inexpensive. It applies to multiclass probabilities and every label; it does not establish that the frozen predictor is accurate, calibrated, or optimal.
