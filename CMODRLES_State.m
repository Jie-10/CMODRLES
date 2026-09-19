function state = CMODRLES_State(PopObj,PopCon,W,Ref)
% State 1: average overall constraint violation of MP.
% State 2: division-form Tchebycheff set indicator over reference vectors.
% The objective normalization range is maintained from evaluated solutions.

    %% Average constraint violation
    CV = sum(max(0,PopCon),2);
    average_cv = mean(CV);

    %% Objective normalization based only on observed solutions
    span = max(Ref.hi-Ref.lo,Ref.spanFloor);
    F = abs((PopObj-Ref.lo)./span);

    %% Division-form Tchebycheff set indicator
    K = size(W,1);
    value = zeros(K,1);
    for i = 1:K
        g = max(F./max(W(i,:),1e-6),[],2);
        value(i) = min(g);
    end
    average_tch = mean(value);

    state = [average_cv,average_tch];
end
