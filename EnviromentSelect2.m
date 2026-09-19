function [Population,Fitness,Z] = EnviromentSelect2(Population,Offspring,W,Z,nr)
% Convergence-oriented environmental selection


    %% Basic data
    N = length(Population);

    if nargin < 5
        nr = 2;
    end

    %% Avoid zero weights
    W = max(W,1e-6);

    %% Check population size
    if size(W,1) ~= N
        error('CMODRLES:WeightNumber', ...
            'The number of weight vectors must equal the AP population size.');
    end

    %% No offspring
    if isempty(Offspring)
        Fitness = CalFitness(Population.objs);
        return;
    end

    %% Update AP using Tchebycheff approach
    for i = 1:length(Offspring)

        %% Update the ideal point
        Z = min(Z,Offspring(i).obj);

        %% Random order of subproblems
        P = randperm(N);

        %% Tchebycheff value of existing solutions
        g_old = max( ...
            abs(Population(P).objs - repmat(Z,length(P),1)) ...
            ./ W(P,:),[],2);

        %% Tchebycheff value of the offspring
        g_new = max( ...
            repmat(abs(Offspring(i).obj-Z),length(P),1) ...
            ./ W(P,:),[],2);

        %% Replace at most nr solutions
        Replace = find(g_old >= g_new,nr);

        Population(P(Replace)) = Offspring(i);
    end

    %% Fitness is only used for mating selection
    Fitness = CalFitness(Population.objs);

end