function [Population,Fitness] = EnviromentSelect2(Population,N,MinAngle,W)
% Diversity-oriented environmental selection

    %% Basic data
    Obj  = Population.objs;
    NumQ = length(Population);
    NW   = size(W,1);          % Actual number of reference vectors

    %% Calculate angles between solutions and reference vectors
    CosQW = 1-pdist2(Obj,W,'cosine');
    CosQW = min(max(CosQW,-1),1);

    AngleQ = acos(CosQW);
    AngleQ(isnan(AngleQ)) = pi/2;

    %% Determine solutions contained in each subspace
    InSubspaceQ = AngleQ <= MinAngle;

    %% Calculate fitness
    FitnessAll = CalFitness(Obj);

    %% Environmental selection
    selectedIndex   = zeros(1,N);
    selectedFitness = zeros(1,N);
    selected        = false(1,NumQ);
    selectedCount   = 0;

    %% Step 1: Select one solution from each subspace
    for i = 1:NW

        if selectedCount >= N
            break;
        end

        %% Remaining solutions
        R = find(~selected);

        if isempty(R)
            break;
        end

        %% Solutions located in the current subspace
        Ti = find(InSubspaceQ(:,i)' & ~selected);

        if isempty(Ti)

            %% Empty subspace:
            % Select the nearest remaining solution
            [~,bestLocal] = min(AngleQ(R,i));
            x = R(bestLocal);

        else

            %% Non-empty subspace:
            % Select the solution with the best fitness
            [~,bestLocal] = min(FitnessAll(Ti));
            x = Ti(bestLocal);

        end

        %% Save selected solution
        selectedCount = selectedCount + 1;

        selectedIndex(selectedCount)   = x;
        selectedFitness(selectedCount) = FitnessAll(x);

        selected(x) = true;
    end

    %% Step 2: Fill remaining positions according to fitness
    if selectedCount < N

        R = find(~selected);

        if ~isempty(R)

            %% Sort remaining solutions by fitness
            [~,rank] = sort(FitnessAll(R),'ascend');

            %% Number of additional solutions required
            Need = min(N-selectedCount,length(R));

            %% Select the best remaining solutions
            Add = R(rank(1:Need));

            selectedIndex(selectedCount+1:selectedCount+Need) = Add;
            selectedFitness(selectedCount+1:selectedCount+Need) = ...
                FitnessAll(Add);

            selectedCount = selectedCount + Need;
        end
    end

    %% Output
    selectedIndex   = selectedIndex(1:selectedCount);
    selectedFitness = selectedFitness(1:selectedCount);

    Population = Population(selectedIndex);
    Fitness    = selectedFitness;

end