function [Population,Fitness] = EnviromentSelect2(Population,N,MinAngle,W)
% Diversity-oriented environmental selection

    %% Basic data
    Obj = Population.objs;
    NumQ = length(Population);

    %% Calculate angles between solutions and reference vectors
    CosQW = 1-pdist2(Obj,W,'cosine');
    CosQW = min(max(CosQW,-1),1);

    AngleQ = acos(CosQW);
    AngleQ(isnan(AngleQ)) = pi/2;

    %% Determine solutions contained in each subspace
    InSubspaceQ = AngleQ <= MinAngle;

    %% Calculate epsilon-relaxed constrained fitness
    FitnessAll = CalFitness(Obj);

    %% Environmental selection
    selectedIndex   = zeros(1,N);
    selectedFitness = zeros(1,N);
    selected        = false(1,NumQ);
    selectedCount   = 0;

    %% Traverse all subspaces
    for i = 1:N

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

            %% Empty subspace: select the nearest remaining solution
            [~,bestLocal] = min(AngleQ(R,i));
            x = R(bestLocal);

        else

            %% Non-empty subspace: select the best constrained solution
            [~,bestLocal] = min(FitnessAll(Ti));
            x = Ti(bestLocal);

        end

        selectedCount = selectedCount + 1;

        selectedIndex(selectedCount)   = x;
        selectedFitness(selectedCount) = FitnessAll(x);

        selected(x) = true;
    end

    %% Fill remaining positions if necessary
    while selectedCount < N

        R = find(~selected);

        if isempty(R)
            break;
        end

        %% Find the closest solution-subspace pair
        AngleRemain = AngleQ(R,:);

        [~,index] = min(AngleRemain(:));
        [r,~] = ind2sub(size(AngleRemain),index);

        x = R(r);

        selectedCount = selectedCount + 1;

        selectedIndex(selectedCount)   = x;
        selectedFitness(selectedCount) = FitnessAll(x);

        selected(x) = true;
    end

    %% Output
    selectedIndex = selectedIndex(1:selectedCount);

    Population = Population(selectedIndex);
    Fitness    = selectedFitness(1:selectedCount);

end