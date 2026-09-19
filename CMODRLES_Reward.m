function reward = CMODRLES_Reward(Candidates,selectedIndex,nParent,nMP,nAux)
% Index-based survival credit, excluding exact duplicate decisions.
% Incumbent/MP decisions have priority; duplicate auxiliary decisions count once.
% All evaluated auxiliary children remain in the denominator.
    assert(nAux > 0,'CMODRLES:Reward','No auxiliary offspring to credit.');
    firstAux = nParent+nMP+1;
    aux = selectedIndex(selectedIndex >= firstAux);
    decisions = Candidates.decs;
    auxDec = decisions(aux,:);
    novel = ~ismember(auxDec,decisions(1:firstAux-1,:),'rows');
    reward = size(unique(auxDec(novel,:),'rows'),1)/nAux;
end
