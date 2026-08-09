function [G, T] = replayPolicy(policyFn, basis)
%REPLAYPOLICY  Replay one policy over the chained test windows (gross + turnover).
%   [G,T] = replayPolicy(policyFn, basis) returns numEvalEpisodes x horizonPeriods
%   matrices of per-step GROSS portfolio return (G) and drift-aware TURNOVER (T).
%
%   policyFn(obs) -> either
%     * an integer action index (scalar, 1..numActions -> a frontier column), or
%     * an nAssets-vector of portfolio weights (static / random / ensemble policies).
%   obs is the 6-D z-scored state [normWealth; timeFrac; macro_z(4)] the agents were
%   trained on. Turnover is drift-aware: w_{t-1} is carried forward by realized asset
%   returns before comparing to the new target, so a hold incurs no turnover.
%
%   Cost is NOT charged here (one replay serves every cost level); windowMetrics
%   applies the bp cost when it computes net returns.

    R_full = basis.R_full;  M = basis.M_test_z;  Wf = basis.W_frontier;
    iw = basis.initialWealth;  gw = basis.goalWealth;
    H  = basis.horizonPeriods; nEp = basis.numEvalEpisodes;  step = basis.evalStepSize;

    G = nan(nEp, H);  T = nan(nEp, H);
    for ep = 1:nEp
        s0 = (ep-1)*step + 1;
        if s0 + H - 1 > size(R_full,1); break; end
        wealth = iw;  wPrev = [];
        for t = 1:H
            d = s0 + t - 1;
            o = [min(wealth/gw, 5); t/H; M(d,:)'];
            out = policyFn(o);
            if isscalar(out); w = Wf(:,out); else; w = out(:); end
            rAsset = R_full(d,:)';  rG = R_full(d,:)*w;
            G(ep,t) = rG;
            if t == 1; T(ep,t) = 0; else; T(ep,t) = sum(abs(w - wPrev)); end
            wPrev  = (w.*(1+rAsset)) / (1+rG);   % drift the held weights
            wealth = wealth*(1+rG);
        end
    end
end
