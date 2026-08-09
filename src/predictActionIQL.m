function [actionIdx, weights] = predictActionIQL(FI, wealth, t, macroRaw)
%PREDICTACTIONIQL  Inference interface for the GBWM IQL agent (self-stabilizing).
%   [actionIdx, weights] = predictActionIQL(FI, wealth, t, macroRaw)
%
%   FI        struct loaded from experiments/models/iql_seed_*.mat (field FI). Holds
%             FI.qNet (a dlnetwork), the frontier basis, and the train-fit macro
%             normalizer — same state contract as predictAction (the DQN one).
%   wealth    current portfolio wealth
%   t         step index within the horizon (1..FI.horizonPeriods)
%   macroRaw  1x4 raw macro [DGS10, T10Y2Y, VIXCLS, DFF], contemporaneous with wealth
%
%   actionIdx integer 1..FI.numActions (efficient-frontier portfolio)
%   weights   nAssets-vector of portfolio weights (sums to ~1)
%
%   Greedy policy = argmax_a Q(s,a) via forward(FI.qNet, obs). In discrete IQL this
%   equals argmax advantage (V(s) is action-independent), so no separate policy net /
%   AWR temperature is needed. State is built identically to training.

    macroRaw = macroRaw(:)';
    if numel(macroRaw) ~= numel(FI.macroMean)
        error("predictActionIQL:macroSize", ...
            "macroRaw has %d elements; expected %d [DGS10,T10Y2Y,VIXCLS,DFF].", ...
            numel(macroRaw), numel(FI.macroMean));
    end

    normWealth = min(wealth / FI.goalWealth, FI.wealthClip);
    timeFrac   = t / FI.horizonPeriods;
    macroZ     = (macroRaw - FI.macroMean) ./ FI.macroStd;
    obs        = [normWealth; timeFrac; macroZ(:)];

    q = extractdata(forward(FI.qNet, dlarray(obs, 'CB')));
    [~, actionIdx] = max(q);
    weights = FI.W_frontier(:, actionIdx);
end
