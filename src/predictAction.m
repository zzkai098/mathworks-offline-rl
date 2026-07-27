function [actionIdx, weights] = predictAction(FA, wealth, t, macroRaw)
%PREDICTACTION  Deliverable inference interface for the GBWM tuned-DQN agent.
%   [actionIdx, weights] = predictAction(FA, wealth, t, macroRaw)
%
%   FA        struct loaded from experiments/models/FinalAgent.mat (field FA)
%   wealth    current portfolio wealth (currency units, e.g. 101000)
%   t         current step index within the horizon (1..FA.horizonPeriods)
%   macroRaw  1x4 raw macro vector [DGS10, T10Y2Y, VIXCLS, DFF] (same order/units
%             as data/macro_*.csv columns 2:end), z-scored internally with the
%             train-fit FA.macroMean / FA.macroStd. Pass the macro observed AS OF
%             the decision (contemporaneous with `wealth`); the agent was trained
%             on same-day macro, so no lag/offset is applied here.
%
%   actionIdx integer 1..FA.numActions selecting an efficient-frontier portfolio
%   weights   nAssets-vector of portfolio weights (one weight per asset in
%             FA.assetNames; sums to ~1). NOTE: this is an ASSET vector, not an
%             action vector — nAssets and numActions are both 15 here by
%             coincidence; index `weights` by asset, never by action.
%
%   The agent is greedy (argmax Q). State is built exactly as in training:
%   [ min(wealth/goal, clip); t/horizon; z-scored macro ].

    macroRaw = macroRaw(:)';                                   % row
    if numel(macroRaw) ~= numel(FA.macroMean)
        error("predictAction:macroSize", ...
            "macroRaw has %d elements; expected %d [DGS10,T10Y2Y,VIXCLS,DFF].", ...
            numel(macroRaw), numel(FA.macroMean));
    end

    normWealth = min(wealth / FA.goalWealth, FA.wealthClip);
    timeFrac   = t / FA.horizonPeriods;
    macroZ     = (macroRaw - FA.macroMean) ./ FA.macroStd;
    obs        = [normWealth; timeFrac; macroZ(:)];

    a = getAction(FA.agent, {obs});
    actionIdx = a{1};
    weights   = FA.W_frontier(:, actionIdx);
end
