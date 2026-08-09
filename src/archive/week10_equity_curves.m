%% Week 10 — test-period equity curves: shipped agents vs a standard baseline panel
% Tracks cumulative wealth over the 2022-2025 hold-out for the two shipped agents
% (tuned-DQN, IQL — seed 1000) against a STANDARD, non-strawman baseline panel:
%   - behavior policy (uniform-random frontier action) = the offline-RL data-
%     generating policy; beating it is the canonical offline-RL success criterion
%   - 1/N equal-weight (DeMiguel et al. 2009 "hard to beat")
%   - MVO max-Sharpe (Markowitz tangency estimated on train — classic, often weak OOS)
%   - 60/40 daily constant-mix (strong balanced baseline)
% The 30 consecutive non-overlapping 30-day eval windows are chained end-to-end into
% one continuous curve; each window resets the agent's episodic state (keeps it in
% training distribution) while wealth compounds across windows. Net of turnover cost.
% Honest note: this is the SAME single test path for all strategies (a paired view);
% ~600 single-regime days, so read the curve as descriptive, not a significance claim.

clear; clc;
addpath(fullfile("src","utils"));
COST_BPS = 10;  c = COST_BPS/1e4;

%% Fixed params
initialWealth = 100000;  goalWealth = 102000;  trainingRange = 3020;
horizonPeriods = 30;  numActions = 15;  numEvalEpisodes = 30;  evalStepSize = 30;
annualizeFactor = sqrt(252);

%% Basis: frontier + macro (extended train)
pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);  nAssets = numel(assetNames);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
p = Portfolio("AssetList", assetNames);
p = setDefaultConstraints(p);  p = setAssetMoments(p, mu, Sigma);
W_frontier = estimateFrontier(p, numActions);
wMVO = estimateMaxSharpeRatio(p);                 % Markowitz tangency (train)
macroTT = readtable("data/macro_train_extended.csv");
M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);

% static baselines
upN = upper(string(assetNames));
isBond = ismember(upN,["TLT","IEF","AGG","BND","SHY","LQD"]);  isGold = ismember(upN,["GLD","IAU"]);
isEq = ~isBond & ~isGold;
w6040 = zeros(nAssets,1);  w6040(isEq)=0.60/nnz(isEq);  w6040(isBond)=0.40/nnz(isBond);
w1N = ones(nAssets,1)/nAssets;

% load shipped agents (seed 1000)
Ld = load("experiments/models/dqn_seed_1000.mat","FA");   dqnAgent = Ld.FA.agent;
Li = load("experiments/models/iql_seed_1000.mat","FI");   iqlQNet  = Li.FI.qNet;

%% Strategy specs: {name, type, handle}
strat = struct( ...
    'name', {'tuned-DQN','IQL','behavior(random)','1/N','MVO','60/40'}, ...
    'type', {'dqn','iql','random','static','static','static'}, ...
    'h',    {dqnAgent, iqlQNet, [], w1N, wMVO, w6040});
nS = numel(strat);

%% Chain the 30 consecutive windows into one continuous net-return series per strategy
curves = cell(nS,1);
for s = 1:nS
    if strcmp(strat(s).type,'random'); rng(20260730,"twister"); end  % decoupled + reproducible
    netChain = [];
    for ep = 1:numEvalEpisodes
        s0 = (ep-1)*evalStepSize + 1;
        if s0 + horizonPeriods - 1 > size(R_full,1); break; end
        [g,tu] = runWindow(strat(s), s0, horizonPeriods, R_full, M_test_z, ...
                           initialWealth, goalWealth, W_frontier, numActions);
        netChain = [netChain; g - c*tu]; %#ok<AGROW>   % net simple returns
    end
    curves{s} = initialWealth * [1; cumprod(1+netChain)];
end

%% Summary stats over the chained curve
fprintf("=== Test-period continuous performance (net %d bp, chained 30x30d windows) ===\n", COST_BPS);
fprintf("%-16s | TotRet | AnnSharpe | MaxDD | FinalWealth\n","strategy");
for s = 1:nS
    w = curves{s};  lr = diff(log(w));
    totRet = w(end)/w(1) - 1;
    shp = mean(lr)/std(lr)*annualizeFactor;
    mdd = max(cummax(w)-w)/max(w);
    fprintf("%-16s | %5.1f%% | %8.2f | %4.1f%% | %8.0f\n", ...
        strat(s).name, 100*totRet, shp, 100*mdd, w(end));
end

%% Plot equity curves
f = figure('Visible','off','Position',[100 100 1000 560]);
hold on;
cols = lines(nS);
for s = 1:nS
    lw = 2.2; ls = '-';
    if strcmp(strat(s).type,'static') || strcmp(strat(s).type,'random'); lw = 1.4; ls = '--'; end
    plot(0:numel(curves{s})-1, curves{s}, ls, 'Color', cols(s,:), 'LineWidth', lw, 'DisplayName', strat(s).name);
end
yline(initialWealth, ':k', 'start', 'HandleVisibility','off');
xlabel('test trading day (30 chained 30-day windows)'); ylabel('cumulative wealth');
title(sprintf('Test-period equity curves — shipped agents vs baseline panel (net %d bp)', COST_BPS));
subtitle('chained policy behavior (episodic state reset each window, in-distribution); window-boundary rebalances uncharged');
legend('Location','northwest'); grid on;
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f, "experiments/figures/week10_equity_curves.png", 'Resolution', 150);
fprintf("\nsaved -> experiments/figures/week10_equity_curves.png\n");


%% ---- local: one window's gross return + drift-aware turnover for any strategy ----
function [grossRet, turnover] = runWindow(st, startEval, H, R_full, M_test_z, ...
        initialWealth, goalWealth, W_frontier, numActions)
    normalizeWealth = @(w) min(w/goalWealth, 5);
    timeFrac = @(t) t / H;
    grossRet = zeros(H,1);  turnover = zeros(H,1);
    wealth = initialWealth;  wPrev = [];
    for t = 1:H
        dayIdx = startEval + t - 1;
        switch st.type
            case 'static'
                w = st.h(:);
            case 'random'
                w = W_frontier(:, randi(numActions));
            case 'dqn'
                obs = { [normalizeWealth(wealth); timeFrac(t); M_test_z(dayIdx,:)'] };
                a = st.h.getAction(obs);  w = W_frontier(:, a{1});
            case 'iql'
                obs = [normalizeWealth(wealth); timeFrac(t); M_test_z(dayIdx,:)'];
                q = extractdata(forward(st.h, dlarray(obs,'CB')));
                [~,a] = max(q);  w = W_frontier(:, a);
        end
        w = w(:);
        rAsset = R_full(dayIdx,:)';  rG = R_full(dayIdx,:) * w;
        grossRet(t) = rG;
        if t == 1; turnover(t) = 0; else; turnover(t) = sum(abs(w - wPrev)); end
        wPrev = (w .* (1 + rAsset)) / (1 + rG);
        wealth = wealth * (1 + rG);
    end
end
