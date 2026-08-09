function basis = buildEvalBasis(trainPrices, trainMacro, trainingRange, numActions)
%BUILDEVALBASIS  Shared eval basis: train-fit frontier + macro z-score + test data.
%   basis = buildEvalBasis(trainPrices, trainMacro, trainingRange, numActions)
%
%   Builds the efficient frontier and macro normalizer from a TRAIN window, then
%   loads the (fixed) 2022-2025 test set through inputTestData using those train-fit
%   stats (no look-ahead). Returns one struct with everything replayPolicy and
%   windowMetrics need, so the ~25-line basis block that was copy-pasted across every
%   week10 eval script now lives in exactly one place.
%
%   Defaults (no args) = the deliverable Part-C basis: extended train 2010-2021
%   (trainingRange 3020), 15 frontier actions — the same basis the shipped tuned-DQN
%   and IQL agents were trained on.
%
%   trainPrices/trainMacro : e.g. 'data/prices_train_extended.csv' /
%                            'data/macro_train_extended.csv'.
%   trainingRange          : rows of train log-returns used for moments/normalizer.
%   numActions             : number of efficient-frontier portfolios (15).

    if nargin < 1 || isempty(trainPrices);  trainPrices  = 'data/prices_train_extended.csv'; end
    if nargin < 2 || isempty(trainMacro);   trainMacro   = 'data/macro_train_extended.csv';  end
    if nargin < 3 || isempty(trainingRange);trainingRange = 3020; end
    if nargin < 4 || isempty(numActions);   numActions   = 15;    end

    addpath(fullfile("src","utils"));   % inputTestData

    % --- frontier from train moments ---
    pricesTT   = readtable(trainPrices);
    assetNames = pricesTT.Properties.VariableNames(2:end);
    P_train    = pricesTT{:,2:end};  R_train = diff(log(P_train));
    R_sub      = R_train(1:trainingRange,:);
    mu = mean(R_sub,1)';  Sigma = cov(R_sub);
    nAssets = numel(assetNames);
    p = Portfolio("AssetList", assetNames);
    p = setDefaultConstraints(p);  p = setAssetMoments(p, mu, Sigma);
    W_frontier = estimateFrontier(p, numActions);
    wMVO       = estimateMaxSharpeRatio(p);          % Markowitz tangency baseline

    % --- macro z-score from train, applied to test (train-only fit) ---
    macroTT   = readtable(trainMacro);  M_train = macroTT{:,2:end};
    macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0) = 1;
    [R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);

    % --- static baseline weights ---
    upN    = upper(string(assetNames));
    isBond = ismember(upN, ["TLT","IEF","AGG","BND","SHY","LQD"]);
    isGold = ismember(upN, ["GLD","IAU"]);
    isEq   = ~isBond & ~isGold;
    w6040  = zeros(nAssets,1);  w6040(isEq) = 0.60/nnz(isEq);  w6040(isBond) = 0.40/nnz(isBond);
    w1N    = ones(nAssets,1)/nAssets;

    % --- pack ---
    basis = struct();
    basis.W_frontier = W_frontier;  basis.wMVO = wMVO;
    basis.w6040 = w6040;            basis.w1N  = w1N;
    basis.macroMean = macroMean;    basis.macroStd = macroStd;
    basis.R_full = R_full;          basis.M_test_z = M_test_z;
    basis.assetNames = {assetNames};basis.nAssets = nAssets;
    % fixed eval config (matches all week4-10 eval)
    basis.initialWealth = 100000;   basis.goalWealth = 102000;
    basis.horizonPeriods = 30;      basis.numEvalEpisodes = 30;  basis.evalStepSize = 30;
    basis.numActions = numActions;  basis.annualizeFactor = sqrt(252);
end
