%% Package step 1 — train + SAVE the deliverable agent (tuned-DQN = "fixed 6.1")
% The week9/week10 experiments trained agents but kept only METRICS (never the
% weights). This produces the ONE deliverable artifact: a tuned-DQN trained at the
% decided config and saved with everything predictAction needs to run in a clean
% session. NOT a new experiment — the config is fixed a priori from the diagnosis:
%   Polyak target tau=1e-3, critic LR=1e-4  (the two-hyperparameter fix), standard
%   6.1 reward, seed=1000, ME=100 (converged/sane operating point, pre-overtraining).
%
% Output: experiments/models/FinalAgent.mat holding a struct `FA` with the agent +
% frontier basis + macro normalizer + state config. Load it and call
%   [aIdx, w] = predictAction(FA, wealth, t, macroRaw)
% Ends with a reload + round-trip check (the "loads and runs" acceptance criterion).

% Parameterizable (no `clear` — would wipe -batch-passed vars). Defaults = the
% shipped deliverable (seed 1000 -> FinalAgent.mat). For the multi-seed backtest,
% pass SEED and MODELPATH (e.g. experiments/models/dqn_seed_2000.mat).
if ~exist('SEED','var');  SEED = 1000;   end
if ~exist('ME','var');    ME = 100;      end
TSF = 1e-3;  LRATE = 1e-4;

%% Fixed Parameters (identical to 6.1 / week9d)
initialWealth = 100000;  goalWealth = 102000;  contribution = 0;  wealthClip = 5;
trainingRange = 3020;    horizonPeriods = 30;  numEpisodes = 200;  numActions = 15;
hiddenUnits = 32;        miniBatchSize = 256;  discountFactor = 0.995;  stepsPerEpoch = 400;
BETA_HIGH = 8.0;  BETA_LOW = 2.0;  DD_THR = 0.03;
VIX_THR_Z = 1.5;  SLOPE_THR_Z = -1.5;  COL_T10Y2Y = 2;  COL_VIX = 3;

modelDir = fullfile("experiments","models");
if ~isfolder(modelDir); mkdir(modelDir); end
logsRoot = fullfile("experiments","logs","final_agent", sprintf("seed_%d", SEED));

%% Train prices + frontier
pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
pf = Portfolio("AssetList", assetNames);
pf = setDefaultConstraints(pf);  pf = setAssetMoments(pf, mu, Sigma);
W_frontier = estimateFrontier(pf, numActions);

%% Macro + stress
macroTT = readtable("data/macro_train_extended.csv");
M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
M_train_z = (M_train - macroMean) ./ macroStd;
isStressDay_train = (M_train_z(:,COL_VIX) > VIX_THR_Z) | (M_train_z(:,COL_T10Y2Y) < SLOPE_THR_Z);

normalizeWealth = @(w) min(w/goalWealth, wealthClip);
timeFrac = @(t) t / horizonPeriods;
obsDim = 2 + size(M_train_z,2);

%% Generate + freeze episodes (deterministic, standard reward, bonus=1.0)
rng(SEED, "twister");
if isfolder(logsRoot); delete(fullfile(logsRoot,"*.mat")); else; mkdir(logsRoot); end
stepSize = max(1, floor((trainingRange - horizonPeriods)/numEpisodes));
starts = (1 + (0:numEpisodes-1)'*stepSize);
starts = starts(starts + horizonPeriods - 1 <= trainingRange);
for ec = 1:numel(starts)
    startIdx = starts(ec);
    expStruct = repmat(struct("Observation",[],"Action",[],"Reward",[], ...
        "NextObservation",[],"IsDone",[]), horizonPeriods, 1);
    wealth = initialWealth;  peakW = initialWealth;
    for t = 1:horizonPeriods
        dayIdx = startIdx + t - 1;
        obs = [normalizeWealth(wealth); timeFrac(t); M_train_z(dayIdx,:)'];
        a = randi(numActions);  w = W_frontier(:,a);
        wealthNext = (wealth + contribution)*(1 + R_sub(dayIdx,:)*w);
        stressFlag = isStressDay_train(dayIdx);
        betaT = BETA_HIGH*stressFlag + BETA_LOW*~stressFlag;
        peakW = max(peakW, wealthNext);  DD_t = (peakW - wealthNext)/peakW;
        reward = log(wealthNext/wealth) - betaT*max(0, DD_t - DD_THR);
        if t==horizonPeriods && wealthNext>=goalWealth; reward = reward + 1.0; end
        nd = min(dayIdx+1, size(M_train_z,1));
        nextObs = [normalizeWealth(wealthNext); timeFrac(min(t+1,horizonPeriods)); M_train_z(nd,:)'];
        expStruct(t).Observation={obs}; expStruct(t).Action={a};
        expStruct(t).Reward=reward; expStruct(t).NextObservation={nextObs};
        expStruct(t).IsDone=(t==horizonPeriods);
        wealth = wealthNext;
    end
    exp = expStruct;
    save(fullfile(logsRoot, sprintf("loggedData%03d.mat", ec)), "exp");
end

%% Build + train tuned-DQN (Polyak target, LR 1e-4)
rng(SEED, "twister");
myReadFcn = @(fname) load(fname, "exp").exp;
fds = fileDatastore(fullfile(logsRoot,"loggedData*.mat"), "ReadFcn", myReadFcn);
reset(fds);
obsInfo = rlNumericSpec([obsDim 1], Name="GBWMObservation");
actInfo = rlFiniteSetSpec(1:numActions, Name="AllocationIndex");
criticLG = layerGraph([
    featureInputLayer(obsDim, Name="state")
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(hiddenUnits)
    reluLayer
    fullyConnectedLayer(numActions, Name="Qout")]);
critic = rlVectorQValueFunction(dlnetwork(criticLG), obsInfo, actInfo);
agentOpts = rlDQNAgentOptions(DiscountFactor=discountFactor, ExperienceBufferLength=1e6, ...
    MiniBatchSize=miniBatchSize, TargetUpdateMethod="smoothing", TargetSmoothFactor=TSF, ...
    UseDoubleDQN=true);
agentOpts.CriticOptimizerOptions.LearnRate = LRATE;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1;
agent = rlDQNAgent(critic, agentOpts);

fprintf("Training final agent: tuned-DQN Polyak tau=%g LR=%g seed=%d ME=%d ...\n", TSF, LRATE, SEED, ME);
tfdOpts = rlTrainingFromDataOptions(MaxEpochs=ME, NumStepsPerEpoch=stepsPerEpoch, ...
    Plots="none", Verbose=false);
rng(SEED + 12345, "twister");
trainFromData(agent, fds, tfdOpts);

%% Bundle everything predictAction needs; SAVE
FA = struct();
FA.agent        = agent;
FA.W_frontier   = W_frontier;      % action index -> asset weights
FA.assetNames   = assetNames;
FA.macroMean    = macroMean;       % raw macro -> z-score
FA.macroStd     = macroStd;
FA.goalWealth   = goalWealth;
FA.wealthClip   = wealthClip;
FA.horizonPeriods = horizonPeriods;
FA.numActions   = numActions;
FA.meta = struct('learner','tuned-DQN (Polyak tau=1e-3, LR=1e-4) = fixed 6.1', ...
    'seed',SEED,'maxEpochs',ME,'trainWindow','2010-2021 extended (trainingRange=3020)', ...
    'state','[normWealth; timeFrac; DGS10_z; T10Y2Y_z; VIXCLS_z; DFF_z]', ...
    'reward','log-ret - regime-gated drawdown penalty (beta 8/2) + terminal bonus 1.0', ...
    'expectedValQ','~1.0-1.2 (bounded, under ~1.14 economic ceiling)', ...
    'seedRationale','seed=1000 is representative NOT optimal: the week10 matrix showed seeds statistically equivalent on policy reproducibility, so NO seed selection was performed; a different seed yields an equivalent agent. ME=100 = converged/pre-overtraining operating point.', ...
    'note','greedy argmax Q; macro z-scored with FA.macroMean/Std; wealth clipped at 5x goal');
if exist('MODELPATH','var'); modelPath = MODELPATH;
else; modelPath = fullfile(modelDir, "FinalAgent.mat"); end
save(modelPath, "FA", "-v7.3");
d = dir(modelPath);
fprintf("Saved deliverable -> %s (%.2f MB)\n", modelPath, d.bytes/1e6);
if d.bytes > 20e6
    fprintf("  WARNING: file >20MB — likely carrying the experience buffer; consider stripping.\n");
end

%% In-process reload SMOKE test (a genuine clean-session test is run separately via
%% matlab -batch — see scripts/check_final_agent.sh — because clearvars stays in-process)
clearvars -except modelPath
S = load(modelPath);  FA = S.FA;
wealth = 101000; t = 5;
macroRaw = [4.2 -0.5 18 5.3];   % realistic raw macro levels (DGS10,T10Y2Y,VIX,DFF)
[aIdx, w] = predictAction(FA, wealth, t, macroRaw);
fprintf("Reload smoke test: predictAction(wealth=%.0f, t=%d) -> action %d\n", wealth, t, aIdx);
fprintf("  top holdings:");
[wv, ord] = sort(w, 'descend');
for k = 1:3; fprintf("  %s=%.0f%%", FA.assetNames{ord(k)}, 100*wv(k)); end
fprintf("\n  weights sum = %.3f (should be ~1)\n", sum(w));
fprintf("DONE (in-process). Run scripts/check_final_agent.sh for the fresh-process test.\n");
