%% Week 10 — test-period equity curves with N=10 ENSEMBLE agents vs baseline panel
% Same 6-curve figure as week10_equity_curves.m, but the two agents are N=10 seed
% ENSEMBLES (average Q over seeds 1000..10000, argmax) instead of single seed 1000.
% Baselines unchanged (seed-independent). Chained 30x30d windows, net 10bp.
% NOTE: ensemble return is N-sensitive; N=10 shown per request.

clear; clc;
addpath(fullfile("src","utils"));
c = 10/1e4;  ensSeeds = 1000:1000:10000;
initialWealth = 100000;  goalWealth = 102000;  trainingRange = 3020;
horizonPeriods = 30;  numActions = 15;  numEvalEpisodes = 30;  evalStepSize = 30;  annualizeFactor = sqrt(252);

pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);  nAssets = numel(assetNames);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
pf = Portfolio("AssetList", assetNames);  pf = setDefaultConstraints(pf);  pf = setAssetMoments(pf, mu, Sigma);
W_frontier = estimateFrontier(pf, numActions);  wMVO = estimateMaxSharpeRatio(pf);
macroTT = readtable("data/macro_train_extended.csv");  M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);
upN=upper(string(assetNames)); isBond=ismember(upN,["TLT","IEF","AGG","BND","SHY","LQD"]); isGold=ismember(upN,["GLD","IAU"]); isEq=~isBond&~isGold;
w6040=zeros(nAssets,1); w6040(isEq)=0.60/nnz(isEq); w6040(isBond)=0.40/nnz(isBond);
w1N=ones(nAssets,1)/nAssets;

% ensemble members
dqnC = cell(10,1); iqlN = cell(10,1);
for i=1:10
    dqnC{i}=getCritic(load(sprintf("experiments/models/dqn_seed_%d.mat",ensSeeds(i)),"FA").FA.agent);
    iqlN{i}=load(sprintf("experiments/models/iql_seed_%d.mat",ensSeeds(i)),"FI").FI.qNet;
end

names = {'tuned-DQN (N=10 ens)','IQL (N=10 ens)','behavior(random)','1/N','MVO','60/40'};
types = {'dqn_ens','iql_ens','random','static','static','static'};
statW = {[],[],[],w1N,wMVO,w6040};
nS = numel(names);  curves = cell(nS,1);

rng(20260730,"twister");
for s=1:nS
    if strcmp(types{s},'random'); rng(20260730,"twister"); end
    net=[];
    for ep=1:numEvalEpisodes
        s0=(ep-1)*evalStepSize+1; if s0+horizonPeriods-1>size(R_full,1); break; end
        wealth=initialWealth; wPrev=[];
        for t=1:horizonPeriods
            d=s0+t-1; o=[min(wealth/goalWealth,5); t/horizonPeriods; M_test_z(d,:)'];
            switch types{s}
                case 'dqn_ens'; qs=0; for m=1:10; qs=qs+getValue(dqnC{m},{o}); end; [~,a]=max(qs); w=W_frontier(:,a);
                case 'iql_ens'; qs=0; for m=1:10; qs=qs+extractdata(forward(iqlN{m},dlarray(o,'CB'))); end; [~,a]=max(qs); w=W_frontier(:,a);
                case 'random';  w=W_frontier(:,randi(numActions));
                case 'static';  w=statW{s};
            end
            w=w(:); rAsset=R_full(d,:)'; rG=R_full(d,:)*w;
            if t==1; tu=0; else; tu=sum(abs(w-wPrev)); end
            net(end+1)=rG-c*tu; %#ok<AGROW>
            wPrev=(w.*(1+rAsset))/(1+rG); wealth=wealth*(1+rG);
        end
    end
    curves{s}=initialWealth*[1; cumprod(1+net')];
end

fprintf("=== N=10 ensemble agents vs baselines — chained test total return (net 10bp) ===\n");
for s=1:nS
    w=curves{s}; lr=diff(log(w));
    fprintf("%-22s | TotRet %+6.1f%% | Sharpe %5.2f | MaxDD %4.1f%%\n", ...
        names{s}, 100*(w(end)/w(1)-1), mean(lr)/std(lr)*annualizeFactor, 100*max(cummax(w)-w)/max(w));
end

f=figure('Visible','off','Position',[100 100 1050 580]); hold on; cols=lines(nS);
for s=1:nS
    lw=2.2; ls='-'; if any(strcmp(types{s},{'static','random'})); lw=1.4; ls='--'; end
    plot(0:numel(curves{s})-1, curves{s}, ls, 'Color',cols(s,:),'LineWidth',lw,'DisplayName',names{s});
end
yline(initialWealth,':k','start','HandleVisibility','off');
xlabel('test trading day (30 chained 30-day windows)'); ylabel('cumulative wealth');
title('Test-period equity — N=10 ensemble agents vs baseline panel (net 10bp)');
subtitle('agents = 10-seed Q-average ensembles; ensemble return is N-sensitive; episodic-reset, boundary rebalances uncharged');
legend('Location','northwest'); grid on;
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f,"experiments/figures/week10_equity_ensemble.png",'Resolution',150);
fprintf("saved -> experiments/figures/week10_equity_ensemble.png\n");
