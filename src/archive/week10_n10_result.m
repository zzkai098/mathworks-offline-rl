%% Week 10 — N=10 seed-ensemble: full backtest metrics + equity curve
% Deliverable view using a 10-seed DQN ensemble (average Q over seeds 1000..10000,
% argmax). Reports the FULL metric set net 10bp (total return, Sharpe, MaxDD P90,
% success, block-bootstrap CI, DSR) vs the baseline panel, and an equity curve with
% the N=10 ensemble, the shipped single seed (1000), and 60/40 for context.
% NOTE: the ensemble return is N-sensitive (N=5 -7.8%, N=10 +12.4%, N=20 -19.6%);
% N=10 is shown per request.

clear; clc;
addpath(fullfile("src","utils"));
c = 10/1e4;  Nens = 10;  ensSeeds = 1000:1000:10000;
initialWealth = 100000;  goalWealth = 102000;  trainingRange = 3020;
horizonPeriods = 30;  numActions = 15;  numEvalEpisodes = 30;  evalStepSize = 30;
annualizeFactor = sqrt(252);  blockLen = horizonPeriods;  nBoot = 2000;  dsrTrials = 40;

%% Basis + test data
pricesTT = readtable("data/prices_train_extended.csv");
assetNames = pricesTT.Properties.VariableNames(2:end);  nAssets = numel(assetNames);
P_train = pricesTT{:,2:end};  R_train = diff(log(P_train));
R_sub = R_train(1:trainingRange,:);  mu = mean(R_sub,1)';  Sigma = cov(R_sub);
pf = Portfolio("AssetList", assetNames);  pf = setDefaultConstraints(pf);  pf = setAssetMoments(pf, mu, Sigma);
W_frontier = estimateFrontier(pf, numActions);
macroTT = readtable("data/macro_train_extended.csv");  M_train = macroTT{:,2:end};
macroMean = mean(M_train,1);  macroStd = std(M_train,0,1);  macroStd(macroStd==0)=1;
[R_full, M_test_z] = inputTestData('data/prices_test.csv','data/macro_test.csv', macroMean, macroStd);
upN=upper(string(assetNames)); isBond=ismember(upN,["TLT","IEF","AGG","BND","SHY","LQD"]); isGold=ismember(upN,["GLD","IAU"]); isEq=~isBond&~isGold;
w6040=zeros(nAssets,1); w6040(isEq)=0.60/nnz(isEq); w6040(isBond)=0.40/nnz(isBond);

%% Load ensemble critics + shipped single seed
critics = cell(Nens,1);
for i=1:Nens; critics{i} = getCritic(load(sprintf("experiments/models/dqn_seed_%d.mat",ensSeeds(i)),"FA").FA.agent); end
shipCritic = critics{1};   % seed 1000

%% Per-window gross+turnover for: N=10 ensemble, seed-1000, 60/40
[gE,tE] = replay(@(o) ensAction(critics,o), W_frontier, numEvalEpisodes, evalStepSize, horizonPeriods, R_full, M_test_z, initialWealth, goalWealth);
[gS,tS] = replay(@(o) ensAction({shipCritic},o), W_frontier, numEvalEpisodes, evalStepSize, horizonPeriods, R_full, M_test_z, initialWealth, goalWealth);
[gB,tB] = replay(@(o) w6040, [], numEvalEpisodes, evalStepSize, horizonPeriods, R_full, M_test_z, initialWealth, goalWealth);
wpE = chainedCurve(gE,tE,c,initialWealth);  wpS = chainedCurve(gS,tS,c,initialWealth);  wpB = chainedCurve(gB,tB,c,initialWealth);

%% Metrics net 10bp
[shE,ddE,twE,scE,~,nrE] = perWindow(gE,tE,c,initialWealth,goalWealth,annualizeFactor);
[shB,ddB,twB,scB,~,nrB] = perWindow(gB,tB,c,initialWealth,goalWealth,annualizeFactor);
[loE,hiE]=blockBootstrapSharpeCI(nrE,blockLen,nBoot,annualizeFactor); dsrE=deflatedSharpe(nrE,dsrTrials,[]);
[loB,hiB]=blockBootstrapSharpeCI(nrB,blockLen,nBoot,annualizeFactor); dsrB=deflatedSharpe(nrB,dsrTrials,[]);
% chained total return
trE = chained(gE,tE,c,initialWealth);  trS = chained(gS,tS,c,initialWealth);  trB = chained(gB,tB,c,initialWealth);

fprintf("=== N=10 DQN ensemble — full metrics (net 10bp) vs 60/40 ===\n");
fprintf("%-14s | TotRet | MeanSh | MaxDD P90 | Succ | 95%%CI          | DSR\n","strat");
fprintf("%-14s | %+5.1f%% | %6.2f | %8.1f%% | %4.1f | [%5.2f,%5.2f] | %.3f\n", ...
    "N=10 ensemble", 100*trE, mean(shE,'omitnan'), 100*prctile(ddE,90), sum(scE), loE, hiE, dsrE);
fprintf("%-14s | %+5.1f%% | %6.2f | %8.1f%% | %4.1f | [%5.2f,%5.2f] | %.3f\n", ...
    "60/40", 100*trB, mean(shB,'omitnan'), 100*prctile(ddB,90), sum(scB), loB, hiB, dsrB);
fprintf("(seed-1000 single: TotRet %+.1f%%)\n", 100*trS);
fprintf("(N-sensitivity: N=5 -7.8%%, N=10 +12.4%%, N=20 -19.6%%)\n");

%% Equity curves: N=10 ensemble, seed-1000, 60/40
f=figure('Visible','off','Position',[100 100 1000 520]); hold on;
plot(0:numel(wpE)-1, wpE, '-', 'LineWidth',2.4,'Color',[0.1 0.5 0.85],'DisplayName','N=10 ensemble');
plot(0:numel(wpS)-1, wpS, '-', 'LineWidth',1.6,'Color',[0.9 0.5 0.1],'DisplayName','seed-1000 (favorable single)');
plot(0:numel(wpB)-1, wpB, '--','LineWidth',1.6,'Color',[0.5 0.5 0.5],'DisplayName','60/40');
yline(initialWealth,':k','start','HandleVisibility','off');
xlabel('test trading day (30 chained 30-day windows)'); ylabel('cumulative wealth');
title('Test-period equity — N=10 DQN ensemble vs seed-1000 vs 60/40 (net 10bp)');
subtitle('ensemble return is N-sensitive (N=5/10/20 = -8%/+12%/-20%); episodic-reset, boundary rebalances uncharged');
legend('Location','northwest'); grid on;
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f,"experiments/figures/week10_n10_equity.png",'Resolution',150);
fprintf("saved -> experiments/figures/week10_n10_equity.png\n");


%% ---- locals ----
function a = ensAction(critics, obs)
    qsum = 0; for m=1:numel(critics); qv=getValue(critics{m},{obs}); qsum=qsum+qv(:); end
    [~,a] = max(qsum);
end
function [G,T] = replay(policyFn, Wf, nEp, step, H, R_full, M_test_z, iw, gw)
    nW=@(w)min(w/gw,5); tF=@(t)t/H; G=nan(nEp,H); T=nan(nEp,H);
    for ep=1:nEp
        s0=(ep-1)*step+1; if s0+H-1>size(R_full,1); break; end
        wealth=iw; wPrev=[];
        for t=1:H
            d=s0+t-1; o=[nW(wealth);tF(t);M_test_z(d,:)'];
            out=policyFn(o);
            if isscalar(out); w=Wf(:,out); else; w=out(:); end  % action idx vs static weights
            rAsset=R_full(d,:)'; rG=R_full(d,:)*w; G(ep,t)=rG;
            if t==1; T(ep,t)=0; else; T(ep,t)=sum(abs(w-wPrev)); end
            wPrev=(w.*(1+rAsset))/(1+rG); wealth=wealth*(1+rG);
        end
    end
end
function wpath = chainedCurve(G,T,c,iw)
    net=[]; for ep=1:size(G,1); if all(~isnan(G(ep,:))); net=[net, G(ep,:)-c*T(ep,:)]; end; end %#ok<AGROW>
    wpath = iw*[1, cumprod(1+net)];
end
function [sh,dd,tw,sc,tn,pooled]=perWindow(G,T,c,iw,gw,af)
    n=size(G,1); sh=nan(n,1);dd=nan(n,1);tw=nan(n,1);sc=nan(n,1);tn=nan(n,1);pooled=[];
    for ep=1:n
        g=G(ep,:)'; tu=T(ep,:)'; if any(isnan(g)); continue; end
        nr=g-c*tu; w=iw*[1;cumprod(1+nr)]; ln=diff(log(w));
        sh(ep)=mean(ln)/std(ln)*af; dd(ep)=max(cummax(w)-w)/max(w); tw(ep)=w(end); sc(ep)=double(w(end)>=gw); tn(ep)=mean(tu);
        pooled=[pooled;ln]; %#ok<AGROW>
    end
end
function tr=chained(G,T,c,iw)
    net=[]; for ep=1:size(G,1); if all(~isnan(G(ep,:))); net=[net, G(ep,:)-c*T(ep,:)]; end; end %#ok<AGROW>
    w=iw*[1,cumprod(1+net)]; tr=w(end)/w(1)-1;
end
