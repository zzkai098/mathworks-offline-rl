%% Round 1 — val/test-ISOLATED cost-aware backtest (no retraining)
% New ground truth for this loop: a strategy must SIGNIFICANTLY beat 60/40.
% To make that an honest question (not p-hacking on the same test set), we
% enforce a strict selection/evaluation split:
%   - Training set UNCHANGED (extended 2010-2021) → reuse the already-trained
%     agents; no retraining.
%   - The 2022-2025 out-of-sample period is split CHRONOLOGICALLY into
%       val  = first VAL_WIN windows  (used to SELECT the winning config)
%       test = remaining windows      (LOCKED; touched once, for the verdict)
%     val-early / test-late = honest deployment order, no look-ahead.
%
% Selection rule (on val only): among the RL configs, pick the one with the
% largest val net-Sharpe EDGE over 60/40. Then report that winner (and all
% configs, for transparency) on the locked test with the same honest estimator
% as cql_winner_harness (per-seed pooled-daily Sharpe, per-seed block-bootstrap
% CI averaged across seeds, DSR with the full 17-trial haircut).
%
% Honest caveats (printed): (1) test has few windows → low power; (2) alpha=1
% was originally eyeballed on the whole 2022-2024 region, so re-selecting on
% val-only here is "as clean as possible without retraining", not a virgin test.

clear; clc;
addpath(fullfile("src","utils"));

%% Params
initialWealth=100000; goalWealth=102000; trainingRange=3020;
horizonPeriods=30; numActions=15; nDense=200;
evalStepSize=30;                 % non-overlapping 30-day windows
seeds=[1000 2000 3000 4000 5000]; nSeeds=numel(seeds);
costBps=[0 5 10 20]; nCost=numel(costBps); annualizeFactor=sqrt(252);
blockLen=horizonPeriods; nBoot=2000;
VAL_WIN = 13;                    % first 13 windows -> val; rest -> locked test

% Honest DSR haircut over the FULL trial universe (incl. losers)
trialSharpesAnnual = [0.59 -0.01 0.25 0.37 0.50 0.36 0.62 0.55 0.51 ...
                      0.66 0.54 0.57 0.29 0.35 0.24 0.70 0.78];
dsrNumTrials      = numel(trialSharpesAnnual);
dsrTrialSharpeVar = var(trialSharpesAnnual)/252;

logRootCQL03 = fullfile("experiments","logs","cql_ddpg","sweep_alpha0.3");
logRootCQL1  = fullfile("experiments","logs","cql_ddpg","sweep_alpha1");
logRootCQL3  = fullfile("experiments","logs","cql_ddpg","sweep_alpha3");
logRootDDPG  = fullfile("experiments","logs","cql_ddpg","alpha_0");
logRoot61    = fullfile("experiments","logs","week6_1_regime_gated");

%% Shared basis + test data
pricesTT=readtable("data/prices_train_extended.csv");
assetNames=pricesTT.Properties.VariableNames(2:end);
R_sub=diff(log(pricesTT{:,2:end})); R_sub=R_sub(1:trainingRange,:);
mu=mean(R_sub,1)'; Sigma=cov(R_sub); nAssets=numel(assetNames);
p=Portfolio("AssetList",assetNames); p=setDefaultConstraints(p); p=setAssetMoments(p,mu,Sigma);
W_frontier=estimateFrontier(p,numActions);
rmin=estimatePortReturn(p,W_frontier(:,1)); rmax=estimatePortReturn(p,W_frontier(:,end));
W_dense=estimateFrontierByReturn(p,linspace(rmin,rmax,nDense));

macroTT=readtable("data/macro_train_extended.csv"); M_train=macroTT{:,2:end};
macroMean=mean(M_train,1); macroStd=std(M_train,0,1); macroStd(macroStd==0)=1;
[R_full,M_test_z]=inputTestData('data/prices_test.csv','data/macro_test.csv',macroMean,macroStd);

nWinTotal=floor((size(R_full,1)-horizonPeriods)/evalStepSize)+1;
valIdx=1:VAL_WIN; testIdx=(VAL_WIN+1):nWinTotal;
fprintf("Windows: total %d -> val %d (first) / test %d (locked, last)\n", ...
    nWinTotal, numel(valIdx), numel(testIdx));

%% 60/40 weights
bondNames={'TLT','IEF','AGG','BND','SHY','LQD'}; goldNames={'GLD','IAU'};
upN=upper(string(assetNames)); isBond=ismember(upN,upper(string(bondNames)));
isGold=ismember(upN,upper(string(goldNames))); isEq=~isBond&~isGold;
w6040=zeros(nAssets,1); w6040(isEq)=0.60/nnz(isEq); w6040(isBond)=0.40/nnz(isBond);

%% Strategy roster (all reuse already-trained agents)
strat = struct( ...
  'name',{'CQL a=0.3','CQL a=1','CQL a=3','DDPG a=0','6.1 DQN','60/40'}, ...
  'kind',{'ddpg','ddpg','ddpg','ddpg','dqn','static'}, ...
  'basis',{W_dense,W_dense,W_dense,W_dense,W_frontier,[]}, ...
  'logRoot',{logRootCQL03,logRootCQL1,logRootCQL3,logRootDDPG,logRoot61,''});
nStrat=numel(strat);

% Precompute gross+turnover for ALL windows once (per seed for RL)
for s=1:nStrat
    if strcmp(strat(s).kind,'static')
        [gW,tW]=replayAll([],'static',[],w6040,nWinTotal,evalStepSize, ...
            horizonPeriods,R_full,M_test_z,initialWealth,goalWealth);
        strat(s).gross={gW}; strat(s).turnover={tW}; strat(s).nS=1;
    else
        strat(s).gross=cell(nSeeds,1); strat(s).turnover=cell(nSeeds,1); strat(s).nS=nSeeds;
        for si=1:nSeeds
            af=fullfile(strat(s).logRoot,sprintf("seed_%d",seeds(si)),"TrainedAgent.mat");
            L=load(af,"agent");
            [gW,tW]=replayAll(L.agent,strat(s).kind,strat(s).basis,[],nWinTotal, ...
                evalStepSize,horizonPeriods,R_full,M_test_z,initialWealth,goalWealth);
            strat(s).gross{si}=gW; strat(s).turnover{si}=tW;
        end
    end
    fprintf("Replayed %s.\n", strat(s).name);
end

%% Report val + test at the headline 10bp, and a cost sweep
ci10 = find(costBps==10,1);
fprintf("\n=== SELECTION on VAL (first %d windows) @10bp ===\n", numel(valIdx));
valSharpe=nan(nStrat,1);
for s=1:nStrat
    m=metricSet(strat(s),valIdx,costBps(ci10)/1e4,initialWealth,goalWealth, ...
        annualizeFactor,blockLen,nBoot,dsrNumTrials,dsrTrialSharpeVar);
    valSharpe(s)=m.sharpe;
    fprintf("  %-9s | valSharpe %5.2f [%5.2f,%5.2f]\n",strat(s).name,m.sharpe,m.lo,m.hi);
end
b6040 = find(strcmp({strat.name},'60/40'));
rlIdx = setdiff(1:nStrat,b6040);
edge = valSharpe(rlIdx) - valSharpe(b6040);
[~,bi] = max(edge); winner = rlIdx(bi);
fprintf("  --> VAL winner (max edge over 60/40): %s (edge %+.2f)\n", strat(winner).name, edge(bi));

fprintf("\n=== LOCKED TEST (last %d windows) — full cost sweep ===\n", numel(testIdx));
for ci=1:nCost
    c=costBps(ci)/1e4;
    fprintf("--- cost %d bp ---\n", costBps(ci));
    fprintf("%-9s | netSharpe (95%% CI)      | MaxDD P90 | TermP10 | turn | succ | DSR\n","strat");
    for s=1:nStrat
        m=metricSet(strat(s),testIdx,c,initialWealth,goalWealth, ...
            annualizeFactor,blockLen,nBoot,dsrNumTrials,dsrTrialSharpeVar);
        star=""; if s==winner, star=" *"; end
        fprintf("%-9s | %5.2f [%5.2f,%5.2f] | %7.2f%% | %7.0f | %.3f | %4.1f | %.2f%s\n", ...
            strat(s).name,m.sharpe,m.lo,m.hi,m.p90*100,m.p10,m.turn,m.succ,m.dsr,star);
    end
    fprintf("\n");
end

% Ground-truth verdict on locked test @10bp
mW=metricSet(strat(winner),testIdx,costBps(ci10)/1e4,initialWealth,goalWealth, ...
    annualizeFactor,blockLen,nBoot,dsrNumTrials,dsrTrialSharpeVar);
mB=metricSet(strat(b6040),testIdx,costBps(ci10)/1e4,initialWealth,goalWealth, ...
    annualizeFactor,blockLen,nBoot,dsrNumTrials,dsrTrialSharpeVar);
fprintf("=== GROUND-TRUTH CHECK @10bp (locked test) ===\n");
fprintf("  winner %s: Sharpe %.2f [%.2f, %.2f] | 60/40: %.2f [%.2f, %.2f]\n", ...
    strat(winner).name,mW.sharpe,mW.lo,mW.hi,mB.sharpe,mB.lo,mB.hi);
beats = mW.lo > mB.hi;   % non-overlapping CIs = a (crude) significance bar
fprintf("  Significantly beats 60/40 (winner CI-low > 60/40 CI-high)? %s\n", string(beats));
if ~beats
    fprintf("  -> NOT significant. Honest: %d test windows lack the power to separate\n", numel(testIdx));
    fprintf("     the winner from 60/40. Ground truth NOT met on data/power grounds.\n");
end

save(fullfile("experiments","logs","cql_valtest_round1.mat"),"strat","valIdx","testIdx","winner");

%% ---- Local functions ----
function [gAll,tAll]=replayAll(agent,kind,basis,staticW,nWin,step,H,R_full,M_test_z,iW,gW_)
    gAll=nan(nWin,H); tAll=nan(nWin,H);
    for ep=1:nWin
        s0=(ep-1)*step+1; if s0+H-1>size(R_full,1), break; end
        [g,t]=runWindow(agent,kind,basis,staticW,s0,H,R_full,M_test_z,iW,gW_);
        gAll(ep,:)=g'; tAll(ep,:)=t';
    end
end

function m=metricSet(st,winIdx,c,iW,gW,ann,blockLen,nBoot,dsrN,dsrVar)
    % Honest estimator: per-seed pooled-daily Sharpe / CI / DSR, averaged.
    poolSharpe=@(r) mean(r)/std(r)*ann;
    nS=st.nS;
    shp=nan(nS,1); lo=nan(nS,1); hi=nan(nS,1); dsr=nan(nS,1);
    p90=nan(nS,1); p10=nan(nS,1); suc=nan(nS,1); tn=nan(nS,1);
    for si=1:nS
        g=st.gross{si}(winIdx,:); t=st.turnover{si}(winIdx,:);
        [~,a90,a10,asu,atn,netRets]=windowStats(g,t,c,iW,gW,ann);
        shp(si)=poolSharpe(netRets);
        [l,h]=blockBootstrapSharpeCI(netRets,blockLen,nBoot,ann); lo(si)=l; hi(si)=h;
        dsr(si)=deflatedSharpe(netRets,dsrN,dsrVar);
        p90(si)=a90; p10(si)=a10; suc(si)=asu; tn(si)=atn;
    end
    m.sharpe=mean(shp,'omitnan'); m.lo=mean(lo,'omitnan'); m.hi=mean(hi,'omitnan');
    m.dsr=mean(dsr,'omitnan'); m.p90=mean(p90,'omitnan'); m.p10=mean(p10,'omitnan');
    m.succ=mean(suc,'omitnan'); m.turn=mean(tn,'omitnan');
end

function [grossRet,turnover]=runWindow(agent,kind,basis,staticW,startEval,H,R_full,M_test_z,iW,gW)
    normW=@(w) min(w/gW,5); tf=@(t) t/H;
    grossRet=zeros(H,1); turnover=zeros(H,1); wealth=iW; wPrev=[];
    for t=1:H
        d=startEval+t-1;
        if isempty(agent)
            w=staticW(:);
        else
            obs={[normW(wealth);tf(t);M_test_z(d,:)']}; aOut=agent.getAction(obs);
            switch kind
                case 'dqn',  w=basis(:,aOut{1});
                case 'ddpg', w=alphaToWeights(max(0,min(1,double(aOut{1}))),basis);
            end
            w=w(:);
        end
        rA=R_full(d,:)'; rG=R_full(d,:)*w; grossRet(t)=rG;
        if t==1, turnover(t)=0; else, turnover(t)=sum(abs(w-wPrev)); end
        wPrev=(w.*(1+rA))/(1+rG); wealth=wealth*(1+rG);
    end
end

function [meanSharpe,p90,p10,succRate,meanTurn,pooledNetRet]=windowStats(gross,turnover,c,iW,gW,ann)
    nWin=size(gross,1); sh=nan(nWin,1); dd=nan(nWin,1); tw=nan(nWin,1); su=nan(nWin,1); tu=nan(nWin,1);
    pooledNetRet=[];
    for ep=1:nWin
        g=gross(ep,:)'; t=turnover(ep,:)'; if any(isnan(g)), continue; end
        nr=g-c*t; wl=iW*[1;cumprod(1+nr)]; ln=diff(log(wl));
        sh(ep)=mean(ln)/std(ln)*ann; dd(ep)=max(cummax(wl)-wl)/max(wl);
        tw(ep)=wl(end); su(ep)=double(wl(end)>=gW); tu(ep)=mean(t);
        pooledNetRet=[pooledNetRet; ln]; %#ok<AGROW>
    end
    meanSharpe=mean(sh,'omitnan'); p90=prctile(dd,90); p10=prctile(tw,10);
    succRate=sum(su,'omitnan'); meanTurn=mean(tu,'omitnan');
end

function w=alphaToWeights(alpha,W)
    alpha=max(0,min(1,alpha)); n=size(W,2); idx=1+alpha*(n-1);
    lo=max(1,floor(idx)); hi=min(n,lo+1); f=idx-lo; w=(1-f)*W(:,lo)+f*W(:,hi);
end
