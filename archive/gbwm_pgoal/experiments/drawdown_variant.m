%% Drawdown-shaping variant (mentor's regime-gated drawdown penalty)
% The primary study used a pure P(goal) reward. Here we add the mentor's Week-6.1
% regime-gated drawdown penalty (beta_stress=8/beta_normal=2, DD_THR=0.03) to the
% per-step reward and ask: does the shaping buy BETTER TAIL (lower path MaxDD)
% than the tail-blind DP optimum and the pure-P(goal) RL, and at what P(goal)
% cost? Objective now = P(goal)+tail, so we report P(goal) AND MaxDD together
% (this is NOT the pure gap-to-DP comparison).
%
% References (tail-blind): DP optimum, best-static, 60/40.
%
% Run: matlab -batch "addpath('gbwm_pgoal'); addpath('gbwm_pgoal/offline'); addpath('gbwm_pgoal/experiments'); drawdown_variant"

clear; clc;
SIZE   = 2000;
SEEDS  = [1000 2000 3000 4000 5000];
METHODS = ["vanilla" "cql" "iql"];
CQL_ALPHA = 0.01;                 % reward-scaled (from alpha-sensitivity)
STEPS = 5000; NMC = 100000;

M = gbwm.model(true);

% --- reference (tail-blind) policies: DP, best-static, 60/40 ---
dpS = gbwm.mcEvalFull(M, M.dp_policyFn, M.goal, NMC, 7);
% best static (recompute quickly via P(goal) over frontier, reuse #15-ish)
bestA = 0; bestP = -1;
for a = 1:M.NFRONT
    p = gbwm.mcEval(M, @(W,t,rg) repmat(a,size(W)), M.goal, 50000, 7);
    if p > bestP, bestP = p; bestA = a; end
end
stS = gbwm.mcEvalFull(M, @(W,t,rg) repmat(bestA,size(W)), M.goal, NMC, 7);
sixS = gbwm.mcEvalFull(M, @(W,t,rg) repmat(M.I6040,size(W)), M.goal, NMC, 7);

fprintf("\n--- tail-blind references (P(goal) | meanMaxDD | p90MaxDD | termP10) ---\n");
fprintf("  DP optimum   : %.3f | %.3f | %.3f | %.0f\n", dpS.pGoal, dpS.meanMaxDD, dpS.p90MaxDD, dpS.termP10);
fprintf("  best static  : %.3f | %.3f | %.3f | %.0f\n", stS.pGoal, stS.meanMaxDD, stS.p90MaxDD, stS.termP10);
fprintf("  60/40        : %.3f | %.3f | %.3f | %.0f\n", sixS.pGoal, sixS.meanMaxDD, sixS.p90MaxDD, sixS.termP10);

% --- RL: pure vs drawdown-shaped, per method, averaged over seeds ---
res = struct();
for k = 1:numel(METHODS)
    meth = METHODS(k);
    for v = [false true]
        pg = zeros(numel(SEEDS),1); mdd = zeros(numel(SEEDS),1);
        p90 = zeros(numel(SEEDS),1); tp10 = zeros(numel(SEEDS),1);
        for j = 1:numel(SEEDS)
            d = gbwm.logEpisodes(M, "random", SIZE, SEEDS(j), v);
            opts = struct('seed', SEEDS(j), 'steps', STEPS, 'cqlAlpha', CQL_ALPHA);
            pf = train_offline(M, d, meth, opts);
            s = gbwm.mcEvalFull(M, pf, M.goal, NMC, 7);
            pg(j)=s.pGoal; mdd(j)=s.meanMaxDD; p90(j)=s.p90MaxDD; tp10(j)=s.termP10;
        end
        tag = "pure"; if v, tag = "shaped"; end
        res.(meth+"_"+tag) = [mean(pg) mean(mdd) mean(p90) mean(tp10)];
        fprintf("  %-8s %-6s : P=%.3f | MaxDD %.3f | p90DD %.3f | termP10 %.0f\n", ...
            meth, tag, mean(pg), mean(mdd), mean(p90), mean(tp10));
    end
end

save("gbwm_pgoal/experiments/out/drawdown_variant.mat", "res", "dpS", "stS", "sixS", "SIZE", "SEEDS");
fprintf("\nsaved out/drawdown_variant.mat\n");
fprintf("Story check: does 'shaped' lower MaxDD vs 'pure' (and vs DP %.3f), and at what P(goal) cost?\n", dpS.meanMaxDD);
