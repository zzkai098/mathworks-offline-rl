%% GBWM P(goal) — PRIMARY DELIVERABLE: gap-to-DP vs dataset-size curve
% For each offline dataset size, each seed, each method (vanilla / CQL / IQL):
% log episodes under a SUBOPTIMAL random-over-frontier behavior policy, train
% offline, evaluate P(goal) by MC under the true regime model, and record
% gap-to-DP = dp_ceiling - P(goal). Aggregate mean +/- across seeds -> the
% sample-efficiency curve (does offline pessimism recover more of the DP optimum
% from less logged data?).
%
% Non-leakage discipline: behavior policy, model, goal all fit on train only;
% cqlAlpha / iqlTau are FIXED literature defaults, NOT tuned on the evaluator.
%
% Run from repo root:
%   matlab -batch "addpath('gbwm_pgoal'); addpath('gbwm_pgoal/offline'); addpath('gbwm_pgoal/experiments'); sweep"

clear; clc;
SIZES   = [50 100 200 500 1000 2000 5000];     % episodes (pre-registered grid)
SEEDS   = [1000 2000 3000 4000 5000];
METHODS = ["vanilla" "cql" "iql"];
STEPS   = 5000;  NMC = 100000;
BEHAVIOR = "random";

M = gbwm.model(true);
ceiling = M.dp_ceiling;

nS = numel(SIZES); nSe = numel(SEEDS); nMe = numel(METHODS);
Pgoal = nan(nS, nSe, nMe);
tStart = tic;
for i = 1:nS
    for j = 1:nSe
        d = gbwm.logEpisodes(M, BEHAVIOR, SIZES(i), SEEDS(j));
        for k = 1:nMe
            opts = struct('seed', SEEDS(j), 'steps', STEPS);
            pf = train_offline(M, d, METHODS(k), opts);
            Pgoal(i,j,k) = gbwm.mcEval(M, pf, M.goal, NMC, 7);
        end
        fprintf("size=%4d seed=%d done (%.0fs)  [v %.3f | c %.3f | i %.3f]\n", ...
            SIZES(i), SEEDS(j), toc(tStart), ...
            Pgoal(i,j,1), Pgoal(i,j,2), Pgoal(i,j,3));
    end
end

gap = ceiling - Pgoal;                     % gap-to-DP
gapMean = squeeze(mean(gap, 2));           % nS x nMe
gapStd  = squeeze(std(gap, 0, 2));

fprintf("\n===== gap-to-DP (mean +/- std across %d seeds), ceiling=%.3f =====\n", nSe, ceiling);
fprintf("%-6s | %-16s | %-16s | %-16s\n", "size", "vanilla", "CQL", "IQL");
for i = 1:nS
    fprintf("%-6d | %.3f +/- %.3f    | %.3f +/- %.3f    | %.3f +/- %.3f\n", SIZES(i), ...
        gapMean(i,1), gapStd(i,1), gapMean(i,2), gapStd(i,2), gapMean(i,3), gapStd(i,3));
end

if ~isfolder("gbwm_pgoal/experiments/out"), mkdir("gbwm_pgoal/experiments/out"); end
save("gbwm_pgoal/experiments/out/sweep.mat", "SIZES","SEEDS","METHODS","Pgoal","gap","ceiling");

% ---- figure: gap-to-DP vs dataset size, CI bands ----
figure('Position',[100 100 640 460]); hold on;
cols = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];  % vanilla/cql/iql
for k = 1:nMe
    mu = gapMean(:,k)'; sd = gapStd(:,k)'/sqrt(nSe);       % SE band
    fill([SIZES fliplr(SIZES)], [mu-sd fliplr(mu+sd)], cols(k,:), ...
        'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
    plot(SIZES, mu, '-o', 'Color', cols(k,:), 'LineWidth', 1.8, 'DisplayName', METHODS(k));
end
set(gca,'XScale','log'); grid on;
xlabel('offline dataset size (episodes)'); ylabel('gap-to-DP on P(goal)');
title(sprintf('Sample efficiency: gap to DP optimum (ceiling %.3f)', ceiling));
legend('Location','northeast');
saveas(gcf, "gbwm_pgoal/experiments/out/gap_to_dp.png");
fprintf("\nsaved out/sweep.mat + out/gap_to_dp.png\n");
