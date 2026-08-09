%% CQL alpha-sensitivity — give CQL a FAIR shot (not a strawman)
% The main sweep used cqlAlpha=1, which for P(goal) rewards in [0,1] is ~30-100x
% too large (the CQL penalty logsumexp over 15 actions ~ log15 ~ 2.7 dwarfs the
% TD loss ~ 0.01-0.1). CQL's penalty strength must be scaled to the reward
% magnitude. Sweep alpha on a log grid at representative dataset sizes; compare
% against the vanilla / IQL reference lines from the main sweep.
%
% Non-leakage: alpha grid is pre-registered here, chosen by REWARD-SCALE
% reasoning (not by peeking at the evaluator).
%
% Run: matlab -batch "addpath('gbwm_pgoal'); addpath('gbwm_pgoal/offline'); addpath('gbwm_pgoal/experiments'); cql_alpha_sweep"

clear; clc;
SIZES  = [100 500 2000];
ALPHAS = [0.01 0.03 0.1 0.3 1.0];
SEEDS  = [1000 2000 3000 4000 5000];
STEPS  = 5000;  NMC = 100000;

M = gbwm.model(true); ceiling = M.dp_ceiling;
nS = numel(SIZES); nA = numel(ALPHAS); nSe = numel(SEEDS);
Pgoal = nan(nS, nA, nSe);
t0 = tic;
for i = 1:nS
    for j = 1:nSe
        d = gbwm.logEpisodes(M, "random", SIZES(i), SEEDS(j));
        for k = 1:nA
            opts = struct('seed', SEEDS(j), 'steps', STEPS, 'cqlAlpha', ALPHAS(k));
            pf = train_offline(M, d, "cql", opts);
            Pgoal(i,j,k) = gbwm.mcEval(M, pf, M.goal, NMC, 7); %#ok<*SAGROW>
        end
        fprintf("size=%4d seed=%d (%.0fs)\n", SIZES(i), SEEDS(j), toc(t0));
    end
end
gap = ceiling - Pgoal;                    % nS x nA x nSe... careful dims
% reorder to nS x nSe x nA for mean over seeds
gap = permute(ceiling - permute(Pgoal,[1 3 2]), [1 3 2]);  % nS x nA x nSe
gapMean = mean(gap, 3);                    % nS x nA
gapStd  = std(gap, 0, 3);

fprintf("\n===== CQL gap-to-DP by alpha (mean over %d seeds), ceiling=%.3f =====\n", nSe, ceiling);
fprintf("%-6s", "size"); fprintf(" | a=%-5.2f", ALPHAS); fprintf("\n");
for i = 1:nS
    fprintf("%-6d", SIZES(i)); fprintf(" | %.3f  ", gapMean(i,:)); fprintf("\n");
end

% reference: vanilla / IQL at these sizes from the main sweep
ref = load("gbwm_pgoal/experiments/out/sweep.mat");
[~, si] = ismember(SIZES, ref.SIZES);
gv = squeeze(mean(ref.gap(si,:,1),2)); gi = squeeze(mean(ref.gap(si,:,3),2));
fprintf("ref vanilla: "); fprintf("%.3f ", gv); fprintf("\nref IQL:     "); fprintf("%.3f ", gi); fprintf("\n");

save("gbwm_pgoal/experiments/out/cql_alpha.mat","SIZES","ALPHAS","SEEDS","Pgoal","gap","ceiling");

figure('Position',[100 100 640 460]); hold on;
cmap = parula(nS);
for i = 1:nS
    errorbar(ALPHAS, gapMean(i,:), gapStd(i,:)/sqrt(nSe), '-o', 'Color', cmap(i,:), ...
        'LineWidth', 1.6, 'DisplayName', sprintf('CQL size=%d', SIZES(i)));
    yline(gv(i), '--', 'Color', cmap(i,:), 'HandleVisibility','off');
    yline(gi(i), ':',  'Color', cmap(i,:), 'HandleVisibility','off');
end
set(gca,'XScale','log'); grid on;
xlabel('CQL \alpha'); ylabel('gap-to-DP on P(goal)');
title('CQL \alpha-sensitivity (dashed=vanilla, dotted=IQL, same color per size)');
legend('Location','northwest');
saveas(gcf, "gbwm_pgoal/experiments/out/cql_alpha.png");
fprintf("saved out/cql_alpha.mat + out/cql_alpha.png\n");
