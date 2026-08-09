%% Final agent — action POLICY MAP over (wealth x time), N=10 seed ensemble
% GBWM-style policy heatmap (cf. the MathWorks demo's "Action" chart): for a grid of
% wealth levels x time steps, show the greedy action the N=10 tuned-DQN ensemble selects
% (average Q over seeds 1000..10000, argmax). Because the state is 6-D (macro-augmented),
% a 2-D map must FIX the macro block, so we draw two panels — NORMAL (macro=0) vs a
% representative STRESS regime (VIX_z=+2, T10Y2Y_z=-2) — to expose regime conditioning.
clear; clc;
addpath("src"); addpath(fullfile("src","utils"));

goalWealth = 102000;  H = 30;  numActions = 15;  ensSeeds = 1000:1000:10000;

% frontier only needed for the config; load the 10 ensemble critics
critics = cell(numel(ensSeeds),1);
for i = 1:numel(ensSeeds)
    Ld = load(sprintf("experiments/models/dqn_seed_%d.mat", ensSeeds(i)), "FA");
    assert(contains(Ld.FA.meta.learner,"tuned-DQN"), "seed %d not tuned-DQN", ensSeeds(i));
    critics{i} = getCritic(Ld.FA.agent);
end

% wealth grid (log-spaced, 0.5x .. 5x goal — 5x is the training clip)
nW = 24;
wealthGrid = goalWealth * logspace(log10(0.5), log10(5), nW)';   % ascending
normW = min(wealthGrid/goalWealth, 5);

% two macro regimes (cols: 1 DGS10, 2 T10Y2Y, 3 VIX, 4 DFF)
regimes = { 'NORMAL (macro=0)',        [0 0 0 0];
            'STRESS (VIX+2, slope-2)', [0 -2 2 0] };

f = figure('Visible','off','Position',[100 100 1250 560]);
for rr = 1:2
    macro = regimes{rr,2};
    A = nan(nW, H);
    for it = 1:H
        for iw = 1:nW
            o = [normW(iw); it/H; macro'];
            q = 0; for m = 1:numel(critics); v = getValue(critics{m},{o}); q = q + v(:); end
            [~, A(iw,it)] = max(q);
        end
    end
    subplot(1,2,rr);
    imagesc(1:H, 1:nW, A); set(gca,'YDir','reverse');   % low wealth at top (cf. demo)
    caxis([1 numActions]); colormap(gca, parula(numActions));
    cb = colorbar; cb.Label.String = 'greedy action (1=defensive .. 15=aggressive)';
    xlabel('time step (of 30-day horizon)'); ylabel('wealth level');
    yticks(1:2:nW); yticklabels(compose('%.0f', wealthGrid(1:2:nW)));
    yline(find(wealthGrid>=goalWealth,1)-0.5, 'r-', 'goal', 'LineWidth',1.2, 'FontSize',8);
    title(sprintf('Action policy map — %s', regimes{rr,1}));
end
sgtitle('Final agent (tuned-DQN, N=10 ensemble) — action over (wealth \times time)');
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f, "experiments/figures/final_agent_action_policymap.png", 'Resolution',150);
fprintf("saved -> experiments/figures/final_agent_action_policymap.png\n");

% summary: mean action by regime (over the whole grid)
for rr = 1:2
    macro = regimes{rr,2};  tot = 0; cnt = 0;
    for it = 1:H; for iw = 1:nW
        o = [normW(iw); it/H; macro'];
        q = 0; for m = 1:numel(critics); v = getValue(critics{m},{o}); q = q + v(:); end
        [~,a] = max(q);  tot = tot + a;  cnt = cnt + 1;
    end; end
    fprintf("mean action over grid — %-24s = %.2f\n", regimes{rr,1}, tot/cnt);
end
