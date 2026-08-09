%% Final agent — action-over-time timeline (shipped tuned-DQN, seed 1000)
% Replays the shipped greedy policy over the chained test windows and plots the
% selected efficient-frontier action index (1 = most defensive .. 15 = most aggressive)
% against the test-day axis, with market-stress days shaded, to visualize whether the
% deliverable agent conditions its allocation on regime. Reuses the reviewed eval basis.
% Note: wealth resets at each 30-day window boundary (episodic eval), so the action
% path is regime/time-driven, not a single compounding trajectory.
clear; clc;
addpath("src"); addpath(fullfile("src","utils"));

basis = buildEvalBasis();                          % extended train 2010-2021, 15 actions
L = load("experiments/models/dqn_seed_1000.mat","FA");
assert(contains(L.FA.meta.learner,"tuned-DQN"), "unexpected agent config: %s", L.FA.meta.learner);
critic = getCritic(L.FA.agent);

R_full = basis.R_full;  M = basis.M_test_z;  Wf = basis.W_frontier;
iw = basis.initialWealth;  gw = basis.goalWealth;
H  = basis.horizonPeriods; nEp = basis.numEvalEpisodes;  step = basis.evalStepSize;

day = [];  act = [];
for ep = 1:nEp
    s0 = (ep-1)*step + 1;  if s0 + H - 1 > size(R_full,1); break; end
    wealth = iw;
    for t = 1:H
        d = s0 + t - 1;
        o = [min(wealth/gw,5); t/H; M(d,:)'];
        q = getValue(critic,{o});  [~,a] = max(q(:));
        rG = R_full(d,:)*Wf(:,a);
        day(end+1) = d;  act(end+1) = a; %#ok<AGROW>
        wealth = wealth*(1+rG);
    end
end

% stress flag on test days (macro cols: 1 DGS10, 2 T10Y2Y, 3 VIX, 4 DFF)
isStress = M(:,3) > 1.5 | M(:,2) < -1.5;

f = figure('Visible','off','Position',[100 100 1150 470]); hold on; grid on;
yl = [0 16];
% shade stress spans (merge consecutive days into patches)
dd = [false; isStress(:); false];  ch = diff(dd);
ons = find(ch==1);  offs = find(ch==-1) - 1;
for k = 1:numel(ons)
    patch([ons(k)-0.5 offs(k)+0.5 offs(k)+0.5 ons(k)-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
          [1 0.85 0.85], 'EdgeColor','none', 'HandleVisibility','off');
end
stairs(day, act, '-', 'LineWidth',1.5, 'Color',[0.10 0.40 0.80], 'DisplayName','greedy action idx');
ylim(yl); yticks(1:15);
xlabel('test trading day'); ylabel('frontier action (1=defensive .. 15=aggressive)');
title('Final agent (tuned-DQN, seed 1000) — action over time');
subtitle('pink = market-stress days (VIX\_z>1.5 OR T10Y2Y\_z<-1.5); wealth resets each 30-day window');
legend('Location','best');
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f, "experiments/figures/final_agent_action_timeline.png", 'Resolution',150);
fprintf("saved -> experiments/figures/final_agent_action_timeline.png\n");

% quick summary
fprintf("\naction usage (idx:count):\n");
for a = 1:15; n = sum(act==a); if n>0; fprintf("  %2d: %d\n", a, n); end; end
sMask = isStress(day);
fprintf("mean action  stress=%.2f  normal=%.2f  (higher = more aggressive)\n", ...
    mean(act(sMask)), mean(act(~sMask)));
