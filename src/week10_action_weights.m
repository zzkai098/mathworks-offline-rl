%% What each of the 15 actions IS — asset-weight composition of every frontier action
% Each discrete action = one efficient-frontier portfolio = a full weight vector over
% all assets (not a single asset). This stacked bar shows, for each action 1..15
% (1=defensive .. 15=aggressive), how the ~15 assets are mixed. Same frontier basis the
% shipped agents were trained/evaluated on (extended train 2010-2021, 15 actions).
clear; clc;
addpath("src"); addpath(fullfile("src","utils"));

basis = buildEvalBasis();                 % gives W_frontier (nAssets x 15) + assetNames
W  = basis.W_frontier;                     % columns = actions, rows = assets
an = basis.assetNames{1};                   % asset names
nAssets = basis.nAssets;  numActions = basis.numActions;

f = figure('Visible','off','Position',[100 100 1150 560]);
b = bar(1:numActions, W', 'stacked', 'EdgeColor',[.2 .2 .2], 'LineWidth',0.2);
cmap = turbo(nAssets);
for k = 1:nAssets; b(k).FaceColor = cmap(k,:); end
xlim([0.5 numActions+0.5]); ylim([0 1]); xticks(1:numActions);
xlabel('action (frontier portfolio index:  1 = defensive .. 15 = aggressive)');
ylabel('portfolio weight');
title('What each action IS — asset composition of the 15 frontier portfolios');
legend(an, 'Location','eastoutside', 'Interpreter','none', 'FontSize',8);
if ~isfolder("experiments/figures"); mkdir("experiments/figures"); end
exportgraphics(f, "experiments/figures/action_weight_composition.png", 'Resolution',150);
fprintf("saved -> experiments/figures/action_weight_composition.png\n");

% print the top-2 holdings of each action for a quick text read
fprintf("\naction | top holdings\n");
for a = 1:numActions
    [ws, ord] = sort(W(:,a), 'descend');
    fprintf("  %2d   | %s %.0f%%,  %s %.0f%%,  %s %.0f%%\n", a, ...
        an{ord(1)}, 100*ws(1), an{ord(2)}, 100*ws(2), an{ord(3)}, 100*ws(3));
end
