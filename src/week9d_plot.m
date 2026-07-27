%% A/B plot: week9c (periodic-every-4, diverges) vs week9d (Polyak slow target)
C = readtable('experiments/logs/week9c_results.csv');
[~,ic] = unique(C(:,{'seed','ME'}),'rows','stable'); C = C(ic,:);
D = readtable('experiments/logs/week9d_results.csv');
[~,id] = unique(D(:,{'tsf','seed','ME'}),'rows','stable'); D = D(id,:);
seeds = [1000 2000 3000];

f = figure('Visible','off','Position',[100 100 1250 480]);

% Panel 1: Q divergence A/B (log-y)
subplot(1,2,1); hold on;
hC=[]; hD=[];
MEg = [50 100 200 400];
hC=[]; hD=[]; hM=[];
for s = seeds
    c = sortrows(C(C.seed==s,:),'ME');
    hC = semilogy(c.ME, c.qGlobal, '--o','Color',[0.85 0.2 0.2],'LineWidth',1.5);
    d = D(D.tsf==1e-3 & D.seed==s,:); d = sortrows(d,'ME');
    hD = semilogy(d.ME, d.qGlobal, '-o','Color',[0.1 0.5 0.85],'LineWidth',1.6);
    % full fix = Polyak tau=1e-3 + LR 1e-4 (from the week10 matrix DQN cells)
    qm = nan(1,4);
    for k=1:4
        fm = sprintf('experiments/logs/matrix/dqn_s%d_me%d.csv', s, MEg(k));
        if isfile(fm); r=readmatrix(fm); qm(k)=r(5); end
    end
    hM = semilogy(MEg, qm, '-s','Color',[0.1 0.6 0.2],'LineWidth',2.2);
end
set(gca,'YScale','log');
yline(1.2,':k','economic ceiling ~1.2','LineWidth',1.2);
xlabel('training epochs (MaxEpochs)'); ylabel('val mean max_a Q  [log]');
title('Target update + LR: partial vs full fix of the divergence');
legend([hC hD hM], {'periodic-every-4 (6.1: diverges)', ...
    'Polyak \tau=1e-3 (bounds ME\leq200, residual at 400)', ...
    'Polyak \tau=1e-3 + LR 1e-4 (full fix: flat)'}, 'Location','northwest');
grid on;

% Panel 2: across-seed spread of Q at each ME (min-max band), A/B
subplot(1,2,2); hold on;
MEs = [50 100 200 400];
qC = nan(numel(MEs),3); qD = nan(numel(MEs),3);
for i=1:numel(MEs)
    for j=1:3
        vc = C.qGlobal(C.seed==seeds(j) & C.ME==MEs(i)); if ~isempty(vc); qC(i,j)=vc(1); end
        vd = D.qGlobal(D.tsf==1e-3 & D.seed==seeds(j) & D.ME==MEs(i)); if ~isempty(vd); qD(i,j)=vd(1); end
    end
end
plot(MEs, max(qC,[],2)-min(qC,[],2), '--s','Color',[0.85 0.2 0.2],'LineWidth',1.5,'DisplayName','week9c across-seed Q spread');
plot(MEs, max(qD,[],2)-min(qD,[],2), '-s','Color',[0.1 0.5 0.85],'LineWidth',2,'DisplayName','week9d \tau=1e-3 spread');
set(gca,'YScale','log');
xlabel('training epochs (MaxEpochs)'); ylabel('across-seed Q spread (max-min)  [log]');
title('Across-seed variance collapses too (ME\leq200)');
legend('Location','northwest'); grid on;

exportgraphics(f,'experiments/figures/week9d_ab_slowtarget.png','Resolution',150);
fprintf('saved -> experiments/figures/week9d_ab_slowtarget.png\n');

% summary table
fprintf('\nQ by ME (week9d tau=1e-3  vs  week9c):\n');
fprintf('ME  | 9d:s1000 s2000 s3000 |  9c:s1000 s2000 s3000\n');
for i=1:numel(MEs)
    fprintf('%3d | %7.2f %6.2f %6.2f | %8.1f %6.1f %6.1f\n', MEs(i), qD(i,1),qD(i,2),qD(i,3), qC(i,1),qC(i,2),qC(i,3));
end
