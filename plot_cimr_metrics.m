% plot_cimr_metrics.m
% ===================
% Refinement-dynamics figures for breadth-depth refinement (N=8, D=4),
% sized for a double-column paper (full text width 17.8 cm).
%
% Inputs: the three CSVs written by export_paper_csvs.py
%   experiments_results/paper_csvs/depth_accuracy_metrics.csv
%   experiments_results/paper_csvs/correction_effectiveness.csv
%   experiments_results/paper_csvs/diversity.csv
%
% Outputs (vector PDF) in experiments_results/paper_figures/
%   fig1_recovery_regression.pdf      recovery rho_rec(d) and regression rho_reg(d)
%   fig2_correction_effectiveness.pdf answer change mu_Delta(d), net benefit B(d),
%                                     W->C and C->W flip rates
%   fig3_diversity.pdf                terminal rollout diversity and agreement U(d)
%   fig4_heatmaps.pdf                 per-dataset recovery and net benefit (d >= 2)
%
% Requires MATLAB R2020b+ (tiledlayout legends; exportgraphics from R2020a).
% -------------------------------------------------------------------------

clear; clc; close all;

% ── Global LaTeX rendering ────────────────────────────────────────────────────
set(groot, 'defaultAxesTickLabelInterpreter',    'latex');
set(groot, 'defaultTextInterpreter',              'latex');
set(groot, 'defaultLegendInterpreter',            'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

% ── Paths ─────────────────────────────────────────────────────────────────────
csv_dir = fullfile('experiments_results', 'paper_csvs');
fig_dir = fullfile('experiments_results', 'paper_figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% ── Load CSVs ─────────────────────────────────────────────────────────────────
T_acc  = readtable(fullfile(csv_dir, 'depth_accuracy_metrics.csv'));
T_corr = readtable(fullfile(csv_dir, 'correction_effectiveness.csv'));
T_div  = readtable(fullfile(csv_dir, 'diversity.csv'));

T_acc.model    = string(T_acc.model);
T_acc.dataset  = string(T_acc.dataset);
T_corr.model   = string(T_corr.model);
T_corr.dataset = string(T_corr.dataset);
T_div.model    = string(T_div.model);
T_div.dataset  = string(T_div.dataset);

% ── Models, styles ────────────────────────────────────────────────────────────
model_keys = ["LLaMA-3.1-8B", "Ministral-8B", "Qwen2.5-1.5B", "Qwen2.5-Math-7B"];
model_tex  = {'LLaMA-3.1-8B', 'Ministral-8B', 'Qwen2.5-1.5B', 'Qwen2.5-Math-7B'};

% Colorblind-safe palette (Wong 2011)
colors = [
    0.00,  0.45,  0.70;   % blue          – LLaMA
    0.90,  0.62,  0.00;   % orange        – Ministral
    0.00,  0.62,  0.45;   % bluish-green  – Qwen-1.5B
    0.80,  0.47,  0.65;   % reddish-purple– Qwen-Math-7B
];
markers = {'o', 's', '^', 'd'};

% Font sizes tuned for double-column papers (body ≈ 9–10 pt)
lw        = 1.8;   % line width
msz       = 6;     % marker size
fsz_tick  = 8;     % axis tick labels
fsz_label = 9;     % x/y axis labels
fsz_title = 9;     % panel title
fsz_leg   = 8;     % legend entries

% Double-column figure widths (full text width = 17.8 cm)
FIG_W     = 17.6;  % cm — full-width figure
FIG_H_2   = 6.0;   % cm — height for 1×2 layout (panels + legend)
FIG_H_4   = 6.0;   % cm — height for 1×4 layout (panels + legend)

nM = numel(model_keys);
depths   = 1:4;
datasets = unique(T_acc.dataset);
nDS      = numel(datasets);

% =============================================================================
%  FIGURE 1 — Recovery Rate and Regression Rate  (1×2, shared top legend)
% =============================================================================
fig1 = figure('Name','Recovery & Regression', ...
    'Units','centimeters','Position',[1 1 FIG_W FIG_H_2]);

tl1 = tiledlayout(fig1, 1, 2, ...
    'Padding','compact','TileSpacing','compact');

panel_cols   = {'recovery_rate',   'regression_rate'};
panel_ylabs  = {'Recovery Rate',   'Regression Rate'};
panel_titles = {'\textbf{(a)} Questions Recovered per Depth', ...
                '\textbf{(b)} Questions Lost per Depth'};

h_leg1 = gobjects(nM, 1);   % keep handles for the shared legend
for p = 1:2
    ax = nexttile(tl1);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    for m = 1:nM
        y = avg_depth(T_acc, model_keys(m), panel_cols{p}, depths);
        hh = plot(ax, depths, y, ['-', markers{m}], ...
            'Color', colors(m,:), 'LineWidth', lw, ...
            'MarkerSize', msz, 'MarkerFaceColor', colors(m,:));
        if p == 1, h_leg1(m) = hh; end   % capture from first panel
    end
    xlabel(ax, 'Refinement Depth $d$', 'Interpreter','latex','FontSize',fsz_label);
    ylabel(ax, panel_ylabs{p},          'Interpreter','latex','FontSize',fsz_label);
    title(ax,  panel_titles{p},         'Interpreter','latex','FontSize',fsz_title);
    xticks(ax, depths); xlim(ax,[0.8 4.2]);
    ylim(ax,[0, max(ax.YLim(2), 0.06)]);
    set(ax,'FontSize',fsz_tick,'TickLabelInterpreter','latex');
end

% Single horizontal legend above all panels
lgd1 = legend(h_leg1, model_tex, ...
    'Interpreter','latex','FontSize',fsz_leg, ...
    'Orientation','horizontal','Box','on');
lgd1.Layout.Tile = 'north';

save_fig(fig1, fullfile(fig_dir,'fig1_recovery_regression.pdf'));

% =============================================================================
%  FIGURE 2 — Correction Effectiveness  (1×4, shared top legend)
% =============================================================================
fig2 = figure('Name','Correction Effectiveness', ...
    'Units','centimeters','Position',[1 1 FIG_W FIG_H_4]);

tl2 = tiledlayout(fig2, 1, 4, ...
    'Padding','compact','TileSpacing','compact');

corr_cols   = {'answer_change_rate',    'net_correction_benefit', ...
               'flip_wrong_to_correct', 'flip_correct_to_wrong'};
corr_ylabs  = {'Fraction of Rollouts',  'Net Benefit (fraction)', ...
               'Fraction of Rollouts',  'Fraction of Rollouts'};
corr_titles = {'\textbf{(a)} Answer Change Rate', ...
               '\textbf{(b)} Net Correction Benefit', ...
               '\textbf{(c)} Flip: W$\rightarrow$C', ...
               '\textbf{(d)} Flip: C$\rightarrow$W'};

h_leg2 = gobjects(nM, 1);
for p = 1:4
    ax = nexttile(tl2);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    for m = 1:nM
        y = avg_depth(T_corr, model_keys(m), corr_cols{p}, depths);
        hh = plot(ax, depths, y, ['-', markers{m}], ...
            'Color', colors(m,:), 'LineWidth', lw, ...
            'MarkerSize', msz, 'MarkerFaceColor', colors(m,:));
        if p == 1, h_leg2(m) = hh; end
    end
    if p == 2   % zero-reference dashed line for net benefit
        yline(ax, 0, '--k', 'LineWidth', 1.0, 'Alpha', 0.45);
    end
    xlabel(ax, 'Refinement Depth $d$', 'Interpreter','latex','FontSize',fsz_label);
    ylabel(ax, corr_ylabs{p},           'Interpreter','latex','FontSize',fsz_label);
    title(ax,  corr_titles{p},          'Interpreter','latex','FontSize',fsz_title);
    xticks(ax, depths); xlim(ax,[0.8 4.2]);
    set(ax,'FontSize',fsz_tick,'TickLabelInterpreter','latex');
end

lgd2 = legend(h_leg2, model_tex, ...
    'Interpreter','latex','FontSize',fsz_leg, ...
    'Orientation','horizontal','Box','on');
lgd2.Layout.Tile = 'north';

save_fig(fig2, fullfile(fig_dir,'fig2_correction_effectiveness.pdf'));

% =============================================================================
%  FIGURE 3 — Diversity  (1×2, shared top legend)
% =============================================================================
fig3 = figure('Name','Diversity', ...
    'Units','centimeters','Position',[1 1 FIG_W FIG_H_2]);

tl3 = tiledlayout(fig3, 1, 2, ...
    'Padding','compact','TileSpacing','compact');

% Pre-compute per-model averages across datasets
agree_cols = {'depth_1_agreement','depth_2_agreement', ...
              'depth_3_agreement','depth_4_agreement'};
div_frac  = zeros(1,nM);
div_agree = zeros(nM,4);
for m = 1:nM
    rows = T_div.model == model_keys(m);
    div_frac(m) = mean(T_div.frac_questions_diverse(rows), 'omitnan');
    for d = 1:4
        div_agree(m,d) = mean(T_div.(agree_cols{d})(rows), 'omitnan');
    end
end

% Panel A — Fraction of questions diverse (bar chart, one bar per model)
ax3a = nexttile(tl3);
hold(ax3a,'on'); box(ax3a,'on'); grid(ax3a,'on');
bh = bar(ax3a, 1:nM, div_frac, 0.6, 'FaceColor','flat', 'EdgeColor','none');
for m = 1:nM, bh.CData(m,:) = colors(m,:); end
set(ax3a, ...
    'XTick', 1:nM, 'XTickLabel', model_tex, ...
    'TickLabelInterpreter', 'latex', 'FontSize', fsz_tick);
xtickangle(ax3a, 12);
ylabel(ax3a, 'Fraction of Questions Diverse', ...
    'Interpreter','latex','FontSize',fsz_label);
title(ax3a, '\textbf{(a)} Rollout Diversity', ...
    'Interpreter','latex','FontSize',fsz_title);
ylim(ax3a,[0, 1.08]);

% Panel B — Depth-wise answer agreement (line plot, one series per model)
ax3b = nexttile(tl3);
hold(ax3b,'on'); box(ax3b,'on'); grid(ax3b,'on');
h_leg3 = gobjects(nM,1);
for m = 1:nM
    h_leg3(m) = plot(ax3b, depths, div_agree(m,:), ['-', markers{m}], ...
        'Color', colors(m,:), 'LineWidth', lw, ...
        'MarkerSize', msz, 'MarkerFaceColor', colors(m,:));
end
xlabel(ax3b, 'Refinement Depth $d$',           'Interpreter','latex','FontSize',fsz_label);
ylabel(ax3b, 'Fraction of Questions Agreeing',  'Interpreter','latex','FontSize',fsz_label);
title(ax3b,  '\textbf{(b)} Rollout Convergence per Depth', ...
    'Interpreter','latex','FontSize',fsz_title);
xticks(ax3b, depths); xlim(ax3b,[0.8 4.2]);
set(ax3b,'FontSize',fsz_tick,'TickLabelInterpreter','latex');

lgd3 = legend(h_leg3, model_tex, ...
    'Interpreter','latex','FontSize',fsz_leg, ...
    'Orientation','horizontal','Box','on');
lgd3.Layout.Tile = 'north';

save_fig(fig3, fullfile(fig_dir,'fig3_diversity.pdf'));

% =============================================================================
%  FIGURE 4 — Per-dataset heatmaps  (1×2, model names as row labels)
%  No line legend needed — models are already labelled on the y-axis.
% =============================================================================
fig4 = figure('Name','Per-dataset Heatmaps', ...
    'Units','centimeters','Position',[1 1 FIG_W 8.5]);

tl4 = tiledlayout(fig4, 1, 2, ...
    'Padding','compact','TileSpacing','loose');

recovery_mat    = NaN(nM,nDS);
net_benefit_mat = NaN(nM,nDS);
for m = 1:nM
    for ds = 1:nDS
        msk_a = T_acc.model==model_keys(m) & T_acc.dataset==datasets(ds) & T_acc.depth>=2;
        msk_c = T_corr.model==model_keys(m) & T_corr.dataset==datasets(ds) & T_corr.depth>=2;
        if any(msk_a), recovery_mat(m,ds)    = mean(T_acc.recovery_rate(msk_a),'omitnan'); end
        if any(msk_c), net_benefit_mat(m,ds) = mean(T_corr.net_correction_benefit(msk_c),'omitnan'); end
    end
end

dataset_tex = strrep(string(datasets), '_', '\_');

ax4a = nexttile(tl4);
render_heatmap(ax4a, recovery_mat, model_tex, dataset_tex, ...
    ylorrd64(), 'Mean Recovery Rate', ...
    '\textbf{(a)} Recovery Rate per Dataset (Depths 2--4)', fsz_tick, fsz_label);

ax4b = nexttile(tl4);
render_heatmap(ax4b, net_benefit_mat, model_tex, dataset_tex, ...
    rdylgn64(), 'Net Correction Benefit', ...
    '\textbf{(b)} Net Correction Benefit per Dataset (Depths 2--4)', fsz_tick, fsz_label);

save_fig(fig4, fullfile(fig_dir,'fig4_heatmaps.pdf'));

fprintf('\nDone. Figures saved to: %s\n', fig_dir);

% =============================================================================
%  LOCAL FUNCTIONS  (must come after all script code in a .m script file)
% =============================================================================

function y = avg_depth(T, model, colname, depth_vals)
% Average of T.(colname) over all datasets at each depth level.
y = arrayfun(@(d) mean(T.(colname)(T.model==model & T.depth==d), 'omitnan'), depth_vals);
end


function render_heatmap(ax, mat, row_labels, col_labels, cmap_arg, ...
                         cb_label, ttl, fsz_tick, fsz_label)
% Labelled imagesc heatmap with numeric cell annotations.
imagesc(ax, mat);
colormap(ax, cmap_arg);
cb = colorbar(ax);
cb.TickLabelInterpreter = 'latex';
cb.Label.String      = cb_label;
cb.Label.Interpreter = 'latex';
cb.Label.FontSize    = fsz_label;
set(ax, ...
    'XTick', 1:numel(col_labels), 'XTickLabel', col_labels, ...
    'YTick', 1:numel(row_labels), 'YTickLabel', row_labels, ...
    'TickLabelInterpreter', 'latex', 'FontSize', fsz_tick);
xtickangle(ax, 18);
title(ax, ttl, 'Interpreter','latex','FontSize',fsz_label);
[nR, nC] = size(mat);
lo = min(mat(:),[],'omitnan');
hi = max(mat(:),[],'omitnan');
for r = 1:nR
    for c = 1:nC
        if ~isnan(mat(r,c))
            norm_v = (mat(r,c)-lo) / max(hi-lo, 1e-9);
            tcol   = 'black';
            if norm_v > 0.65, tcol = 'white'; end
            text(ax, c, r, sprintf('%.3f', mat(r,c)), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontSize', fsz_tick, 'Interpreter','none','Color',tcol);
        end
    end
end
end


function save_fig(fig, path)
% Export vector PDF (R2020a+) or fall back to print().
if exist('exportgraphics','builtin') || exist('exportgraphics','file')
    exportgraphics(fig, path, 'ContentType','vector','BackgroundColor','none');
else
    print(fig, strrep(path,'.pdf',''), '-dpdf', '-painters');
end
fprintf('  saved  %s\n', path);
end


function cmap = ylorrd64()
% 64-level Yellow-Orange-Red sequential colormap (ColorBrewer YlOrRd).
% Light yellow → orange → dark red: suitable for recovery rate heatmap.
y2o = [linspace(1.00,0.99,32)', linspace(1.00,0.55,32)', linspace(0.80,0.23,32)'];
o2r = [linspace(0.99,0.69,32)', linspace(0.55,0.00,32)', linspace(0.23,0.15,32)'];
cmap = [y2o; o2r];
end


function cmap = rdylgn64()
% 64-level diverging Red-Yellow-Green colormap for net-benefit heatmap.
r2y = [linspace(0.84,1.00,32)', linspace(0.19,0.88,32)', linspace(0.15,0.14,32)'];
y2g = [linspace(1.00,0.10,32)', linspace(0.88,0.59,32)', linspace(0.14,0.22,32)'];
cmap = [r2y; y2g];
end
