%% Load data
T = readtable('all_models_summary.csv');

% Ensure placement_strategy is categorical for clean x-axis labels
T.placement_strategy = categorical(T.placement_strategy);

% Get unique sensor budgets (sorted descending looks nicer for discussion)
budgets = sort(unique(T.k_budget), 'descend');

%% Loop through each sensor budget
for i = 1:length(budgets)
    k = budgets(i);

    % Filter table for this budget
    Tk = T(T.k_budget == k, :);

    % Sort by placement strategy for consistent ordering
    Tk = sortrows(Tk, 'placement_strategy');

    %% -------- Plot 1: Pipe-only F1 --------
    figure;
    bar(Tk.pipe_only_f1_micro);
    xticks(1:height(Tk));
    xticklabels(Tk.placement_strategy);
    xtickangle(45);
    ylabel('Pipe-only F1 Score');
    xlabel('Sensor Placement Strategy');
    title(sprintf('Pipe-only F1 vs Placement Strategy (k = %d)', k));
    grid on;

    %% -------- Plot 2: Pipe + Position F1 (tolerance) --------
    figure;
    bar(Tk.pipe_pos_tol_f1_micro);
    xticks(1:height(Tk));
    xticklabels(Tk.placement_strategy);
    xtickangle(45);
    ylabel('Pipe + Position F1 Score');
    xlabel('Sensor Placement Strategy');
    title(sprintf('Localization F1 (with tolerance) vs Placement Strategy (k = %d)', k));
    grid on;

    %% -------- Plot 3: Localization MAE --------
    figure;
    bar(Tk.localization_mae_on_correct_pipe);
    xticks(1:height(Tk));
    xticklabels(Tk.placement_strategy);
    xtickangle(45);
    ylabel('Localization MAE (normalized)');
    xlabel('Sensor Placement Strategy');
    title(sprintf('Localization MAE vs Placement Strategy (k = %d)', k));
    grid on;
end
