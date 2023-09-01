function fit_kww()
    % load data (csv file, first column 'time', following columns w/
    % normalized stress relaxation data

    data = readtable('data.csv');    
    stress_columns = {'data'};
    
    time = data.time;
    colors = {'r'};
    
    % define KWW function with S_0 = 1
    kww_func = @(b,t) exp(-(t/b(1)).^b(2));
    
    % initialize variables to hold tau and beta values
    tau_values = zeros(1, length(stress_columns));
    beta_values = zeros(1, length(stress_columns));
    
    % loop over stress columns
    for i = 1:length(stress_columns)
        stress = data.(stress_columns{i});
        
        % initial guess
        b0 = [median(time), 0.5];
    
        % perform non-linear least squares fitting
        opts = statset('nlinfit');
        opts.RobustWgtFun = 'bisquare';
        [beta,~,~,~,mse] = nlinfit(time, stress, kww_func, b0, opts);
    
        % save tau and beta values
        tau_values(i) = beta(1);
        beta_values(i) = beta(2);
        
        fprintf('Fitted Parameters for %s:\n', stress_columns{i});
        fprintf('τ = %.4f\n', beta(1));
        fprintf('β = %.4f\n', beta(2));
    end
    
    % plot full time series
    figure;
    hold on;
    xlabel('Time');
    ylabel('Normalized Stress');
    
    for i = 1:length(stress_columns)
        stress = data.(stress_columns{i});
        beta = [tau_values(i), beta_values(i)];
        plot(time, stress, 'o', 'Color', colors{i});
        t_fit = linspace(min(time), max(time), 1000);
        plot(t_fit, kww_func(beta, t_fit), '-', 'Color', colors{i});
    end
    
    hold off;
    
    % plot first 10 seconds
    figure;
    hold on;
    xlabel('Time');
    ylabel('Normalized Stress');
    
    for i = 1:length(stress_columns)
        stress = data.(stress_columns{i});
        beta = [tau_values(i), beta_values(i)];
        plot(time, stress, 'o', 'Color', colors{i});
        t_fit = linspace(0, 10, 1000);
        plot(t_fit, kww_func(beta, t_fit), '-', 'Color', colors{i});
    end
    
    xlim([0, 10]);
    hold off;
end
