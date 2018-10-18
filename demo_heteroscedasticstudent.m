%% set up the models

% elicited penalized complexity prior and inverse-t for length-scales
pm1 = prior_t('s2', 200);
pm2 = prior_t('s2', 200);
nup = prior_pcnu('ualpha', [2 0.1]);
pl1 = prior_invt('s2', 1);
pl2 = prior_invt('s2', 1);

% heteroscedastic student-model with traditional LP approximation
likheteTNF = lik_inputdependentSt2('sig2', 1, 'sig2_prior', ...
    [], 'nu', 20, 'nu_prior', nup, 'fisher', 'off');

% heteroscedastic student-model with traditional FL approximation
likheteF = lik_inputdependentSt2('sig2', 1, 'sig2_prior', ...
    [], 'nu', 20, 'nu_prior', nup, 'fisher', 'on');

% options
opt = optimset('TolFun', 1e-3, 'TolX', 1e-3, 'Display', 'iter', 'maxiter', 600);

% We alredy known the covariate space dimension in advance ...
% set the GP models. 
nin = 1;

% ─ covariance functions
cf1 = gpcf_sexp('selectedVariables', 1:nin, 'magnSigma2_prior', pm1, ...
    'lengthScale_prior', pl1, 'lengthScale', exp(0.01 * randn(1, nin)));

cf2 = gpcf_sexp('selectedVariables', 1:nin, 'magnSigma2_prior', pm2, ...
    'lengthScale_prior', pl2, 'lengthScale', exp(0.01 * randn(1, nin)));

% ─ follow the order of the models in the paper (as above)
gpC = {};
gpC{1} = gp_set('lik', likheteTNF, 'cf', {cf1 cf2}, 'comp_cf', {1 2});
gpC{2} = gp_set('lik', likheteF, 'cf', {cf1 cf2}, 'comp_cf', {1 2});

%% perform the inference 

% set a random number generator
rng(5);

% full data 
L = load('datasets/motorcycle.mat');

[n, ~] = size(L.x);                      % sample size
y = L.y;                                 % measured data
x = L.x;                                 % covariates

% initialize storage for data prediction;
Ey   = cell(1, 2); Ef   = cell(1, 2);
Vary = cell(1, 2); Varf = cell(1, 2);

% find the MAP estimate of the hyperposterior
for m = 1:2
    go = 0;
    while go == 0
        try
            th = [1.5*randn(size(gp_pak(gpC{m})) - [0 1]) (abs(randn) + 2)];
            gpC{m} = gp_unpak(gpC{m}, th);
            gpC{m} = gp_optim(gpC{m}, x, y, 'opt', opt);
            
            go = 1;
        catch
            warning('error optim')
            go = 0;
        end
        
        if go == 1
            g = gp_g(gp_pak(gpC{m}), gpC{m}, x, y);
            if all(abs(g) < 1)
                go = 1;
            else
                go = 0;
            end
        end
    end
end

L = 5;                                   % limit for the plots
xp = linspace(min(x)-L, max(x)+L, 300)';  % new covariates
np = length(xp);                          % sample size new covariates  

for m = 1:2
    % do future data prediction
    [Ef{m}, Varf{m}, ~, Ey{m}, Vary{m}] = gp_pred(gpC{m}, x, y, xp);
end

%% plots

fig = figure('units', 'centimeters', 'position', [1, 1, 15, 25]);
fnts = 10;
set(fig, 'defaulttextfontsize', fnts)  
set(fig, 'defaultaxesfontsize', fnts)

meth = ["Laplace"; "Laplace-Fisher"];                             % methods
titel = ["Data prediction with LP", "Data prediction with LF"];   % title 

for m = 1:2
    
    clear p;
    clear h;
    
    s = subplot(2, 1, m);
    title(titel(m)); ylabel('y'); xlabel('x');
    
    xAxis = [xp; xp(end:-1:1)];
    yAxis = [Ey{m} - 1.96*sqrt(Vary{m}); flipud(Ey{m} + 1.96*sqrt(Vary{m}))]; hold on;
    
    p(3) = fill(xAxis, yAxis, [0 0 0]); 
    set(p(3), 'facealpha', .2, 'EdgeColor','None');
    
    p(1) = plot(x, y, '.', 'markerSize', 10, 'color', [0.5 0 1]);
    p(2) = plot(xp, Ey{m}, 'color', [0 0 0], 'LineWidth', 1.5);
    
    Efp = Ef{m}; Varfp = Varf{m};
    nu = gpC{m}.lik.nu;
    
    p(4) = plot(xp, Efp(1:np) - exp(Efp(np+1:end))*sqrt(nu), '--', 'color', 'k', 'LineWidth', 1.2);
           plot(xp, Efp(1:np) + exp(Efp(np+1:end))*sqrt(nu), '--', 'color', 'k', 'LineWidth', 1.2);

    xlim([min(xp) max(xp)]);
    ylim([-250; 190]);
    
    h = legend(p, 'measured data', 'expected value for future data', 'uncertainty (variance)', ...
        'outlier region', 'location', 'southeast'); 

    % ylim([min(Ey{m}-1.96*sqrt(Vary{m})) max(Ey{m}+1.96*sqrt(Vary{m}))]);
    % set(s, 'pos', get(s, 'pos') + [-0.05 -0.04 0.02 0.06], 'Visible', 'off', ...
    %    'YTick', [], 'XTick', [], 'xtick', [],  'FontSize', 11);
    % text(min(x), max(p(1).YData) + 4, 'Data prediction', 'FontSize', 14);
   
end




