function [model, train_metadata] = train_model(data)
%% function to train model
%
% @author Shane Yuan
% @date Jan 18, 2017
%
params = load_params;

addpath(genpath('./minFunc'));
addpath(genpath('./lib_fft'))
addpath(genpath('./lib_flatten'))

% normalize 
X_sums = arrayfun(@(x) sum(sum(x.X,2),1), data, 'UniformOutput', false);
X_sums = squeeze(cat(3, X_sums{:}));
assert(all((abs(X_sums - 1) < 1e-8) | (abs(X_sums) < 1e-8)))

rng('default')
models = {};
n_folds = 1;
train_metadata.final_losses = nan(1, n_folds);
train_metadata.train_times = nan(1, n_folds);
train_metadata.opt_traces = ...
  repmat({[]}, 2, n_folds);

% compute fft
for i_data = 1:length(data)
    data(i_data).X_fft = fft2(double(data(i_data).X));
end

% Precompute the weights used to regularize model.F_fft, which correspond to
% a total variation measure in the Fourier domain.
X_sz = size(data(1).X);
if numel(X_sz) == 2
    X_sz(3) = 1;
end
u_variation_fft = abs(fft2([-1; 1]/sqrt(8), X_sz(1), X_sz(2))).^2;
v_variation_fft = abs(fft2([-1, 1]/sqrt(8), X_sz(1), X_sz(2))).^2;
total_variation_fft = u_variation_fft + v_variation_fft;

% A helper function for applying a scale and shift to a stack of images.
apply_scale_shift = @(x, m, b)...
  bsxfun(@plus, ...
    bsxfun(@times, x, permute(m(:), [2,3,1])), permute(b(:), [2,3,1]));

regularizer.F_fft = apply_scale_shift(total_variation_fft, ...
    params.HYPERPARAMS.FILTER_MULTIPLIERS, params.HYPERPARAMS.FILTER_SHIFTS);

regularizer.B_fft = apply_scale_shift(total_variation_fft, ...
    params.HYPERPARAMS.BIAS_MULTIPLIER, params.HYPERPARAMS.BIAS_SHIFT);

% Construct the initial model
shallow_model = [];
shallow_model.F_fft_latent = zeros(X_sz(1) * X_sz(2), X_sz(3));
shallow_model.B_fft_latent = zeros(X_sz(1) * X_sz(2), 1);
[shallow_model_vec, params.shallow_model_meta] = BlobToVector(shallow_model);

params.fft_mapping = Fft2ToVecPrecompute([X_sz(1), X_sz(2)]);
preconditioner.F_fft_latent = Fft2RegularizerToPreconditioner(regularizer.F_fft);
preconditioner.B_fft_latent = Fft2RegularizerToPreconditioner(regularizer.B_fft);

% Collapse the model struct down to a vector, while preserving the metadata
% necessary to reconstruct it.
model = shallow_model;
[model_vec, params.model_meta] = BlobToVector(model);

for i_anneal = 1:params.TRAINING.NUM_ITERS_ANNEAL
    
    vonmises_weight = (i_anneal - 1) / (params.TRAINING.NUM_ITERS_ANNEAL-1);
    params.loss_mult.crossent = (1 - vonmises_weight) * ...
        params.HYPERPARAMS.CROSSENT_MULTIPLIER;
    if isinf(params.HYPERPARAMS.VONMISES_MULTIPLIER)
        params.loss_mult.vonmises = vonmises_weight;
    else
        params.loss_mult.vonmises = vonmises_weight *...
            params.HYPERPARAMS.VONMISES_MULTIPLIER;
    end
      
    iter_weight = (i_anneal - 1) / (params.TRAINING.NUM_ITERS_ANNEAL - 1);
    num_iters = round(exp(...
        log(params.TRAINING.NUM_ITERS_LBFGS_INITIAL) * (1-iter_weight) + ...
        log(params.TRAINING.NUM_ITERS_LBFGS_FINAL) * iter_weight));

    lossfun = @train_loss_func;
    
    lbfgs_options = struct( ...
      'Method', 'lbfgs', ...
      'MaxIter', num_iters, ...
      'Corr', num_iters, ...
      'MaxFunEvals', 4 + 2*num_iters, ...
      'optTol', 0, ...
      'progTol', 0);
    lbfgs_options.Display = 'iter';

    [model_vec, ~, ~, output] = minFunc(lossfun, model_vec, lbfgs_options, ...
      data, regularizer, preconditioner, params);

    train_metadata.opt_traces{i_anneal, 1} = output.trace.fval(:)';
    model = VectorToBlob(model_vec, params.model_meta);
    
end

[train_metadata.final_losses(1), ~, model] = train_loss_func( ...
      model_vec, data, regularizer, preconditioner, params);
model = rmfield(model, 'F_fft_latent');
% Remove unnecessary black-body metadata from the model.
model = rmfield(model, {'B_fft_latent', 'B_fft'});

end