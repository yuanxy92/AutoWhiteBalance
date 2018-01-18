% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function [loss, d_loss_precond_vec, model, output] = train_loss_func(...
  model_precond_vec, data, regularizer, preconditioner, params)

% Unpack the model from a vector to a struct.
model_precond = VectorToBlob(model_precond_vec, params.shallow_model_meta);

model_fields = {'F'};
model_fields{end+1} = 'B';


for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];

  model.(Q_fft_latent) = bsxfun(@times, preconditioner.(Q_fft_latent), ...
    model_precond.(Q_fft_latent));
  model.(Q_fft) = VecToFft2(model.(Q_fft_latent), params.fft_mapping);
  model.(Q) = ifft2(model.(Q_fft));
end

loss = 0;
d_loss.F_fft = zeros(size(model.F_fft));
d_loss.B = zeros(size(model.B));

data_mass = 0;  % The total weight of all datapoints.

Y_pred = {};
for i_data = 1:length(data)

  X = data(i_data).X;

  if ~isfield(data(i_data), 'X_fft')
    X_fft = fft2(X);
  else
    X_fft = data(i_data).X_fft;
  end
  Y = data(i_data).Y;
  avg_rgb = data(i_data).avg_rgb;

  F_fft = model.F_fft;
  B = model.B;


  [output1, output2, sub_loss, output4] ...
    = evaluate_model(F_fft, B, X, X_fft, Y, [], avg_rgb, params);
  output = {output1, output2, sub_loss, output4};

  Y_pred{i_data} = output1.mu;

  % Each datapoint is weighted, but that weight is currently fixed to 1.
  W = 1;

  % Add to the total loss and its gradients.
  data_mass = data_mass + W;
  loss = loss + W * sub_loss.loss;
  d_loss.F_fft = d_loss.F_fft + W * sub_loss.d_loss_F_fft;
  d_loss.B = d_loss.B + W * sub_loss.d_loss_B;

end
Y_pred = cat(2, Y_pred{:});
d_loss.B_fft = (1 / (size(model.B,1) * size(model.B,2))) * fft2(d_loss.B);

% Vectorize and precondition the gradients.
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];
  d_loss.(Q_fft_latent) = VecToFft2Backprop(d_loss.(Q_fft));
end

d_loss_precond = struct();
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft_latent = [Q, '_fft_latent'];
  d_loss_precond.(Q_fft_latent) = ...
    bsxfun(@times, preconditioner.(Q_fft_latent), d_loss.(Q_fft_latent));
end

% Regularize each model parameter, in the preconditioned space. The magnitude of
% each regularizer is multiplied by the total weight of all datapoints
% (which makes the loss additive). Because this loss is computed and imposed in
% the vectorized and preconditioned space, it is just the sum of squares.
for i_field = 1:length(model_fields)
  Q = model_fields{i_field};
  Q_fft = [Q, '_fft'];
  Q_fft_latent = [Q, '_fft_latent'];

  loss = loss + ...
    0.5 * data_mass * sum(reshape(model_precond.(Q_fft_latent), [], 1).^2);
  d_loss_precond.(Q_fft_latent) = d_loss_precond.(Q_fft_latent) + ...
    data_mass * model_precond.(Q_fft_latent);

  if params.DEBUG.CORRECT_PRECONDITIONER
    error = ...
      (sum(model_precond.(Q_fft_latent)(:).^2)) -  ...
      sum(sum(sum(bsxfun(@times, regularizer.(Q_fft), ...
      (real(model.(Q_fft)).^2 + imag(model.(Q_fft)).^2)))));
    fprintf('========================================\n');
    fprintf('%s preconditioner error    = %e\n', Q, abs(error));
  end
end

d_loss_precond_vec = BlobToVector(d_loss_precond);

end
