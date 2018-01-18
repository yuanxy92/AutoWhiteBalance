function [response, l_u, l_v] = apply_ffcc_model(hist, model)
%% calculate log chrominace histogram
%
% @author: Shane Yuan
% @date: Jan 18, 2018
%
    uv_0 = -1.421875;
    bin_size = 1 / 64;
    bin_num = 256;
    
    hist_f = fft2(hist);
    model_f = fft2(model.F(:, :, 1));
    bias_f = fft2(model.B) / 2;
    
    response_f = hist_f .* model_f + bias_f;
    response = real(ifft2(response_f));
     
    [~, idx] = max(response(:));
    [i, j] = ind2sub(size(response), idx);
    
    l_u = i * bin_size + uv_0;
    l_v = j * bin_size + uv_0;
end