function [hist] = calc_log_hist(img)
%% calculate log chrominace histogram
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%
    uv_0 = -1.421875;
    bin_size = 1 / 64;
    bin_num = 256;
    
    [h, w, ~] = size(img);
    I_log = log(single(img));
    u = I_log(:, :, 2) - I_log(:, :, 1);
    v = I_log(:, :, 2) - I_log(:, :, 3);

    % calculate mask
    valid = ~isinf(u) & ~isinf(v) & ~isnan(u) & ~isnan(v);
    
%     Xc = Psplat2(u(valid), v(valid), ones(nnz(valid),1), ...
%         uv_0, bin_size, bin_num);
%     Xc = Xc / max(eps, sum(Xc(:)));
    
    hist = zeros(256, 256);  
    for i = 1:h 
        for j = 1:w 
            if (valid(i, j)) 
                u_val = round((u(i, j) - uv_0) / bin_size); 
                v_val = round((v(i, j) - uv_0) / bin_size); 
                u_val = max(min(u_val, 256), 1); 
                v_val = max(min(v_val, 256), 1); 
                hist(u_val, v_val) = hist(u_val, v_val) + 1; 
            end 
        end 
    end 
    hist = hist / max(eps, sum(hist(:)));

end

%% function to calculate histogram
function N = Psplat2(u, v, c, bin_lo, bin_step, n_bins)
% Splat a bunch of (u, v) coordinates to a 2D histogram. We splat with a
% periodic edge condition, which is important because we will later
% convolve with periodic edges (FFT)
    ub = 1 + mod(round((u - bin_lo) / bin_step), n_bins);
    vb = 1 + mod(round((v - bin_lo) / bin_step), n_bins);
    N = reshape(accumarray(sub2ind([n_bins, n_bins], ub(:), vb(:)), c(:), ...
                               [n_bins^2, 1]), n_bins, n_bins);
end