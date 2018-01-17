function [hist] = calc_log_hist(img)
%% calculate log chrominace histogram
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%
img = single(img);
[h, w, ~] = size(img);
I_log = log(img);
u = I_log(:, :, 2) - I_log(:, :, 1);
v = I_log(:, :, 2) - I_log(:, :, 3);

% calculate mask
valid = ~isinf(u) & ~isinf(v) & ~isnan(u) & ~isnan(v);

% calculate hist
hist = zeros(256, 256);
step = 0.025;
for i = 1:h
    for j = 1:w
        if (valid(i, j))
            u_val = round(u(i, j) / step) + 128;
            v_val = round(v(i, j) / step) + 128;
            u_val = max(min(u_val, 256), 1);
            v_val = max(min(v_val, 256), 1);
            hist(u_val, v_val) = hist(u_val, v_val) + 1;
        end
    end
end

% normalize
hist = hist / max(sum(hist(:)), eps);
hist = sqrt(hist);

end