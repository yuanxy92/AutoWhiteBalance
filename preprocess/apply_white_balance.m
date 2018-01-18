function [out] = apply_white_balance(img, l_u, l_v)
%% function to apply white balance correction
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%

z = sqrt(exp(-l_u)^2 + exp(-l_v)^2 + 1);
l_r = exp(-l_u) / z;
l_g = 1 / z;
l_b = exp(-l_v) / z;

img = single(img);
img(:, :, 1) = img(:, :, 1) / l_r;
img(:, :, 2) = img(:, :, 2) / l_g;
img(:, :, 3) = img(:, :, 3) / l_b;
out = img ./ max(img(:));

out(out >= 1) = 1;
out(out <= 0) = 0;

end