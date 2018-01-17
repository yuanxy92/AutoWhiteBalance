function [out] = apply_white_balance(img, gain)
%% function to apply white balance correction
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%

img(:, :, 1) = img(:, :, 1) * gain(1);
img(:, :, 2) = img(:, :, 2) * gain(2);
img(:, :, 3) = img(:, :, 3) * gain(3);
out = img;

out(out >= 1) = 1;
out(out <= 0) = 0;

end