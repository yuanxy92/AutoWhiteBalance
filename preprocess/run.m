%% run auto white balance 
%

clear;
close all;
fclose all;

path = '/home/shaneyuan/Project/AutoWhiteBalance/data/shi_gehler/preprocessed/GehlerShi';

[img, gt_gain] = data_loader(path, 500);

img = single(img);
img = img / max(img(:));

gt_gain = 1 ./ gt_gain;
gt_gain = gt_gain / sum(gt_gain) * 3;
out = apply_white_balance(img, gt_gain);

img = apply_srgb_gamma(img);
out = apply_srgb_gamma(out);

subplot(1, 2, 1);
imshow(img);
subplot(1, 2, 2);
imshow(out);
