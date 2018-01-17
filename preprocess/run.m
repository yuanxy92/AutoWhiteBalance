%% run auto white balance 
%

clear;
close all;
fclose all;

path = '/home/shaneyuan/Project/AutoWhiteBalance/data/shi_gehler/preprocessed/GehlerShi';
num = 568;

train_data = zeros(num, 256, 256);
train_label = zeros(num, 256, 256);

for i = 1:num
    fprintf('Process image %d ...\n', i);
    [img, gt_gain] = data_loader(path, 500);
    [gt_response] = gt_response_from_gt_gain(gt_gain);
    img = single(img);
    img = img / max(img(:));
    hist = calc_log_hist(img);
    
    train_data(i, :, :) = hist;
    train_label(i, :, :) = gt_response;
    
end

save('../data/data.mat','-v7.3', 'train_data', 'train_label');