function [img, gt_gain] = data_loader(path, ind)
%% function to load ground truth image and gain from files
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%
img_path = sprintf('%s/%06d.png', path, ind);
img = imread(img_path);

gain_path = sprintf('%s/%06d.txt', path, ind);

fp = fopen(gain_path);
gt_gain = fscanf(fp, '%f', [1, 3]);
fclose(fp);

end