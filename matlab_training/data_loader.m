function data = data_loader()
%% load data into cell struct array
%
% @author Shane Yuan
% @date Jan 18, 2018
%

path = '../data/shi_gehler/preprocessed/GehlerShi';
num = 568;
data = [];

for ind = 1:num
    
    fprintf('Preprocess image %d ...\n', ind);
    
    img_path = sprintf('%s/%06d.png', path, ind);
    img = imread(img_path);

    gain_path = sprintf('%s/%06d.txt', path, ind);

    fp = fopen(gain_path);
    gt_gain = fscanf(fp, '%f', [1, 3]);
    fclose(fp);
    
    X =  calc_log_hist(img);
    
    avg_rgb = squeeze(mean(mean(img,1),2));
    avg_rgb = avg_rgb / sqrt(sum(avg_rgb.^2));
    
    l_u = log(gt_gain(2) / gt_gain(1));
    l_v = log(gt_gain(2) / gt_gain(3));
    Y = [l_u; l_v];
    data_element.X = X;
    data_element.Y = Y;
    data_element.avg_rgb = avg_rgb;
    
    data = [data; data_element];

end

end