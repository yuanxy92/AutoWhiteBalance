function save_csv_data()
%% save training and testing data to csv file
%
% @author Shane Yuan
% @date Jan 17, 2018
%
path = '/home/shaneyuan/Project/AutoWhiteBalance/data/shi_gehler/preprocessed/GehlerShi';
[img, gt_gain] = data_loader(path, 500);

csvwrite('/home/shaneyuan/Project/AutoWhiteBalance/data/train_data.csv', img);

end