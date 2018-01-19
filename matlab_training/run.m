clear;
close all;
fclose all;

% load ../data/model/data.mat
load ../data/model/data_single_channel.mat
% load ../data/model/GehlerShi.mat

% data = data_loader();
% save('../data/model/data_single_channel.mat', 'data');

[model, train_metadata] = train_model(data);
save('../data/model/model.mat', 'model');
saveModelMat('../data/model/model.bin', model);