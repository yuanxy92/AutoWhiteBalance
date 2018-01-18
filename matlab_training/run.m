clear;
close all;
fclose all;

% load ../data/model/data.mat
load ../data/model/data_single_channel.mat
load ../data/model/GehlerShi.mat

%data = data_loader();
%save('../data/model/data_single_channel.mat', 'data');

train_model(data);