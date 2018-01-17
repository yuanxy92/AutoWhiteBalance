function [response] = gt_response_from_gt_gain(gt_gain)
%% calculate log chrominace histogram
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%

step = 0.025;

gt_gain = 1 ./ gt_gain;
gt_gain = gt_gain / gt_gain(2);

Lu = log(gt_gain(2) / gt_gain(1)) / step;
Lv = log(gt_gain(2) / gt_gain(3)) / step;

response = zeros(256, 256);
Lu = int32(Lu) + 128;
Lv = int32(Lv) + 128;
response(Lu, Lv) = 1.0;

end