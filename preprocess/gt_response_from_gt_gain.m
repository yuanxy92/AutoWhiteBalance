function [response] = gt_response_from_gt_gain(gt_gain)
%% calculate log chrominace histogram
%
% @author: Shane Yuan
% @date: Jan 17, 2018
%

step = 0.025;

gt_gain = 1 ./ gt_gain;
gt_gain = gt_gain / gt_gain(2);

l2 = [log(gt_gain(2) / gt_gain(1)), 1, log(gt_gain(2) / gt_gain(3))];
response = zeros(256, 256);
for i = 1:256
    for j = 1:256
        u = (i - 128) * step;
        v = (j - 128) * step;
        l1 = [u, 1, v];
        response(i, j) = acos(l1' * l2 / (norm(l1, 2) * norm(l2, 2))); 
    end
end

end