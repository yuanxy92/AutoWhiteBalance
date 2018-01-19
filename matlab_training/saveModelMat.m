function saveModelMat(filename, model)
%% function to save model matrix to filename
%
% @author Shane Yuan
% @date Jan 19, 2018
%
fp = fopen(filename, 'w');
fwrite(fp, 256, 'integer*4');
fwrite(fp, 256, 'integer*4');
fwrite(fp, model.F(:), 'single');
fwrite(fp, model.B(:), 'single');
fclose(fp);

end