%% load Chikusei dataset
load('HyperspecVNIR_Chikusei_20140729.mat');
%% center crop this image to size 2304 x 2048
img = chikusei(107:2410,144:2191,:);
clear chikusei;
% normalization
img = img ./ max(max(max(img)));
img = single(img);
%% select first row as test images
[H, W, C] = size(img);
test_img_size = 512;
test_pic_num = floor(W / test_img_size);
mkdir test;
cd test;
for i = 1:test_pic_num
    left = (i - 1) * test_img_size + 1;
    right = left + test_img_size - 1;
    test = img(1:test_img_size,left:right,:);
    save(strcat('Chikusei_test_', int2str(i), '.mat'),'test');
end
cd ..
%% the rest left for training
mkdir ('train');
cd train;
img = img((test_img_size+1):end,:,:);
save('Chikusei_train.mat', 'img');
cd ..;