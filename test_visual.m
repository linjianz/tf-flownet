clear;clc;

i = 9;
directory = ['/media/csc105/Data/dataset/FlyingChairs/data/0000', num2str(i),'_flow.flo'];
flow_gt = readFlowFile(directory);
img_gt = flowToColor(flow_gt);
figure(1); subplot(1,2,1); imshow(img_gt);

flow_pr = importdata('test/20170630_1/flow_batch_0.mat');
img_pr = flowToColor(squeeze(flow_pr(i,:,:,:)));
subplot(1,2,2); imshow(img_pr);