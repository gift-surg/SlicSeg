clear;
I=imread('rice.png');
tic;
features=image2DWTfeature(I);
% features=image2GLCMfeature(I);
toc;
