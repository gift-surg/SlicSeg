function features=vag_addSpatialFeatures(features,img,maxRange)
% Function to add spatial coordinates as features to existing calculated features
% Usage: features=vag_addSpatialFeatures(features,img,maxRange)
%
% Inputs:
%   feautures: a DxN vector of D dimensional features
%   img : only used for computing the size
%   maxRange: the range into which scaling the spatial coordinates should occur.
%     IMP NOTE: We do not scale both the x and y spatial coordinated to [0,1], as that would 
%     mean that we have transformed the image from a rectangle to a square, thereby making
%     a non-similarity transformation. We scale only the larger dimension to [0,1] , the smaller
%     one will get scaled into something [0,c] c<=1

[h,w,nCh]=size(img);
inds=[1:h*w];

[yS,xS]=ind2sub([h w],inds);
scale=maxRange/max(h,w);
yS=scale*(yS-1);
xS=scale*(xS-1); % -1 is to keep them 0 indexed

spFeatures(1,inds)=yS;
spFeatures(2,inds)=xS;

features=[features;spFeatures];
