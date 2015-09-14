function featureMatrix=ImageToFeature(I)
dwtFeature=image2DWTfeature(I);
hogFeature=image2HOGFeature(I);
intensityFeature=image2IntensityFeature(I);
% glmcfeatures=image2GLCMfeature(I);
% featureMatrix=[intensityFeature dwtFeature];% glmcfeatures];
featureMatrix=[intensityFeature hogFeature dwtFeature];
% featureMatrix=dwtFeature;
end
