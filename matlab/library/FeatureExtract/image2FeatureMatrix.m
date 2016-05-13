function featureMatrix = image2FeatureMatrix(I)
    % image2FeatureMatrix Constructs features based on an input image
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.

    dwtFeature = image2DWTfeature(I);
    hogFeature = image2HOGFeature(I);
    intensityFeature = image2IntensityFeature(I);
    featureMatrix = [intensityFeature hogFeature dwtFeature];
end