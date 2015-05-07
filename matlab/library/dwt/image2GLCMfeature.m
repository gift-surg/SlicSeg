function featureSet=image2GLCMfeature(I)
featureSet=double(wgtGLCM(I));
[m n]=size(featureSet);
MeanValues=repmat(mean(featureSet),m,1);
VarValues=repmat(var(featureSet),m,1);
featureSet=(featureSet-MeanValues)./VarValues;