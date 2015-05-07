function updateSeg(obj,labelImg)

[h,w]=size(labelImg);

[diffLabels,okLabels]=checkAnd_diffLabels(obj.prevLabelImg,labelImg);
if(~okLabels)
  error('Problem in sequential system: old brushes not subset of new brushes, perhaps some brush was overwritten\n');
end

opts=obj.opts;
posteriorImage=gscSeq.getPosteriorImage(obj.features,labelImg,obj.opts);
obj.posteriorImage=posteriorImage;
posteriorImage=posteriorImage(:)';

bgClamp=(labelImg(:)==2);
fgClamp=(labelImg(:)==1);

probDensities=[-log(1-posteriorImage); -log(posteriorImage)];
probDensities(probDensities>100)=100;

probDensities(2,bgClamp)=inf;probDensities(1,bgClamp)=0;
probDensities(1,fgClamp)=inf;probDensities(2,fgClamp)=0;

probDensities=int32(round(probDensities*opts.gcScale));

switch(opts.starMethod)
  case 'likeliGrad'
    [Wstar,obj.starInfo]=gscSeq.updateStarEdges(diffLabels,opts.starNbrhood_size,...
                                      obj.posteriorImage,opts.geoGamma,obj.seg,obj.starInfo);
  case 'imgGrad'
    [Wstar,obj.starInfo]=gscSeq.updateStarEdges(diffLabels,opts.starNbrhood_size,...
                                      obj.img,opts.geoGamma,obj.seg,obj.starInfo);
  otherwise
    error('Invalid star method %s\n',opts.starMethod);
end
[E_n,E_w]=gscSeq.convertMatrix_toEdgePairs(obj.W,Wstar);
E_n=E_n';
E_w=E_w';

dgcHandle=gscSeq.cpp.mexDGC('initialize',uint32(h*w),E_n); 
[cut,flow]=gscSeq.cpp.mexDGC('minimizeDynamic',dgcHandle,uint32([1:h*w]), ...
             checkInfSTE(probDensities), E_n, E_w); 
seg = reshape( ~cut, h, w );
gscSeq.cpp.mexDGC('cleanup',dgcHandle)

obj.seg=255*uint8(seg);
obj.prevLabelImg=labelImg;
ok=true;

function [diffLabelImg,labelsOk]=checkAnd_diffLabels(oldLabels,newLabels)
% This function checks if the new user strokes have not overwritten
% any old brush strokes. this condition is required to guarentee
% that the fast edit will work correctly

oldLabels(oldLabels==3|oldLabels==4)=0;
newLabels(newLabels==3|newLabels==4)=0;

mask=(oldLabels(:)~=0);

if(any(newLabels(mask)~=oldLabels(mask)) ),
  labelsOk=false;
  diffLabelImg=[];
  return;
end

[h,w,nFrames]=size(oldLabels);
diffLabelImg=zeros([h w nFrames],'uint8');

mask=(newLabels(:)~=0 & newLabels(:)~=oldLabels(:));
diffLabelImg(mask)=newLabels(mask);

labelsOk=true;

function STE=checkInfSTE(STE)
% STE passed should be int32

steDiff=STE(1,:)-STE(2,:);
infInd=steDiff>(intmax-1);
STE(1,infInd)=intmax-1+STE(2,infInd);
infInd=steDiff<(-intmax+1);
STE(2,infInd)=intmax-1+STE(1,infInd);
