function [annoSeq,segSeq]=run(segH,robotOpts,img,gtSeg,initLabels)
% [annoSeq,segSeq]=run(segH,robotOpts,img,gtSeg,initLabels)
% Function to run the robot user on the image and return the sequence of segmentations
% and brush strokes.
% Inputs:
%   segH -> handles to a segmentation object
%   robotOpts -> options for robot user
%   img -> original image
%   gtSeg -> ground truth segmentation
%   initLabels -> initial labels
% Outputs:
%  annoSeq -> m x n x K size array of brush strokes.
%  segSeq -> m x n x K sequence of output segmentations

[h w nCh]=size(img);
segH.preProcess(img);

switch(robotOpts.brushStyle),
  case 'dotMiddle'
    startOk=segH.start(initLabels);
    
    numStrokes=robotOpts.numStrokes;
    segSeq=zeros([h w numStrokes+1],'uint8');
    annoSeq=zeros([h w numStrokes+1],'uint8');
    
    annoSeq(:,:,1)=initLabels;
    segSeq(:,:,1)=segH.seg;
    
    prevSeg=segH.seg;
    prevLabels=initLabels;

    brushRad=robotOpts.brushRad;
    robotOpts.brushMask=makeBrush(brushRad);
    for i=1:numStrokes
      labelImg=robot_getNextAnnotation(prevLabels,prevSeg,gtSeg,robotOpts);
      segH.updateSeg(labelImg);
      annoSeq(:,:,i+1)=labelImg;
      segSeq(:,:,i+1)=segH.seg;
      prevSeg=segSeq(:,:,i+1);
      prevLabels=labelImg;
    end
end

function labelImg=robot_getNextAnnotation(prevLabels,seg,gtSeg,robotOpts)
% Function to return the next brush stroke given the current segmentation and brush strokes

errorSeg=(seg~=gtSeg & gtSeg~=128);
fgErrorSeg=(errorSeg& (seg==0));
bgErrorSeg=(errorSeg& (seg==255));

[fgL,fgNum]=bwlabel(fgErrorSeg);
fgErrSizes=zeros(1,fgNum);
for i=1:fgNum
  fgErrSizes(i)=nnz(fgL==i);
end

[bgL,bgNum]=bwlabel(bgErrorSeg);
bgErrSizes=zeros(1,bgNum);
for i=1:bgNum
  bgErrSizes(i)=nnz(bgL==i);
end

if( (fgNum+bgNum)==0),
  fprintf('Perfect segmentation has been attained, no strokes can be added!\n');
  labelImg=prevLabels;
  return;
end

if(fgNum==0),
  maxFgSize=0;
  maxFgInd=0;
else
  [maxFgSize,maxFgInd]=max(fgErrSizes);
end

if(bgNum==0),
  maxBgSize=0;
  maxBgInd=0;
else
  [maxBgSize,maxBgInd]=max(bgErrSizes);
end

if(maxFgSize>maxBgSize),
  newLabelIndex=1; % 6 corr to fg brush
  dTformImg=(fgL==maxFgInd);
else
  newLabelIndex=2;
  dTformImg=(bgL==maxBgInd);
end

% Padding the dTformImg to handle borders correctly
[h,w]=size(dTformImg);
pad_dTformImg=logical(zeros([h+2 w+2]));
pad_dTformImg(2:h+1,2:w+1)=dTformImg;

dTform=bwdist(~pad_dTformImg);
dTform=dTform(2:h+1,2:w+1);
[xx,indCenter]=max(dTform(:));

[centerY,centerX]=ind2sub(size(dTform),indCenter);
boxL=centerX-robotOpts.brushRad;
boxR=centerX+robotOpts.brushRad;
boxU=centerY-robotOpts.brushRad;
boxD=centerY+robotOpts.brushRad;

boxLtrim=max(1,boxL);lOffset=(boxLtrim-boxL);
boxRtrim=min(w,boxR);rOffset=(boxR-boxRtrim);
boxUtrim=max(1,boxU);uOffset=(boxUtrim-boxU);
boxDtrim=min(h,boxD);dOffset=(boxD-boxDtrim);

brushMask=logical(zeros([h w],'uint8'));
brushSize=2*robotOpts.brushRad+1;
brushMask(boxUtrim:boxDtrim,boxLtrim:boxRtrim)=robotOpts.brushMask( (1+uOffset):(brushSize-dOffset), (1+lOffset):(brushSize-rOffset));

switch(newLabelIndex)
  case 1
    brushMask=brushMask & gtSeg==255; % Truncate the brush to lie within the foreground region
  case 2
    brushMask=brushMask & gtSeg==0; % Truncate the brush to lie within the background region

end
%brushMask=brushMask & dTformImg & (gtSeg~=128); % Truncate the brush to lie within the error region

labelImg=prevLabels;
labelImg(brushMask)=newLabelIndex;

function mask=makeBrush(rad)

[x,y]=meshgrid([-rad:rad],[-rad:rad]);
d=(x.*x+y.*y);
mask=(d<=rad*rad);
