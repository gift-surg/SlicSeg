function ok=start(obj,labelImg)

ok=false;
[h,w]=size(labelImg);

opts=obj.opts;
posteriorImage=gsc.getPosteriorImage(obj.features,labelImg,obj.opts);
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
    [Wstar,obj.starInfo]=gsc.getStarEdges(labelImg,opts.starNbrhood_size,...
                                      obj.posteriorImage,opts.geoGamma);
  case 'imgGrad'
    [Wstar,obj.starInfo]=gsc.getStarEdges(labelImg,opts.starNbrhood_size,...
                                      obj.img,opts.geoGamma);
  otherwise
    error('Invalid star method %s\n',opts.starMethod);
end
[E_n,E_w]=gsc.convertMatrix_toEdgePairs(obj.W,Wstar);
E_n=E_n';
E_w=E_w';

dgcHandle=gsc.cpp.mexDGC('initialize',uint32(h*w),E_n); 
[cut,flow]=gsc.cpp.mexDGC('minimizeDynamic',dgcHandle,uint32([1:h*w]), ...
             checkInfSTE(probDensities), E_n, E_w); 
seg = reshape( ~cut, h, w );
gsc.cpp.mexDGC('cleanup',dgcHandle)

obj.seg=255*uint8(seg);
ok=true;

function STE=checkInfSTE(STE)
% STE passed should be int32

steDiff=STE(1,:)-STE(2,:);
infInd=steDiff>(intmax-1);
STE(1,infInd)=intmax-1+STE(2,infInd);
infInd=steDiff<(-intmax+1);
STE(2,infInd)=intmax-1+STE(1,infInd);
