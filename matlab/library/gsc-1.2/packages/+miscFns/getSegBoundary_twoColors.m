function [mask,colors]=getSegBoundary_twoColors(seg,clr_outside,clr_inside,width_outside,width_inside)

tmpSeg=(seg==0);
stEl=strel('disk',width_outside);
outer=imerode(tmpSeg,stEl);
mask1=tmpSeg&(~outer);

tmpSeg=(seg==255);
stEl=strel('disk',width_inside);
inner=imerode(tmpSeg,stEl);
mask2=(tmpSeg&(~inner));

mask=mask1|mask2;

indexMask=zeros(size(mask),'uint8');
indexMask(mask1)=1;
indexMask(mask2)=2;

clear mask1 mask2;

clr=[reshape(clr_outside,[1 3]); reshape(clr_inside,[1 3])];
colors=clr(indexMask(mask(:)),:);
colors=colors(:);

mask=repmat(mask,[1 1 3]);
