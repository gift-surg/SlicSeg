function ok=start(obj,labelImg)

ok=false;
[h,w,nFrames]=size(labelImg);

stpointsfg=find(labelImg==1);
stpointsbg=find(labelImg==2);

[mask,walker_probabilities]=random_walker(obj.img,[stpointsfg;stpointsbg],[ones(length(stpointsfg),1);2*ones(length(stpointsbg),1)]);
seg=(mask==1);
obj.seg=255*uint8(seg);
ok=true;
