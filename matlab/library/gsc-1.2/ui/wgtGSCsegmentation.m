% I: uint8
% Label: 127--forground;255--background
function seg=wgtGSCsegmentation(I,Label)
addpath('..');
addpath('../packages');
setup();
opts=sp.segOpts;
obj=sp.segEngine(1,opts);
rgbI=repmat(I,1,1,3);
preProcess(obj,double(rgbI));
labelImg=Label;
labelImg(Label==127)=1;
labelImg(Label==255)=2;
ok=start(obj,labelImg);
seg=obj.seg;