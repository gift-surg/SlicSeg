function maxflow_callback(handles)
global volumeSeg;
global sliceStatus;
global currentSegIndex;
global currentViewImageIndex;
global currentI;
global currentILabel;
global currentSegLabel;
global segRGB;
global onlineP;
global lambda;
global sigma;
global runtime;
addpath('./library/maxflow'); 

[flow, currentSegLabel]=wgtmaxflowmex(currentI,currentILabel,onlineP,lambda,sigma);
currentSegLabel=1-currentSegLabel;
se= strel('disk',2);
currentSegLabel=imclose(currentSegLabel,se);
currentSegLabel=imopen(currentSegLabel,se);
volumeSeg(:,:,currentSegIndex)=currentSegLabel(:,:);
sliceStatus(currentSegIndex)=1;
currentViewImageIndex=currentSegIndex;
set(handles.slider_imageIndex,'Value',currentSegIndex);
showResult(handles);