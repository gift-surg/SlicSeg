function maxflow_callback(handles)
global currentI;
global currentILabel;
global currentOffLineLabel;
global currentGCLabel;
global currentGeosLabel;
global currentSegLabel;
global segRGB;
global offSegLabel;
global offSegRGB;
global gcSegLabel;
global gcSegRGB;
global geosSegLabel;
global geosSegRGB;
global onlineP;
global offlineP;
global lambda;
global sigma;
% global lambdaoff;
% global sigmaoff;
global SegEnable;
global OffSegEnable;
global GeoSEnable;
global GCEnable;
global runtime;
addpath('./library/maxflow'); 
addpath('./library/gsc-1.2/ui');

if(SegEnable)
    [flow, currentSegLabel]=wgtmaxflowmex(currentI,currentILabel,onlineP,lambda,sigma);
    currentSegLabel=1-currentSegLabel;
    segRGB=showSegResult(currentI,currentSegLabel,handles.axes_seg);
end

if(OffSegEnable)
    tic;
    [flow, offSegLabel]=wgtmaxflowmex(currentI,currentOffLineLabel,offlineP,lambda,sigma);
    runtime=runtime+toc;
    offSegLabel=1-offSegLabel;
    offSegRGB=showSegResult(currentI,offSegLabel,handles.axes_segoffline);
end

if(GeoSEnable)
    geosSegLabel=wgtGSCsegmentation(currentI,currentGeosLabel);
    geosSegRGB=showSegResult(currentI,geosSegLabel,handles.axes_geosseg);
end

if(GCEnable)
    alpha=4.8;
    beta=3.5;
    [flow, gcSegLabel]=maxflow_revisemex(currentI,currentGCLabel,alpha,beta);
    gcSegLabel=1-gcSegLabel;
    gcSegRGB=showSegResult(currentI,gcSegLabel,handles.axes_gcseg);

end
saveFrame_callback(handles,false,false);
%% show the result

function IRGB=showSegResult(I,Label,axis)
Isize=size(I);
IRGB=zeros([Isize,3]);
IRGB=uint8(IRGB);
for i=1:3
    IRGB(:,:,i)=I(:,:);
end
axes(axis);
imshow(I);
for i=1:Isize(1)
    for j=1:Isize(2)
        if(i==1 || i==Isize(1) || j==1 || j==Isize(2))
            continue;
        end
        if(Label(i,j)~=0 && ~(Label(i-1,j)~=0 && Label(i+1,j)~=0 && Label(i,j-1)~=0 && Label(i,j+1)~=0))
            IRGB(i,j,1)=0;
            IRGB(i,j,2)=255;
            IRGB(i,j,3)=0;
            hold on;
            plot(j,i,'.g','MarkerSize',2);
        end
    end
end

