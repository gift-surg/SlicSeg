function next_slice_callback(handles,forward)
global currentI;           % image data
global currentILabel;      % seeds generated based on shape prior
global currentOffLineLabel;
global offSegLabel;
global ILabel;             % seeds obtained by user interaction
global lastSegLabel;           % segmentation result of last slice
global currentSegLabel;
global forwardFlag;
global currentImageIndex;
global imgFolderName;
global orf;
global onlineP;           % probability map
global offlineP;           % probability map
global offlineB;           % BaggerTree
global SegEnable;
global OffSegEnable;
global forest_method;

if(SegEnable)
    lastSegLabel=currentSegLabel;
    ISize=size(currentI);
    ILabel=uint8(zeros(ISize));

    %% get new stroke for graph cut using segmentation result of last slice
    currentILabel=UpdateLabel(lastSegLabel,10,10);
    bgSe1= strel('disk',5);
    bgSe2= strel('disk',6);
    bgMask=imdilate(lastSegLabel,bgSe2)-imdilate(lastSegLabel,bgSe1);
    forground=find(currentILabel==127);
    background=find(bgMask>0);
    disp(['new strokes have been generated']);

    %% load feature set for training
    featureMatrix=ImageToFeature(currentI);
    TrainingSetPos=featureMatrix(forground,:);
    TrainingSetNeg=featureMatrix(background,:);
    TrainingSetPosLabel=ones(length(forground),1);
    TrainingSetNegLabel=0*ones(length(background),1);
    TrainingSet=[TrainingSetPos; TrainingSetNeg];
    TrainingLabel=[TrainingSetPosLabel; TrainingSetNegLabel];
    TrainingDataWithLabel=[TrainingSet TrainingLabel];

    if(forest_method==1)
        orf.Train(TrainingDataWithLabel');
    end
end

%% load new image and new feature sets to segment
forwardFlag=forward;
if(forward)
    currentImageIndex=currentImageIndex+1;
else
    currentImageIndex=currentImageIndex-1;    
end
set(handles.text_currentimage,'String',['current image index ' num2str(currentImageIndex)]);
currentFileName=fullfile(imgFolderName,[num2str(currentImageIndex) '.png']);
currentI=imread(currentFileName);
axes(handles.axes_image);
imshow(currentI);

featureMatrix=ImageToFeature(currentI);

if(SegEnable)
    if(forest_method==0)
        [Label Prob]=orf.predict(featureMatrix);
        P0=reshape(Prob(:,2),ISize);
    else
        Prob=orf.Predict(featureMatrix');
%         Prob=orf.GPUPredict(featureMatrix');
        P0=reshape(Prob,ISize);
    end
    onlineP=PossibilityConnect2(P0,lastSegLabel);
    axes(handles.axes_segoffline);
    imshow(onlineP);
end
% imshow(uint8(onlineP*255));
if(OffSegEnable)
    if(forest_method==0)
        [Label1 Prob1]=offlineB.predict(featureMatrix);
        P1=reshape(Prob1(:,2),size(currentI));
    else
        Prob=offlineB.Predict(featureMatrix');
        P1=reshape(Prob,ISize);
    end
    offlineP=PossibilityConnect2(P1,offSegLabel);
    currentOffLineLabel=UpdateLabel(offSegLabel,10,10);
end    
maxflow_callback(handles);

function newLabel=UpdateLabel(tempSegLabel,fgr,bgr)
    fgSe1= strel('disk',fgr);
    fgMask=imerode(tempSegLabel,fgSe1);
    if(length(find(fgMask>0))<100)
        fgMask=bwmorph(tempSegLabel,'skel',Inf);
    else
        fgMask=bwmorph(fgMask,'skel',Inf);
    end
    bgSe1= strel('disk',bgr);
    bgMask=imdilate(tempSegLabel,bgSe1);
    bgMask=1-bgMask;
    newLabel=uint8(zeros(size(tempSegLabel)));
    newLabel(fgMask>0)=127;
    newLabel(bgMask>0)=255;