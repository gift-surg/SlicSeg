function segment_callback_Treebag(handles)
global I;           % image data
global ILabel;      % user-input strokes
global currentI;       % current slice
global currentILabel;  % stroke for current slice
global currentOffLineLabel;
global currentGCLabel;
global currentGeosLabel;
global ExistingTrainingSet;
global ExistingTrainingLabel;
global seedsRGB;
global orf;
global dis;
global onlineP;           % probability map
global offlineP;
global offlineB;           % BaggerTree
global SegEnable;
global OffSegEnable;
global GeoSEnable;
global GCEnable;
global forest_method;
%addpath('./library/OnineRandomForest');
forest_method=1;%0--'treebagger provided in MATLAB';%%1--Online Random Forest_wgt
SegEnable=true;
if(get(handles.checkbox1,'Value'))
    OffSegEnable=true;
    GeoSEnable=true;
    GCEnable=true;
else
    OffSegEnable=false;
    GeoSEnable=false;
    GCEnable=false;    
end
currentI=I;
currentILabel=ILabel;

%% create RGB image with stroke
Isize=size(currentI);
seedsRGB=repmat(currentI,1,1,3);
for i=1:Isize(1)
    for j=1:Isize(2)
        if(currentILabel(i,j)==0)
            continue;
        end
        if(currentILabel(i,j)==127)
            seedsRGB(i,j,1)=255;
            seedsRGB(i,j,2)=0;
            seedsRGB(i,j,3)=0;
        elseif(currentILabel(i,j)==255)
            seedsRGB(i,j,1)=0;
            seedsRGB(i,j,2)=0;
            seedsRGB(i,j,3)=255;
        end
    end
end  
currentGCLabel=currentILabel;
currentGeosLabel=currentILabel;
currentOffLineLabel=currentILabel;

%% Load features from training data
forground=find(currentILabel==127);
background=find(currentILabel==255);
totalseeds=length(forground)+length(background);

featureMatrix=ImageToFeature(currentI);
TrainingSet=zeros(totalseeds,size(featureMatrix,2));
TrainingLabel=zeros(totalseeds,1);
TrainingSet(1:length(forground),:)=featureMatrix(forground,:);
TrainingLabel(1:length(forground))=1;
TrainingSet(length(forground)+1:length(forground)+length(background),:)=featureMatrix(background,:);
TrainingLabel(length(forground)+1:length(forground)+length(background))=0;
ExistingTrainingSet=TrainingSet;
ExistingTrainingLabel=TrainingLabel;
TrainingDataWithLabel=[TrainingSet,TrainingLabel];

disp(['random forest training']);
treeDepth=8;
nTrees=20;
if(forest_method==0)
    orf=TreeBagger(nTrees,ExistingTrainingSet,ExistingTrainingLabel, 'Method', 'classification');
    [Label, Prob]=orf.predict(featureMatrix);
    P0=reshape(Prob(:,2),Isize);
else
    orf=Forest_interface();
    orf.Init(nTrees,treeDepth,nTrees);
    orf.Train(TrainingDataWithLabel');
    Prob=orf.Predict(featureMatrix');
%     Prob=orf.GPUPredict(featureMatrix');
    P0=reshape(Prob,Isize);
end
onlineP=PossibilityConnect(currentI,P0,currentILabel==127);
axes(handles.axes_segoffline);
imshow(onlineP);

offlineB=orf;
offlineP=onlineP;
dis=false;
maxflow_callback(handles);

