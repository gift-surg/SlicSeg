function segment_callback_Treebag(handles)
global ILabel;      % user-input strokes
global currentI;       % current slice
global currentILabel;  % stroke for current slice
global ExistingTrainingSet;
global ExistingTrainingLabel;
global seedsRGB;
global orf;
global dis;
global onlineP;           % probability map
global currentViewImageIndex;
global startSegIndex;
global currentSegIndex;
global volumeImage;
global forest_method;
addpath('./library/OnineRandomForest');
forest_method=1;%0--'treebagger provided in MATLAB';%%1--Online Random Forest_wgt
startSegIndex=currentViewImageIndex;
currentSegIndex=currentViewImageIndex;

currentI=volumeImage(:,:,currentViewImageIndex);
currentILabel=ILabel;

%% create RGB image with stroke
Isize=size(currentI);
seedsR=currentI;
seedsG=currentI;
seedsB=currentI;
forground=find(currentILabel==127);
background=find(currentILabel==255);
seedsR(forground)=255;
seedsG(forground)=0;
seedsB(forground)=0;
seedsR(background)=0;
seedsG(background)=0;
seedsB(background)=255;
seedsRGB=[seedsR;seedsG;seedsB];

%% Load features from training data
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
% axes(handles.axes_image);
% imshow(onlineP);

dis=false;
maxflow_callback(handles);
showResult(handles);
