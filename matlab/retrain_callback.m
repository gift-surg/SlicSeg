function retrain_callback(handles)
global currentI;           % image data
global currentILabel;      % seeds generated based on shape prior, seeds in last frame
global ILabel;             % seeds obtained by user interaction, seeds in current frame
global lastSegLabel;           % segmentation result of last slice

global orf;
global onlineP;           % probability map
global forest_method;
ISize=size(currentI);

%% get new stroke for graph cut using segmentation result of last slice
fgCurrent=find(ILabel==127);% seeds number for forground in current frame 
bgCurrent=find(ILabel==255);% seeds number for background in current frame 
disp(['new strokes have been generated']);

%% load feature set for training

featureMatrix=ImageToFeature(currentI);
TrainingSetPos1(1:length(fgCurrent),:)=featureMatrix(fgCurrent,:);
TrainingSetNeg1(1:length(bgCurrent),:)=featureMatrix(bgCurrent,:);
TrainingSetPosLabel1(1:length(fgCurrent),:)=1;
TrainingSetNegLabel1(1:length(bgCurrent),:)=0;

TrainingSet=[TrainingSetPos1;TrainingSetNeg1];
TrainingSetLabel=[TrainingSetPosLabel1; TrainingSetNegLabel1];
TrainingSetWithLabel=[TrainingSet,TrainingSetLabel];


if(forest_method==0)
    [Label, Prob]=orf.predict(featureMatrix);
    P0=reshape(Prob(:,2),ISize);
else
    orf.Train(TrainingSetWithLabel');
    Prob=orf.Predict(featureMatrix');
    P0=reshape(Prob,ISize);
end
onlineP=PossibilityConnect2(P0,lastSegLabel);

for i=1:ISize(1)
    for j=1:ISize(2)
        if(ILabel(i,j)==127)
            currentILabel(i,j)=ILabel(i,j);
        elseif(ILabel(i,j)==255)
            currentILabel(i,j)=ILabel(i,j);
        end
    end
end
maxflow_callback(handles);