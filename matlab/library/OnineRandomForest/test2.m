% test 2, compare performance of CPU prediction and GPU prediction
clear;
I=load('testFeature.txt');
isize=size(I);
for i=1:isize(1)
    if(I(i,isize(2))>=1)
        I(i,isize(2))=1;
    else
        I(i,isize(2))=0;
    end
end
featureN=isize(2)-1;
treeN=20;
treeDepth=5;
leastNsample=10;
Ntr=500;
Nte=100;
rf=Forest_interface();
%use tree number and feature number to init
rf.Init(treeN,treeDepth,leastNsample);

TrainSet=I(1:Ntr,:);
TestSet=I(Ntr+1:Ntr+Nte,1:featureN);
testLabel=I(Ntr+1:Ntr+Nte,featureN+1:featureN+1);

% training data set and testing data set should be matries
% each column of which is a featre vector.
rf.Train(TrainSet');
tic;
P=rf.Predict(TestSet');
cpuT=toc;
T=zeros(size(P));
T(find(P>0.5))=1;
T(find(P<=0.5))=0;
dif=T-testLabel;
correctN=length(find(dif==0));
correctRate=correctN/length(T);
disp(['CPU predict time = ' num2str(cpuT)]);
disp(['    correctRate  = ' num2str(correctRate)]);

% [l,r,feature,value]=rf.ConvertTreeToList();

tic;
P=rf.GPUPredict(TestSet');
gpuT=toc;
T=zeros(size(P));
T(find(P>0.5))=1;
T(find(P<=0.5))=0;
dif=T-testLabel;
correctN=length(find(dif==0));
correctRate=correctN/length(T);
disp(['GPU predict time = ' num2str(gpuT)]);
disp(['    correctRate  = ' num2str(correctRate)]);