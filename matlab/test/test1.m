% test 1, expand training set sequencially
%% load training data
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
Ntr=500;    % number of all the training data
Iter=5;     % iteration time
Ns=Ntr/Iter;% number of training data each iteration
Nte=100;    % number of testing sample

TrainSet=I(1:Ntr,:)';
TestSet=I(Ntr+1:Ntr+Nte,1:featureN)';
testLabel=I(Ntr+1:Ntr+Nte,featureN+1:featureN+1);

%% 1, use all the training data to train just one time
% training data set and testing data set should be matries
% each column of which is a featre vector.
rf=Forest_interface();
rf.Init(treeN,treeDepth,leastNsample);
rf.Train(TrainSet);
P=rf.Predict(TestSet);
T=zeros(size(P));
T(find(P>0.5))=1;
T(find(P<=0.5))=0;
dif=T-testLabel;
correctN=length(find(dif==0));
correctRate=correctN/length(T);
disp(['add all traning samples to training set at once']);
disp(['correctRate  = ' num2str(correctRate)]);


%% 2, use sequentially arrived training data to train Iter times
rf2=Forest_interface();
rf2.Init(treeN,treeDepth,leastNsample);
disp([' ']);
disp(['expand training set gradually']);
for i=1:Iter
    rf2.Train(TrainSet(:,(i-1)*Ns+1:i*Ns));
    P=rf2.Predict(TestSet);
    T=zeros(size(P));
    T(find(P>0.5))=1;
    T(find(P<=0.5))=0;
    dif=T-testLabel;
    correctN=length(find(dif==0));
    correctRate=correctN/length(T);
    disp(['Number of training samples  = ' num2str(i*Ns)])
    disp(['correctRate  = ' num2str(correctRate)]);
end