classdef RandomForestWrapper < handle
    
    properties (Access = private)
        randomForest
    end
    
    methods
        function d = RandomForestWrapper()
            d.randomForest=Forest_interface();
            d.randomForest.Init(20,8,20);        
        end
        
        function Train(d,currentTrainLabel,featureMatrix)
            % train the random forest using scribbles in on slice
            if(isempty(currentTrainLabel) || isempty(find(currentTrainLabel>0)))
                error('the training set is empty');
            end
            forground=find(currentTrainLabel==127);
            background=find(currentTrainLabel==255);
            totalseeds=length(forground)+length(background);
            if(totalseeds==0)
                error('the training set is empty');
            end
            TrainingSet=zeros(totalseeds,size(featureMatrix,2));
            TrainingLabel=zeros(totalseeds,1);
            TrainingSet(1:length(forground),:)=featureMatrix(forground,:);
            TrainingLabel(1:length(forground))=1;
            TrainingSet(length(forground)+1:length(forground)+length(background),:)=featureMatrix(background,:);
            TrainingLabel(length(forground)+1:length(forground)+length(background))=0;
            TrainingDataWithLabel=[TrainingSet,TrainingLabel];
            d.randomForest.Train(TrainingDataWithLabel');
        end
        
        function P=PredictUsingPrior(d,volSlice,priorSegSlice)
            % get the probability in one slice
            featureMatrix=SlicSegAlgorithm.GetSliceFeature(volSlice);
            Prob=d.randomForest.Predict(featureMatrix');
            P0=reshape(Prob,size(volSlice, 1),size(volSlice, 2));            
            P=RandomForestWrapper.ProbabilityProcessUsingShapePrior(P0,priorSegSlice);
        end
        
        function P=PredictUsingConnectivity(d,currentSeedLabel,volSlice)
            % get the probability in one slice
            featureMatrix=SlicSegAlgorithm.GetSliceFeature(volSlice);
            Prob=d.randomForest.Predict(featureMatrix');
            P0=reshape(Prob,size(volSlice, 1),size(volSlice, 2));
            P=RandomForestWrapper.ProbabilityProcessUsingConnectivity(currentSeedLabel,P0,volSlice);
        end
    end
    
    methods (Static, Access = private) 
        function P = ProbabilityProcessUsingShapePrior(P0,lastSeg)
            Isize=size(lastSeg);
            dis=zeros(Isize);
            se= strel('disk',1);
            temp0=lastSeg;
            temp1=imerode(temp0,se);
            currentdis=0;
            while(~isempty(find(temp1>0)))
                dis0=temp0-temp1;
                currentdis=currentdis+1;
                dis(dis0>0)=currentdis;
                temp0=temp1;
                temp1=imerode(temp0,se);
            end
            maxdis=currentdis;
            
            P=P0;
            outsideIndex=intersect(find(dis==0),find(P>0.5));
            P(outsideIndex)=0.4*P(outsideIndex);
            insideIndex=intersect(find(dis>0) , find(P<0.8));
            P(insideIndex)=P(insideIndex)+0.2*dis(insideIndex)/maxdis;
        end
        
        function P = ProbabilityProcessUsingConnectivity(currentSeedLabel,P0,I)
            PL=P0>=0.5;
            pSe= strel('disk',3);
            pMask=imclose(PL,pSe);
            [H,W]=size(P0);
            HW=H*W;
            indexHW=uint32(zeros(HW,1));
            seedsIndex=find(currentSeedLabel==127);
            seeds=length(seedsIndex);
            indexHW(1:seeds)=seedsIndex(1:seeds);
            L=uint8(zeros(H,W));
            P=P0;
            L(seedsIndex)=1;
            P(seedsIndex)=1.0;
            
            
            fg=I(seedsIndex);
            fg_mean=mean(fg);
            fg_std=sqrt(var(double(fg)));
            fg_min=fg_mean-fg_std*3;
            fg_max=fg_mean+fg_std*2;
            
            current=1;
            while(current<=seeds)
                currentIndex=indexHW(current);
                NeighbourIndex=[currentIndex-1,currentIndex+1,...
                    currentIndex+H,currentIndex+H-1,currentIndex+H+1,...
                    currentIndex-H,currentIndex-H-1,currentIndex-H+1];
                for i=1:8
                    tempIndex=NeighbourIndex(i);
                    if(tempIndex>0 && tempIndex<HW && L(tempIndex)==0 && pMask(tempIndex)>0 && I(tempIndex)>fg_min && I(tempIndex)<fg_max)
                        L(tempIndex)=1;
                        seeds=seeds+1;
                        indexHW(seeds,1)=tempIndex;
                    end
                end
                current=current+1;
            end
            
            Lindex=find(L==0);
            P(Lindex)=P(Lindex)*0.4;
        end
    end
end

