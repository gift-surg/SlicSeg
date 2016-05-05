classdef RandomForestWrapper < handle
    
    properties (Access = private)
        randomForest
    end
    
    methods
        function obj = RandomForestWrapper()
            obj.randomForest = ForestWrapper();
            obj.randomForest.Init(20,8,20);        
        end
        
        function Train(obj, currentTrainLabel, volumeSlice)
            % train the random forest using scribbles in on slice
            
            featureMatrix = RandomForestWrapper.GetSliceFeature(volumeSlice);
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
            obj.randomForest.Train(TrainingDataWithLabel');
        end
        
        function [probabilitySlice, segmentationSlice] = PredictUsingPrior(obj, currentSeedLabel, volumeSlice, priorSegSlice, lambda, sigma)
            % get the probability in one slice
            P0 = obj.Predict(volumeSlice);
            probabilitySlice=RandomForestWrapper.ProbabilityProcessUsingShapePrior(P0,priorSegSlice);
            segmentationSlice = SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel, volumeSlice, probabilitySlice, lambda, sigma);
        end
        
        function [probabilitySlice, segmentationSlice] = PredictUsingConnectivity(obj, currentSeedLabel, volumeSlice, lambda, sigma)
            % get the probability in one slice
            P0 = obj.Predict(volumeSlice);
            probabilitySlice = RandomForestWrapper.ProbabilityProcessUsingConnectivity(currentSeedLabel, P0, volumeSlice);
            segmentationSlice = RandomForestWrapper.GetSingleSliceSegmentation(currentSeedLabel, volumeSlice, probabilitySlice, lambda, sigma);            
        end
    end
    
    methods (Access = private)
        function P0 = Predict(obj, volumeSlice)
            featureMatrix = SlicSegAlgorithm.GetSliceFeature(volumeSlice);
            Prob = obj.randomForest.Predict(featureMatrix');
            P0 = reshape(Prob, size(volumeSlice,1), size(volumeSlice,2));
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
        
        function featureMatrix = GetSliceFeature(I)
            % get the feature matrix for given slice
            dwtFeature = image2DWTfeature(I);
            hogFeature = image2HOGFeature(I);
            %             lbpFeature=image2LBPFeature(I);
            intensityFeature = image2IntensityFeature(I);
            % glmcfeatures=image2GLCMfeature(I);
            % featureMatrix=[intensityFeature dwtFeature];% glmcfeatures];
            featureMatrix = [intensityFeature hogFeature dwtFeature];
        end
        
        function seg = GetSingleSliceSegmentation(currentSeedLabel, currentI, currentP, lambda, sigma)
            % use max flow to get the segmentation in one slice
            
            currentSeed = currentSeedLabel;
            [flow, currentSegLabel] = wgtmaxflowmex(currentI, currentSeed, currentP, lambda, sigma);
            currentSegLabel = 1-currentSegLabel;
            se = strel('disk', 2);
            currentSegLabel = imclose(currentSegLabel, se);
            currentSegLabel = imopen(currentSegLabel, se);
            seg = currentSegLabel(:,:);
        end
        
    end
end

