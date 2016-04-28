classdef SlicSegAlgorithm < handle
    % Interactive segmentation algorithm of Slic-Seg
    % The user selects one start slice and draws some scribbles in that
    % slice to start segmentation.
    properties
        startIndex       % start slice index
        sliceRange       % 2x1 matrix to store the minimum and maximum slice index
        
        seedImage         % 2D seed image containing user-provided scribbles in the start slice
        volumeImage       % 3D input volume image
        segImage          % 3D image for segmentation result
        probabilityImage  % 3D image of probability of being foreground
        
        randomForest      % Random Forest to learn and predict
        lambda            % parameter for max-flow; controls the weight of unary term and binary term
        sigma             % parameter for max-flow; controls the sensitivity of intensity difference
        innerDis          % radius of erosion when generating new training data
        outerDis          % radius of dilation when generating new training data
    end
    
    properties (Dependent)
        imageSize        % 3x1 matrix, size of image (width, height, slices)
    end
    
    events
        SegmentationProgress
    end
    
    methods
        function d=SlicSegAlgorithm
            d.startIndex=0;
            d.sliceRange=[0,0];
            d.randomForest=Forest_interface();
            d.randomForest.Init(20,8,20);
            
            d.lambda=5.0;
            d.sigma=3.5;
            d.innerDis=5;
            d.outerDis=6;
        end
        
        function imageSize = get.imageSize(d)
            if isempty(d.volumeImage)
                imageSize=[0,0,0];
            else
                imageSize=size(d.volumeImage);
            end
        end
        
        function set.volumeImage(d,volumeImage)
            d.volumeImage=volumeImage;
            d.ResetSegmentationResult();
        end
        
        function SetMultipleProperties(d,varargin)
            argin=varargin;
            while(length(argin)>=2)
                d.(argin{1})=argin{2};
                argin=argin(3:end);
            end
        end
        
        function val=Get2DSlice(d,dataName, sliceIndex)
            switch dataName
                case 'volumeImage'
                    val=d.volumeImage(:,:,sliceIndex);
                case 'probabilityImage'
                    val=d.probabilityImage(:,:,sliceIndex);
                case 'segImage'
                    val=d.segImage(:,:,sliceIndex);
                otherwise
                    error([prop_name,'is not a valid image']);
            end
        end
        
        function OpenImage(d,imgFolderName)
            % read volume image from a folder, which contains a chain of
            % *.png images indexed from 1 to the number of slices.
            dirinfo=dir(fullfile(imgFolderName,'*.png'));
            sliceNumber=length(dirinfo);
            
            longfilename=fullfile(imgFolderName,'1.png');
            I=imread(longfilename);
            size2d=size(I);
            size3d=[size2d, sliceNumber];
            volume=uint8(zeros(size3d));
            for i=1:sliceNumber
                tempfilename=fullfile(imgFolderName,[num2str(i) '.png']);
                tempI=imread(tempfilename);
                volume(:,:,i)=tempI(:,:);
            end
            d.volumeImage = volume;
            
        end
        
        function OpenScribbleImage(d,labelFileName)
            % read scribbles in the start slice (*.png rgb file)
            rgbLabel=imread(labelFileName);
            ISize=size(rgbLabel);
            ILabel=uint8(zeros(ISize(1),ISize(2)));
            for i=1:ISize(1)
                for j=1:ISize(2)
                    if(rgbLabel(i,j,1)==255 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==0)
                        ILabel(i,j)=127;
                    elseif(rgbLabel(i,j,1)==0 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==255)
                        ILabel(i,j)=255;
                    end
                end
            end
            d.seedImage = ILabel;
            disp('seed image has been loaded successfully');
            
        end
        
        function ResetSegmentationResult(d)
            d.seedImage=uint8(zeros(d.imageSize(1),d.imageSize(2)));
            d.segImage=uint8(zeros(d.imageSize));
            d.probabilityImage=zeros(d.imageSize);
        end
        
        function SaveSegmentationResult(d,segSaveFolder)
            for index=1:d.imageSize(3)
                segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
                imwrite(d.segImage(:,:,index)*255,segFileName);
            end
            
        end
        
        function StartSliceSegmentation(d)
            if(d.startIndex==0)
                error('slice index should not be 0');
            end
            % segmentation in the start slice
            SeedLabel=d.GetSeedLabelImage();
            currentSeedLabel  = SeedLabel;
            currentTrainLabel = SeedLabel;
            currentSegIndex   = d.startIndex;
            d.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(d.volumeImage(:,:,currentSegIndex)));
            d.probabilityImage(:,:,currentSegIndex)=d.PredictUsingConnectivity(currentSeedLabel,d.volumeImage(:,:,currentSegIndex));
            d.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,d.volumeImage(:,:,currentSegIndex),d.probabilityImage(:,:,currentSegIndex),d.lambda,d.sigma);
        end
        
        function SegmentationPropagate(d)
            % propagate to previous slices
            if(d.sliceRange(1)==0 || d.sliceRange(2)==0)
                error('index range should not be 0');
            end
            currentSegIndex=d.startIndex;
            for i=1:d.startIndex-d.sliceRange(1)
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex-1;
                d.TrainAndPropagate(d,i>1,currentSegIndex,priorSegIndex);
            end
            
            % propagate to following slices
            currentSegIndex=d.startIndex;
            notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));
            for i=d.startIndex:d.sliceRange(2)-1
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex+1;
                d.TrainAndPropagate(d,i>d.startIndex,currentSegIndex,priorSegIndex);
            end
        end
        
        function RunSegmention(d)
            d.StartSliceSegmentation();
            d.SegmentationPropagate();
        end
    end
    
    methods (Access=private)
        function TrainAndPropagate(d,train,currentSegIndex,priorSegIndex)
            [currentSeedLabel,currentTrainLabel]=SlicSegAlgorithm.UpdateSeedLabel(d.segImage(:,:,priorSegIndex),d.innerDis,d.outerDis);
            if(train)
                d.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(d.volumeImage(:,:,priorSegIndex)));
            end
            d.probabilityImage(:,:,currentSegIndex)=d.PredictUsingPrior(d.volumeImage(:,:,currentSegIndex),d.segImage(:,:,priorSegIndex));
            d.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,d.volumeImage(:,:,currentSegIndex),d.probabilityImage(:,:,currentSegIndex),d.lambda,d.sigma);
            notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));            
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
            P=SlicSegAlgorithm.ProbabilityProcessUsingShapePrior(P0,priorSegSlice);
        end
        
        function P=PredictUsingConnectivity(d,currentSeedLabel,volSlice)
            % get the probability in one slice
            featureMatrix=SlicSegAlgorithm.GetSliceFeature(volSlice);
            Prob=d.randomForest.Predict(featureMatrix');
            P0=reshape(Prob,size(volSlice, 1),size(volSlice, 2));
            P=SlicSegAlgorithm.ProbabilityProcessUsingConnectivity(currentSeedLabel,P0,volSlice);
        end        
        
        function Label=GetSeedLabelImage(d)
            Label=d.seedImage;
            [H,W]=size(Label);
            for i=5:5:H-5
                Label(i,5)=255;
                Label(i,W-5)=255;
            end
            for j=5:5:W-5
                Label(5,j)=255;
                Label(H-5,j)=255;
            end
        end
        
    end
    
    methods (Static, Access=private)
        function featureMatrix=GetSliceFeature(I)
            % get the feature matrix for given slice
            dwtFeature=image2DWTfeature(I);
            hogFeature=image2HOGFeature(I);
            %             lbpFeature=image2LBPFeature(I);
            intensityFeature=image2IntensityFeature(I);
            % glmcfeatures=image2GLCMfeature(I);
            % featureMatrix=[intensityFeature dwtFeature];% glmcfeatures];
            featureMatrix=[intensityFeature hogFeature dwtFeature];
        end
        
        function seg = GetSingleSliceSegmentation(currentSeedLabel,currentI,currentP,lambda,sigma)
            % use max flow to get the segmentatio in one slice
            currentSeed=currentSeedLabel;
            
            [flow, currentSegLabel]=wgtmaxflowmex(currentI,currentSeed,currentP,lambda,sigma);
            currentSegLabel=1-currentSegLabel;
            se= strel('disk',2);
            currentSegLabel=imclose(currentSegLabel,se);
            currentSegLabel=imopen(currentSegLabel,se);
            seg=currentSegLabel(:,:);
        end
        
        function [currentSeedLabel,currentTrainLabel] = UpdateSeedLabel(currentSegImage,fgr,bgr)
            % generate new training data (for random forest) and new seeds
            % (hard constraint for max-flow) based on segmentation in last slice
            
            tempSegLabel=currentSegImage;
            fgSe1= strel('disk',fgr);
            fgMask=imerode(tempSegLabel,fgSe1);
            if(length(find(fgMask>0))<100)
                fgMask=bwmorph(tempSegLabel,'skel',Inf);
            else
                fgMask=bwmorph(fgMask,'skel',Inf);
            end
            bgSe1= strel('disk',bgr);
            bgSe2= strel('disk',bgr+1);
            fgDilate1=imdilate(tempSegLabel,bgSe1);
            fgDilate2=imdilate(tempSegLabel,bgSe2);
            bgMask=fgDilate2-fgDilate1;
            currentTrainLabel=uint8(zeros(size(tempSegLabel)));
            currentTrainLabel(fgMask>0)=127;
            currentTrainLabel(bgMask>0)=255;
            
            bgMask=1-fgDilate1;
            currentSeedLabel=uint8(zeros(size(tempSegLabel)));
            currentSeedLabel(fgMask>0)=127;
            currentSeedLabel(bgMask>0)=255;
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
    end
end

