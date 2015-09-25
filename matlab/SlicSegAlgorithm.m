
classdef SlicSegAlgorithm < handle
    % Interactive segmentation algorithm of Slic-Seg
    % The user selects one start slice and draws some scribbles in that
    % slice to start segmentation.
    properties
        startIndex;      % start slice index
        sliceRange;      % 2x1 matrix to store the minimum and maximum slice index
        currentSegIndex; % current slice index during propagation
        imageSize;       % 3x1 matrix, size of image (width, height, slices)
        
        seedImage;        % 2D seed image containging user-provided scribbles in the start slice
        volumeImage;      % 3D input volume image
        segImage;         % 3D image for segmentation result
        probabilityImage; % 3D image of probability of being foreground
        currentSeedLabel; % 2D image, seeds (hard constraint) for max flow
        currentTrainLabel;% 2D image, labeled scribbles (training data) for random forest
        
        randomForest;     % Random Forest to learn and predict
        lambda;           % parameter for max-flow, control the weight of unary term and binary term
        sigma;            % parameter for max-flow, control the sensitivity of intensity difference
        innerDis;         % radius of erosion when generating new training data
        outerDis;         % radius of dilation when generating new training data
    end
    
    events
        SegmentationProgress
    end
    
    methods (Access=public)
        function d=SlicSegAlgorithm()
            % construction function
            addpath('./library/OnlineRandomForest');
            d.startIndex=0;
            d.sliceRange=[0,0];
            d.currentSegIndex=0;
            d.imageSize=[0,0,0];
            d.randomForest=Forest_interface();
            d.randomForest.Init(20,8,20);
            
            d.lambda=5.0;
            d.sigma=3.5;
            d.innerDis=5;
            d.outerDis=6;
        end
        
        function d=Set(d,varargin)
            argin=varargin;
            while(length(argin)>=2)
                prop=argin{1};
                val=argin{2};
                argin=argin(3:end);
                switch prop
                    case 'startIndex'
                        d.startIndex=val;
                    case 'sliceRange'
                        d.sliceRange=val;
                    case 'volumeImage'
                        d.volumeImage=val;
                        Isize=size(val);
                        d.imageSize=Isize;
                        d.ResetSegmentationResult();
                    case 'seedImage'
                        d.seedImage=val;
                    case 'lambda'
                        d.lambda=val;
                    case 'sigma'
                        d.sigma=val;
                    case 'innerDis'
                        d.innerDis=val;
                    case 'outerDis'
                        d.outerDis=val;
                    otherwise
                        error([prop_name,'is not a valid property']);
                end
            end
        end
        
        function val=Get(d,prop_name)
            switch prop_name
                case 'startIndex'
                    val=d.startIndex;
                case 'currentSegIndex'
                    val=d.currentSegIndex;
                case 'randomForest'
                    val=d.randomForest;
                case 'imageSize'
                    val=d.imageSize;
                case 'volumeImage'
                    val=d.volumeImage;
                case 'seedImage'
                    val=d.seedImage;
                case 'segImage'
                    val=d.segImage;
                case 'probabilityImage'
                    val=d.probabilityImage;
                otherwise
                    error([prop_name,'is not a valid property']);
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
        
        function d=OpenImage(d,imgFolderName)
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
            d.Set('volumeImage',volume);

        end
        
        function d=OpenScribbleImage(d,labelFileName)
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
            d.Set('seedImage',ILabel);
            disp('seed image has been loaded successfully');
            
        end
        
        function d=ResetSegmentationResult(d)
            d.currentSeedLabel=uint8(zeros(d.imageSize(1),d.imageSize(2)));
            d.seedImage=uint8(zeros(d.imageSize(1),d.imageSize(2)));
            d.segImage=uint8(zeros(d.imageSize));
            d.probabilityImage=zeros(d.imageSize);
        end
        
        function d=SaveSegmentationResult(d,segSaveFolder)
            for index=1:d.imageSize(3)
                segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
                imwrite(d.segImage(:,:,index)*255,segFileName);
            end

        end

        function d=StartSliceSegmentation(d)
            if(d.startIndex==0)
                error('slice index should not be 0');
            end
            % segmentation in the start slice
            SeedLabel=d.GetSeedLabelImage();
            d.currentSeedLabel  = SeedLabel;
            d.currentTrainLabel = SeedLabel;
            d.currentSegIndex   = d.startIndex;
            d.Train();
            d.Predict();
            d.GetSingleSliceSegmentation();
            d.UpdateSeedLabel(d.innerDis,d.outerDis);
        end
        
        function d=SegmentationPropagate(d)
            % propagate to previous slices
            if(d.sliceRange(1)==0 || d.sliceRange(2)==0)
                error('index range should not be 0');
            end
            d.currentSegIndex=d.startIndex;
            for i=1:d.startIndex-d.sliceRange(1)
                if(i>1)
                    d.Train();
                end
                d.currentSegIndex=d.currentSegIndex-1;
                d.Predict();
                d.GetSingleSliceSegmentation();
                d.UpdateSeedLabel(d.innerDis,d.outerDis);
                notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(d.currentSegIndex));
            end
            
            
            % propagate to following slices
            d.currentSegIndex=d.startIndex;
            d.UpdateSeedLabel(d.innerDis,d.outerDis);
            notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(d.currentSegIndex));
            for i=d.startIndex:d.sliceRange(2)-1
                if(i>d.startIndex)
                    d.Train();
                end
                d.currentSegIndex=d.currentSegIndex+1;
                d.Predict();
                d.GetSingleSliceSegmentation();
                d.UpdateSeedLabel(d.innerDis,d.outerDis);
                notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(d.currentSegIndex));
            end
        end
        
        function d=RunSegmention(d)
            d.StartSliceSegmentation();
            d.SegmentationPropagate();
        end
    end
    
    methods (Access=protected)
        function featureMatrix=GetSliceFeature(d,n)
            % get the feature matrix for n-th slice
            addpath('./library/dwt');
            addpath('./library/FeatureExtract');
            I=d.volumeImage(:,:,n);
            dwtFeature=image2DWTfeature(I);
            hogFeature=image2HOGFeature(I);
%             lbpFeature=image2LBPFeature(I);
            intensityFeature=image2IntensityFeature(I);
            % glmcfeatures=image2GLCMfeature(I);
            % featureMatrix=[intensityFeature dwtFeature];% glmcfeatures];
            featureMatrix=[intensityFeature hogFeature dwtFeature];
        end
                                
        function d=Train(d)
            % train the random forest using scribbles in on slice
            if(isempty(d.currentSeedLabel) || isempty(find(d.currentSeedLabel>0)))
                error('the training set is empty');
            end
            forground=find(d.currentTrainLabel==127);
            background=find(d.currentTrainLabel==255);
            totalseeds=length(forground)+length(background);
            if(totalseeds==0)
                error('the training set is empty');
            end
            featureMatrix=d.GetSliceFeature(d.currentSegIndex);
            TrainingSet=zeros(totalseeds,size(featureMatrix,2));
            TrainingLabel=zeros(totalseeds,1);
            TrainingSet(1:length(forground),:)=featureMatrix(forground,:);
            TrainingLabel(1:length(forground))=1;
            TrainingSet(length(forground)+1:length(forground)+length(background),:)=featureMatrix(background,:);
            TrainingLabel(length(forground)+1:length(forground)+length(background))=0;
            TrainingDataWithLabel=[TrainingSet,TrainingLabel];
            d.randomForest.Train(TrainingDataWithLabel');
        end
        
        function d=Predict(d)
            % get the probability in one slice
            featureMatrix=d.GetSliceFeature(d.currentSegIndex);
            Prob=d.randomForest.Predict(featureMatrix');
            P0=reshape(Prob,d.imageSize(1),d.imageSize(2));
            d.probabilityImage(:,:,d.currentSegIndex)=P0;
            d.ProbilityProcess();
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
        
        function d=ProbabilityProcessUsingConnectivity(d)
            P0=d.probabilityImage(:,:,d.currentSegIndex);
            
            PL=P0>=0.5;
            pSe= strel('disk',3);
            pMask=imclose(PL,pSe);
            [H,W]=size(P0);
            HW=H*W;
            indexHW=uint32(zeros(HW,1));
            seedsIndex=find(d.currentSeedLabel==127);
            seeds=length(seedsIndex);
            indexHW(1:seeds)=seedsIndex(1:seeds);
            L=uint8(zeros(H,W));
            P=P0;
            L(seedsIndex)=1;
            P(seedsIndex)=1.0;

            I=d.volumeImage(:,:,d.currentSegIndex);
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
            
            d.probabilityImage(:,:,d.currentSegIndex)=P;
        end
        
        function d=ProbabilityProcessUsingShapePrior(d)
            if(d.currentSegIndex<d.startIndex)
                lastSeg=d.segImage(:,:,d.currentSegIndex+1);
            else
                lastSeg=d.segImage(:,:,d.currentSegIndex-1);
            end
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
            
            P=d.probabilityImage(:,:,d.currentSegIndex);
            outsideIndex=intersect(find(dis==0),find(P>0.5));
            P(outsideIndex)=0.4*P(outsideIndex);
            insideIndex=intersect(find(dis>0) , find(P<0.8));
            P(insideIndex)=P(insideIndex)+0.2*dis(insideIndex)/maxdis;
            
            d.probabilityImage(:,:,d.currentSegIndex)=P;
        end
        
        function d=ProbilityProcess(d)
            if(d.currentSegIndex==d.startIndex)
                d.ProbabilityProcessUsingConnectivity();
            else
                d.ProbabilityProcessUsingShapePrior();
            end
        end
        
        function d=GetSingleSliceSegmentation(d)
            % use max flow to get the segmentatio in one slice
            addpath('./library/maxflow'); 
            currentI=d.volumeImage(:,:,d.currentSegIndex);
            currentP=d.probabilityImage(:,:,d.currentSegIndex);
            currentSeed=d.currentSeedLabel;
       
            [flow, currentSegLabel]=wgtmaxflowmex(currentI,currentSeed,currentP,d.lambda,d.sigma);
            currentSegLabel=1-currentSegLabel;
            se= strel('disk',2);
            currentSegLabel=imclose(currentSegLabel,se);
            currentSegLabel=imopen(currentSegLabel,se);
            d.segImage(:,:,d.currentSegIndex)=currentSegLabel(:,:);
        end
        
        function d=UpdateSeedLabel(d,fgr,bgr)
            % generate new training data (for random forest) and new seeds 
            % (hard constraint for max-flow) based on segmentation in last slice
            tempSegLabel=d.segImage(:,:,d.currentSegIndex);
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
            d.currentTrainLabel=uint8(zeros(size(tempSegLabel)));
            d.currentTrainLabel(fgMask>0)=127;
            d.currentTrainLabel(bgMask>0)=255;
            
            bgMask=1-fgDilate1;
            d.currentSeedLabel=uint8(zeros(size(tempSegLabel)));
            d.currentSeedLabel(fgMask>0)=127;
            d.currentSeedLabel(bgMask>0)=255;
        end
       
    end
end

