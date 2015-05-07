function drawTree_geodesic(qTree,D,img,StepXY,segMask)
% Function to visalize geodesic tree,
% derived from Antonio's code
% Usage: drawTree_geodesci(qTree,D,img,[StepXY])
% qTree is the backlinks, as returned by Varun's code
% D is the distance transform image
% img is the original image
% StepXY is the subsampling factor for visualizing the tree

KernelSize = 32;                % def. 64
%PathMinLength = 16;             % def. 10
PathMinLength = 8;             % def. 10
blendAlpha=0; % 1 means only img is shown, 0 means only forest is shown
drawOnImage=true; % If this is set to true, then overwrites on image instead of blending it! set blend alpha=0 for that please
BLUR_LINES=0;

if(~exist('StepXY','var')),
  StepXY=10;
end

if(~exist('segMask','var')),
  [h,w,nCh]=size(img);
  segMask=logical(ones(h,w));
end

SAVE_OUT = false;       

DEBUG = true;

Folder = 'ForestComputation';

MyColorMapForBackLinks = [ 0 0 0; 1 0 0; 0.66 0.33 0; 0.33 0.66 0; 0 1 0; 0 0.5 0.5; 0 0 1; 0.33 0 0.66; 0.66 0 0.33]; 


%%% Start -----------------------------------------

%addpath code; warning off;

% create large colormap
Ncm = 1024;
ColRoot = zeros(Ncm,3);
for i=1:Ncm,     ColRoot(i,:) = rand(1,3); end;

% Overiding some colors manually right now
%ColRoot(1:5,:)=[ 150 240 183; 255 87 107;32 0 240;178 154 255; 251 139 0]/255;
%ColRoot(1:1024,:)=repmat([32 0 240]/255,[1024 1]);
%ColRoot(1:1024,:)=repmat([0 0 128]/255,[1024 1]);
ColRoot(1:9,:) = [ 0 0 0; 1 0 0; 0.66 0.33 0; 0.33 0.66 0; 0 1 0; 0 0.5 0.5; 0 0 1; 0.33 0 0.66; 0.66 0 0.33]; 

% input image
[H,W] = size(qTree);

% Convert Varun's backlinks to Antonio's backlinks
qOrig=reshape([1:H*W],[H W]);
qDelta=qTree-qOrig;
BackLinks=zeros([H W]);
for i=1:(H*W)
  linkNum=0;
  switch(qDelta(i))
    case -H
      linkNum=1;
    case -(H+1)
      linkNum=2;
    case -1
      linkNum=3;
    case H-1
      linkNum=4;
    case H
      linkNum=5;
    case H+1
      linkNum=6;
    case 1
      linkNum=7;
    case -(H-1)
      linkNum=8;
    case 0
      linkNum=0;
    otherwise
      error('Unexpected case in backlink conversion\n');
  end
  BackLinks(i)=linkNum;
end

BackLinks_MF = medfilt2(BackLinks);
for i=1:10
    BackLinks_MF = medfilt2(BackLinks_MF);
end;

%figure(4);
%subplot(1,2,1); imagesc(BackLinks); axis image; axis off; colormap(MyColorMapForBackLinks);
%subplot(1,2,2); 
%imagesc(BackLinks_MF); axis image; axis off; colormap(MyColorMapForBackLinks);

% initializations
%Ilayer = 0*Icol;
if(drawOnImage)
  Ilayer=img;
else
  Ilayer = zeros([H W 3]);
end
M_src=ones(H,W);

%%% all the paths
count = 0;
RootList = zeros(1,2);
FIRST_TIME = true;
for PointX = 1:StepXY:W
    for PointY = 1:StepXY:H

        X = PointX;        Y = PointY;        
        if(~segMask(Y,X)), continue; end;
        if( M_src(Y,X) == 0 ), continue; end;  % point has already been taken care of
        
        Path(1,1) = X; Path(2,1) = Y;  % initialization

        % extracting the minimum cost path
        pos = 2;
        for ptindex=1:1000;
            [X,Y,bl] = nextPoint(X,Y,BackLinks);
            Path(1,pos) = X; Path(2,pos) = Y;
            if(bl==0), break; end;
            pos = pos + 1;
        end;

        % check on minimum length of path
        Length = size(Path,2);
        if(Length < PathMinLength), continue; end;

        % extracting the root
        Xroot = Path(1,end);        Yroot = Path(2,end);
        if(FIRST_TIME)
            FIRST_TIME = false;
            RootList(1,1) = Xroot;  RootList(1,2) = Yroot;
            BranchColor = ColRoot(1,:);
        else
            N = size(RootList,1);
            FOUND = false;
            for p = 1:N
                if( RootList(p,1)==Xroot && RootList(p,2)==Yroot )
                    FOUND = true;
                    BranchColor = ColRoot(p,:);
                    break;
                end;
            end;
            if(~FOUND)
                RootList(N+1,1)=Xroot;  RootList(N+1,2)=Yroot;
                BranchColor = ColRoot(N+1,:);
                fprintf('\n %d trees ',size(RootList,1));
            end;
        end;
    
        % smoothing paths
        Kernel = ones(1,KernelSize) / KernelSize;           Pad = 5 * KernelSize;
        tmpx = padarray( Path(1,:), [0 Pad], 'replicate');  tmpy = padarray( Path(2,:), [0 Pad], 'replicate');
        tmpx = convn(tmpx,Kernel,'same');                   tmpy = convn(tmpy,Kernel,'same');
        Path(1,:) = tmpx(1+Pad:end-Pad);                    Path(2,:) = tmpy(1+Pad:end-Pad);

        % now drawing paths
        for pos = 1:Length,
           X = round(Path(1,pos));  Y = round(Path(2,pos));
           if( M_src(Y,X) == 0 ), continue; end;   % joining onto existing paths
           for ch=1:3, Ilayer(Y,X,ch) = BranchColor(1,ch); end;
           M_src(Y,X) = 0;
        end;
        clear Path;
        
        % feedback
        if( rem(count,500)==0 ), fprintf('.'); end;
        if( rem(count,10000)==0 && count > 2), 
            fprintf(' %0.1f%% \n', 100 * count / (W*H/(StepXY*StepXY))); 
            if(DEBUG) % this is for debug purposes only
                %sfigure(4);
                %imshow(Ilayer); title('trees','FontSize',16);
                %drawnow;
            end;
        end;
        count = count+1;
    end;
end;
fprintf('\n');

% Figures --------------------

sfigure(1); clf;
imshow(D/max(D(:))); title('Distances','FontSize',16);

%sfigure(2); clf;
%imshow(M_src); title('Source mask','FontSize',16);

%sfigure(3);
%imshow(Ilayer); title('Trees','FontSize',16);

if(exist('img','var')),
  % blurring the paths a very small amount
  if(BLUR_LINES)
        fact = 2;
        Ilayer_blur = imresize(Ilayer,fact,'bicubic');
        Ilayer_blur = imresize(Ilayer_blur,1/fact,'bicubic');
  else
        Ilayer_blur = Ilayer;
  end;
          
  for ch=1:3
        Iout(:,:,ch) = Ilayer_blur(:,:,ch) + blendAlpha* img(:,:,ch);
  end;    
  sfigure(5); clf;
  imshow(Iout); title('Trees overlay','FontSize',16);
end

drawnow;

function [x,y,bl] = nextPoint(x,y,BackLinks)

bl = BackLinks(y,x);

switch(bl)
       case 0
            x = x; y = y;
        case 1
            x = x - 1; y = y;
        case 2
            x = x - 1; y = y - 1;
        case 3
            x = x; y = y - 1;
        case 4
            x = x + 1; y = y - 1;
        case 5
            x = x + 1; y = y;
        case 6
            x = x + 1; y = y + 1;
        case 7
            x = x; y = y + 1;
        case 8
            x = x - 1; y = y + 1;
end;

function h = sfigure(h)
% SFIGURE  Create figure window (minus annoying focus-theft).
%
% Usage is identical to figure.
%
% Daniel Eaton, 2005
%
% See also figure

if nargin>=1 
	if ishandle(h)
		set(0, 'CurrentFigure', h);
	else
		h = figure(h);
	end
else
	h = figure;
end
