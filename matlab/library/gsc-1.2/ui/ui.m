% Copyright (C) 2009-10 Varun Gulshan
function varargout = ui(varargin)
% UI M-file for ui.fig
%      UI, by itself, creates a new UI or raises the existing
%      singleton*.
%
%      H = UI returns the handle to a new UI or the handle to
%      the existing singleton*.
%
%      UI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UI.M with the given input arguments.
%
%      UI('Property','Value',...) creates a new UI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ui_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ui

% Last Modified by GUIDE v2.5 10-Jun-2010 00:21:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ui_OpeningFcn, ...
                   'gui_OutputFcn',  @ui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ui is made visible.
function ui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ui (see VARARGIN)

% --- Initializating the state machine of the UI -----

handles.data=[];

set(handles.canvas,'Units','pixels');
handles.data=uidata();
handles.data.baseLineCanvasPosition=get(handles.canvas,'Position');
handles.data.uiState='opened';
cwd=miscFns.extractDirPath(mfilename('fullpath'));
handles.data.cwd=cwd;
handles=stateTransition_function(handles,'started');
% Choose default command line output for ui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ui wait for user response (see UIRESUME)
% uiwait(handles.figure1);

function brush=makeBrush(sz)
  [x,y] = meshgrid(-sz:sz,-sz:sz);
  r = x.^2 + y.^2;
  brush = r<=(sz*sz);
  brush=uint8(brush);

% --- Outputs from this function are returned to the command line.
function varargout = ui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in button_loadImage.
function button_loadImage_Callback(hObject, eventdata, handles)
% hObject    handle to button_loadImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[handles,allOk]=stateTransition_function(handles,'imageLoaded');
guidata(hObject,handles);

function refreshCanvas(handles)
switch(handles.data.canvasView)
  case 'drawing_segBoundary'
    refreshLabelImg_boundary(handles.data);
  case 'orig'
    showOrig(handles);
  case 'posterior'
    showPosterior(handles);
  case 'fg'
    showSegmentation(handles,'fg');
  case 'bg'
    showSegmentation(handles,'bg');
  otherwise
    error('Unknown view state %s\n',handles.data.canvasView);
end

function showOrig(handles)
backImg=handles.data.curImg;
set(handles.data.labelImgHandle,'Cdata',backImg);

function showPosterior(handles,showWhat)
data=handles.data;

try
  evalCmd='data.segmenterH.viewPosterior()';
  postImg=eval(evalCmd);
  if(isempty(postImg))
    postImg=zeros(size(data.labelImg));
  end
catch
  fprintf('No posterior view available\n');
  [h w nChannels]=size(data.curImg);
  img=zeros([h w nChannels]);
  set(data.labelImgHandle,'Cdata',img);

  txtH=text(10,20,'No posterior view available');
  set(txtH,'color','w','fontsize',20,'fontweight','bold');
  snapShot=getframe;
  snapShot.cdata=snapShot.cdata(1:h,1:w,:);
  set(data.labelImgHandle,'Cdata',snapShot.cdata);
  delete(txtH);
  drawnow expose;
  return;
end

postImg=imresize(postImg,size(data.labelImg),'method','nearest');
postImg=repmat(postImg,[1 1 3]);
set(data.labelImgHandle,'Cdata',postImg);
drawnow expose;

function showSegmentation(handles,showWhat)
data=handles.data;
seg=data.curSeg;

switch(showWhat)
  case 'fg'
    mask=(seg==255);
  case 'bg'
    mask=(seg==0);
  otherwise
    error('Unknown case in showSegmentation\n');
end

backImg=data.curImg;
img=backImg;
canvasColor=[0 0 1];
for i=1:3
  tmpImg=img(:,:,i);
  tmpImg(~mask)=canvasColor(i);
  img(:,:,i)=tmpImg;
end
%img=(backImg).*repmat(double(mask),[1 1 size(backImg,3)]);
set(data.labelImgHandle,'Cdata',img);
drawnow expose;

function refreshLabelImg_boundary(data)
  labelImg=data.labelImg;
  backImg=data.curImg;
  backImg(data.segBoundaryMask)=data.segBoundaryColors;

  mask=(labelImg~=0);
  labels=labelImg(mask);
  clrs=data.cmap(labels+1,:);
  clrs=clrs(:);
  backImg(repmat(mask,[1 1 3]))=clrs;

  set(data.labelImgHandle,'CData',backImg);
  %drawnow expose;

function [handles,allOk]=stateTransition_function(handles,nxtState)
% Function to implement the state machine for the user interface
% See your notebook #3 for the state machine diagram, i dont
% have a image of it (If only i had a tablet)

curState=handles.data.uiState;
switch(nxtState)
  case 'started'
     switch(curState)
       case {'imagePPed','segStarted'}
         handles=cleanUp(handles);
     end
     [handles,allOk]=resetUI(handles);
  case 'imageLoaded'
    if(strcmp(curState,'image_reload'))
      % Do nothing
      allOk=true;
    else
      switch(curState)
        case {'imagePPed','segStarted'}
          handles=cleanUp(handles);
      end
      [handles,allOk]=loadImage(handles);
    end
  case 'imagePPed'
    % You can only come to this state from imageLoaded, any other state
    % raise error
    if(~strcmp(curState,'imageLoaded'))
      error('Going to state imagePPed from %s state not allowed, bug\n',curState);
    end
    [handles,allOk]=ppImage(handles);
  case 'segStarted'
    % raise error
    if(~(strcmp(curState,'imagePPed')|strcmp(curState,'segStarted') ))
      error('Going to state segStarted from %s state not allowed, bug\n',curState);
    end
    [handles,allOk]=segmentImg(handles);
  case 'image_reload'
    if(~(strcmp(curState,'imagePPed')|strcmp(curState,'segStarted')|...
         strcmp(curState,'imageLoaded')))
      error('Going to state image_reload from %s state not allowed, bug\n',curState);
    end
    [handles,allOk]=reloadImage(handles);

  otherwise
    error('Invalid state %s requested, bug\n',nxtState);
end

if(allOk)
  handles.data.uiState=nxtState;
else
  fprintf('UI could not make state transition to %s\n',nxtState);
end

function [handles,allOk]=resetUI(handles)

set(handles.canvas,'Units','pixels');
set(handles.panel_stroke,'SelectedObject',handles.radio_strokeFG_hard);

set(handles.panel_view,'visible','off');
set(handles.button_startSeg,'enable','off');
set(handles.button_saveLabels,'enable','off');
set(handles.button_loadLabels,'enable','off');
set(handles.button_preProcess,'enable','off');
set(handles.button_clearFrame,'enable','off');
set(handles.button_preProcess,'enable','off');
set(handles.chkbox_autoEdit,'Value',1);
set(handles.button_updateSeg,'enable','off');
set(handles.button_saveSeg,'enable','off');
set(handles.button_diagnose,'enable','off');
set(handles.button_reloadImage,'enable','off');
set(handles.button_popOut,'enable','off');
set(handles.slider_gamma,'enable','off');
set(handles.popup_segMethod,'String',{...
   'BJ','PP','ESC','GSC','ESCseq','GSCseq','RW','SP-IG',...
   'SP-LIG','SP-SIG','Custom'});
set(handles.popup_segMethod,'Value',2);

set(handles.figure1,'KeyPressFcn',@figure1_MyKeyPressFcn);

data=handles.data;
%data.baseLineCanvasPosition=handles.data.baseLineCanvasPosition;
data.debugLevel=1;
data.drag=[];
data.strokeType=1; % 1 is for fgHard
set(handles.panel_view,'SelectedObject',handles.radio_boundaryView);
data.canvasView='drawing_segBoundary';
data.uiState='started';
data.brushRad=3;
data.brushMask=makeBrush(data.brushRad);
data.brushType=1;
%set(handles.panel_strokeType,'SelectedObject',handles.radio_brushBrush);
%data.boxStarted=false;

data.segBoundaryColors=[];
data.segBoundaryMask=[];
data.segBoundaryColor_inside=[1 0 0];
data.segBoundaryColor_outside=[0 1 0];
data.segBoundaryWidth=4;

data.gamma=150;
data.geoGamma=0.3;

data.origCanvasPosition=data.baseLineCanvasPosition;
set(handles.canvas,'Position',data.baseLineCanvasPosition);
cla(handles.canvas,'reset');

segMethodStrings=get(handles.popup_segMethod,'String');
data.segMethod=segMethodStrings{get(handles.popup_segMethod,'Value')};

set(handles.canvas,'xlimmode','manual','ylimmode','manual','zlimmode','manual');
set(handles.canvas,'XTick',[],'YTick',[]);
set(handles.slider_canvasSize,'enable','on');
set(handles.slider_canvasSize,'value',1.0);
set(handles.slider_brushSize,'min',0);
set(handles.slider_brushSize,'max',10);
set(handles.slider_brushSize,'sliderstep',[1/9 1/9]);
set(handles.slider_brushSize,'value',data.brushRad);
set(handles.text_brushSize,'string',sprintf('Brush size: %d',data.brushRad));

set(handles.slider_gamma,'value',data.gamma);
set(handles.text_gamma,'string',sprintf('Gamma: %.2f',data.gamma));
set(handles.text_geoGamma,'string',sprintf('Geo Gamma: %.2f',data.geoGamma));
set(handles.textEdit_geoGamma,'string',sprintf('%.3f',data.geoGamma));

handles.data=data;
allOk=true;

function handles=cleanUp(handles)
try
  if(~isempty(handles.data.segmenterH)),
    delete(handles.data.segmenterH);
  end
  handles.data.segmenterH=[];
catch
  fprintf('No cleaning up to do\n');
end

function [handles,allOk]=segmentImg(handles)
  allOk=false;
  allOk=handles.data.segmenterH.start(handles.data.labelImg_orig);
  if(~allOk), return; end;
  curSeg=imresize(handles.data.segmenterH.seg, ...
                  [handles.data.hCanvas handles.data.wCanvas],'nearest');
  handles.data.curSeg=curSeg;

  [handles.data.segBoundaryMask,handles.data.segBoundaryColors]= ...
  miscFns.getSegBoundary_twoColors(curSeg,handles.data.segBoundaryColor_outside, ...
                           handles.data.segBoundaryColor_inside...
                          ,handles.data.segBoundaryWidth,handles.data.segBoundaryWidth);

  refreshCanvas(handles);
  set(handles.panel_stroke,'SelectedObject',handles.radio_brushAuto);
  guidata(handles.radio_brushAuto,handles);
  panel_stroke_SelectionChangeFcn(handles.radio_brushAuto,[],handles);
  handles=guidata(handles.radio_brushAuto);
  set(handles.button_updateSeg,'enable','on');
  set(handles.button_startSeg,'enable','off');
  allOk=true;

function [handles,allOk]=ppImage(handles)
  allOk=false;
  clear segOptions;
  handles.data.segmenterH=segOptions(handles.data);
  handles.data.segmenterH.preProcess(handles.data.inImg);

  set(handles.button_startSeg,'enable','on');
  set(handles.button_preProcess,'enable','off');
  set(handles.slider_gamma,'enable','off');
  set(handles.popup_segMethod,'enable','off');
  allOk=true;

function [handles,allOk]=reloadImage(handles)

allOk=false;

handles=cleanUp(handles);
[handles,allOk]=loadImage(handles,true);

function [handles,allOk]=loadImage(handles,reload)

if(~exist('reload','var')), reload=false; end
allOk=false;
if(~reload)
  [filename, pathname] = uigetfile({'*.*';'*.png';'*.jpg';'*.bmp'},'Open image to segment',...
                        [handles.data.cwd '../data/']);
                      
  if isequal(filename,0),  return; end;
else
  filename=handles.data.inFileName;
  pathname=handles.data.pathname;
end

inImg=imread([pathname filename]);
[h,w,nCh]=size(inImg);
handles.data.inImg=im2double(inImg);

set(handles.button_updateSeg,'enable','off');
set(handles.button_clearFrame,'enable','on');
set(handles.button_preProcess,'enable','on');
set(handles.button_loadLabels,'enable','on');
set(handles.button_saveLabels,'enable','on');
set(handles.button_saveSeg,'enable','on');
set(handles.button_diagnose,'enable','on');
set(handles.button_reloadImage,'enable','on');
set(handles.button_popOut,'enable','on');
set(handles.slider_canvasSize,'enable','off');
set(handles.slider_gamma,'enable','on');
set(handles.popup_segMethod,'enable','on');
set(handles.button_startSeg,'enable','off');

set(handles.panel_stroke,'SelectedObject',handles.radio_strokeFG_hard);
panel_stroke_SelectionChangeFcn(handles.radio_strokeFG_hard,[],handles);
handles=guidata(handles.radio_strokeFG_hard);

set(handles.panel_view,'visible','on');

handles.data.segBoundaryMask=[];
handles.data.segBoundaryColors=[];
handles.data.inFileName=filename;
handles.data.pathname=pathname;

axes(handles.canvas);
set(gcf,'DoubleBuffer','on');
set(handles.canvas,'xlimmode','manual','ylimmode','manual','zlimmode','manual');
set(handles.canvas,'XTick',[],'YTick',[]);

numLabels=length(get(handles.panel_stroke,'Children')); 

alpha=1;
amap=alpha*ones(numLabels,1);
amap(1)=0;
handles.data.amap=amap;

% -- Set up the color map ---
cmap=colorcube(10*numLabels);
cmap=cmap([1:10:10*numLabels],:);

% Make the colors bright:
cmap=brighten(cmap,0.7);

handles.data.cmap=cmap;
origPos=handles.data.origCanvasPosition;
hScale=w/origPos(3);
vScale=h/origPos(4);
if(hScale>vScale), 
  handles.data.imgReScale=hScale;
  newCanvasHeight=floor(h/hScale);
  newY=floor((origPos(4)-newCanvasHeight)/2+origPos(2));
  set(handles.canvas,'Position',[origPos(1) newY origPos(3) newCanvasHeight]);
else
  handles.data.imgReScale=vScale;
  %newCanvasWidth=floor(w/vScale);
  newCanvasWidth=8*(round((w/vScale)/8)); % Hack to make width multiple of 8
  newX=floor((origPos(3)-newCanvasWidth)/2+origPos(1));
  set(handles.canvas,'Position',[newX origPos(2) newCanvasWidth origPos(4)]);
end;
p=get(handles.canvas,'Position');p=round(p);

fprintf('Original image resolution = %d x %d\n',w,h);
fprintf('Canvas resolution %d x %d\n',p(3),p(4));
handles.data.hCanvas=p(4);
handles.data.wCanvas=p(3);

if(~reload)
  handles.data.labelImg=repmat(uint8(0),[p(4) p(3)]);
  handles.data.labelImg_orig=zeros([h w],'uint8');
end

handles.data.labelImgHandle=imshow(handles.data.labelImg(:,:,1)); 
handles.data.curImg=im2double(imresize(inImg,[p(4) p(3)],'nearest'));

handles.data.curSeg=zeros([p(4) p(3)],'uint8');

handles.data.defSaveLocation=['data/savedLabels/' filename(1:end-4) '-labels.png'];
handles.data.defSaveLocation_seg=['data/savedSegs/' filename(1:end-4) sprintf('-seg.png')];
refreshCanvas(handles);
allOk=true;

% --- Executes on button press in button_startSeg.
function button_startSeg_Callback(hObject, eventdata, handles)
% hObject    handle to button_startSeg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles=stateTransition_function(handles,'segStarted');
guidata(hObject,handles);

% --- Executes on button press in button_saveLabels.
function button_saveLabels_Callback(hObject, eventdata, handles)
% hObject    handle to button_saveLabels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname, filIndex] = uiputfile({'*.png'},'Save labels',handles.data.defSaveLocation);
if(isequal(filename,0)), return; end;
handles.data.defSaveLocation=[pathname filename];
labelImg=handles.data.labelImg_orig;
cmap=handles.data.cmap;
cmap(1,:)=0;
%save([pathname filename],'labelImg');
imwrite(labelImg,cmap,[pathname filename]);
guidata(hObject,handles);

% --- Executes on button press in button_loadLabels.
function button_loadLabels_Callback(hObject, eventdata, handles)
% hObject    handle to button_loadLabels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname] = uigetfile({'*.png'},'Select labels to load',handles.data.defSaveLocation);
if(isequal(filename,0)), return; end;

labelImg_orig=imread([pathname filename]);
labelImg=imresize(labelImg_orig,size(handles.data.labelImg),'method','nearest');

handles.data.labelImg_orig=labelImg_orig;
handles.data.labelImg=labelImg;
refreshCanvas(handles);
guidata(hObject,handles);

% --- Executes on button press in button_preProcess.
function button_preProcess_Callback(hObject, eventdata, handles)
% hObject    handle to button_preProcess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles=stateTransition_function(handles,'imagePPed');
guidata(hObject,handles);

% --- Executes when selected object is changed in panel_stroke.
function panel_stroke_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in panel_stroke 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

switch get(hObject,'tag')
  case 'radio_strokeNone'
    handles.data.strokeType=0;
  case 'radio_strokeFG_hard'
    handles.data.strokeType=1;
  case 'radio_strokeBG_hard'
    handles.data.strokeType=2;
  case 'radio_brushAuto'
    handles.data.strokeType=3;
  otherwise
    fprintf('Warning: unknown stroke type selected\n');
end;
guidata(hObject,handles);

% -------- Documenting the numbers in the label image ---
%  label 0:  no user brush
%  label 1: hard fg user brush
%  label 2: hard bg user brush
%  label 3: Segmented by algorithm fg
%  label 4: Segmented by algorithm bg
%  label 5: Auto brush stroke, this will only
%  remain temporarlily while the stroke is under construction
%  should not be part of the label img usually.

% --- Executes when selected object is changed in panel_view.
function panel_view_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in panel_view 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch get(hObject,'tag')
  case 'radio_boundaryView'
    handles.data.canvasView='drawing_segBoundary';
  case 'radio_viewOrig'
    handles.data.canvasView='orig';
  case 'radio_viewPosterior'
    handles.data.canvasView='posterior';
  case 'radio_canvasFG'
    handles.data.canvasView='fg';
  case 'radio_canvasBG'
    handles.data.canvasView='bg';
  otherwise
    fprintf('Warning: unknown view type selected\n');
end;
refreshCanvas(handles);
guidata(hObject,handles);

% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(strcmp('started',handles.data.uiState)) return; end;
if(~(strcmp('drawing_segBoundary',handles.data.canvasView)|strcmp('fg',handles.data.canvasView)|strcmp('bg',handles.data.canvasView))) return; end;

pt1= get(handles.canvas,'CurrentPoint'); pt1=pt1(1,1:2);
if(pt1(1) <= 0 || pt1(1)>size(handles.data.labelImg,2) || pt1(2) <= 0 || pt1(2)>size(handles.data.labelImg,1))
 return;
end;

switch(handles.data.brushType)
  case 1 %Brush
    handles.data.drag = pt1;
  otherwise
    return;
end;

guidata(hObject, handles);

% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch(handles.data.brushType)
  case 1 %Brush
    if isempty(handles.data.drag), return,end;
    pt1 = handles.data.drag;
    pt2 = get(handles.canvas,'CurrentPoint');pt2=pt2(1,1:2);
    handles.data.drag = pt2;
    handles.data = drawBrush(handles.data,pt1,pt2);
    refreshCanvas(handles);
    guidata(hObject, handles);
  otherwise
    return;
end

function data = drawBrush(data,pt1,pt2)
diff = pt2-pt1;
[H W]=size(data.labelImg);
r = ceil(sqrt(sum(diff.^2))+1E-10);
col=uint8(data.strokeType);
for i=0:r
    pt = round(pt1+diff*(i/r));
    xLeftOffset=max(1-(pt(1)-data.brushRad),0);
    xRightOffset=min(0,W-(pt(1)+data.brushRad));
    yLeftOffset=max(1-(pt(2)-data.brushRad),0);
    yRightOffset=min(0,H-(pt(2)+data.brushRad));
    xrng = pt(1)-data.brushRad+xLeftOffset:pt(1)+data.brushRad+xRightOffset;
    yrng = pt(2)-data.brushRad+yLeftOffset:pt(2)+data.brushRad+yRightOffset;
    brmask = data.brushMask(1+yLeftOffset:2*data.brushRad+1+yRightOffset,1+xLeftOffset:2*data.brushRad+1+xRightOffset);
    data.labelImg(yrng,xrng) = data.labelImg(yrng,xrng).*(1-brmask) + brmask*col;
end

% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonUpFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if(strcmp('started',handles.data.uiState)) return; end;
if(~(strcmp('drawing_segBoundary',handles.data.canvasView)|strcmp('fg',handles.data.canvasView)|strcmp('bg',handles.data.canvasView))) return; end;
pt1= get(handles.canvas,'CurrentPoint'); pt1=pt1(1,1:2);
if(pt1(1) <= 0 || pt1(1)>size(handles.data.labelImg,2) || pt1(2) <= 0 || pt1(2)>size(handles.data.labelImg,1))
  switch(handles.data.brushType)
    case 1
      if(~isempty(handles.data.drag))
        handles.data.drag = [];
        if(handles.data.strokeType==3)
          handles.data=drawAutoBrush(handles.data);
          handles.data=remapLabelImg(handles.data);
          refreshCanvas(handles);
        end
        guidata(hObject,handles);
        if(get(handles.chkbox_autoEdit,'Value')==1 & handles.data.brushType~=2),
          if(strcmp('segStarted',handles.data.uiState)),
            fprintf('Auto update enabled: Calling update seg\n');
            button_updateSeg_Callback(hObject,[],handles);
            handles=guidata(hObject);
          end;
        end
      end
  end
return;
end

figure1_WindowButtonMotionFcn(hObject,eventdata,handles);
handles=guidata(hObject);

switch(handles.data.brushType)
  case 1
    handles.data.drag = [];
    if(handles.data.strokeType==3)
      handles.data=drawAutoBrush(handles.data);
    end
    handles.data=remapLabelImg(handles.data);
    refreshCanvas(handles);
end;

if(get(handles.chkbox_autoEdit,'Value')==1 & handles.data.brushType~=2),
  if(strcmp('segStarted',handles.data.uiState)),
    fprintf('Auto update enabled: Calling update seg\n');
    button_updateSeg_Callback(hObject,[],handles);
    handles=guidata(hObject);
  end;
end

guidata(hObject,handles);

function data=remapLabelImg(data)
labelImg=imresize(data.labelImg,[size(data.labelImg_orig,1) size(data.labelImg_orig,2)],...
         'nearest');
data.labelImg_orig=labelImg;
data.labelImg=imresize(labelImg,[data.hCanvas data.wCanvas],'nearest');

function data=drawAutoBrush(data)
autoBrushMask=(data.labelImg==3);
curSeg=data.curSeg;
autoBrushMask=imresize(autoBrushMask,size(curSeg),'nearest');
numLabels=zeros(2,1);
% numLabels(1) is the number of fg pixels in curSeg under autoBrush
% numLabels(2) is the number of bg pixels in curSeg under autoBrush
autoBrushLabels=curSeg(autoBrushMask);
numLabels(1)=nnz(autoBrushLabels==255);
numLabels(2)=nnz(autoBrushLabels==0);

[xx,maxInd]=max(numLabels);

switch(maxInd)
  case 1
    data.labelImg(autoBrushMask)=2;
  case 2
    data.labelImg(autoBrushMask)=1;
end

% --- Executes on slider movement.
function slider_brushSize_Callback(hObject, eventdata, handles)
% hObject    handle to slider_brushSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.data.brushRad=round(get(hObject,'Value'));
handles.data.brushMask=makeBrush(handles.data.brushRad);
set(handles.text_brushSize,'String',sprintf('Brush size: %d',handles.data.brushRad));
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider_brushSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_brushSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider_canvasSize_Callback(hObject, eventdata, handles)
% hObject    handle to slider_canvasSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
scale=get(hObject,'Value');
baseLine=handles.data.baseLineCanvasPosition;
centerX=baseLine(1)+baseLine(3)/2;
centerY=baseLine(2)+baseLine(4)/2;
newW=8*(round(baseLine(3)*scale/8));
newH=baseLine(4)*scale;

handles.data.origCanvasPosition=[centerX-newW/2 centerY-newH/2 newW newH];
fprintf('Canvas size =%dx%d\n',...
        round(newW),round(newH));
set(handles.canvas,'Position',handles.data.origCanvasPosition);
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider_canvasSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_canvasSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on key press with focus on figure1 and no controls selected.
function figure1_MyKeyPressFcn(h, evnt)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles=guidata(h);
%fprintf('Character: %c\nModifier: %s\nKey: %s\n',evnt.Character,'No modifier key',evnt.Key);
switch evnt.Key
  case 'uparrow'
    if(strcmp(get(handles.slider_canvasSize,'enable'),'off')),
      return;
    end
    sliderMax=get(handles.slider_canvasSize,'max');
    sliderValue=min(sliderMax,get(handles.slider_canvasSize,'value')+0.1);
    set(handles.slider_canvasSize,'value',sliderValue);
    slider_canvasSize_Callback(handles.slider_canvasSize, [], handles)
  case 'downarrow'
    if(strcmp(get(handles.slider_canvasSize,'enable'),'off')),
      return;
    end
    sliderMin=get(handles.slider_canvasSize,'min');
    sliderValue=max(sliderMin,get(handles.slider_canvasSize,'value')-0.1);
    set(handles.slider_canvasSize,'value',sliderValue);
    slider_canvasSize_Callback(handles.slider_canvasSize, [], handles)
end;

if(strcmp(get(handles.panel_stroke,'visible'),'on'))
  switch(evnt.Key)
    case '1'
      set(handles.panel_stroke,'SelectedObject',handles.radio_strokeFG_hard);
      panel_stroke_SelectionChangeFcn(handles.radio_strokeFG_hard,[],handles);
      handles=guidata(h);
    case '2'
      set(handles.panel_stroke,'SelectedObject',handles.radio_strokeBG_hard);
      panel_stroke_SelectionChangeFcn(handles.radio_strokeBG_hard,[],handles);
      handles=guidata(h);
  end
end

if(strcmp(get(handles.panel_view,'visible'),'on'))
  switch(evnt.Key)
    case 's'
      set(handles.panel_view,'SelectedObject',handles.radio_boundaryView);
      panel_view_SelectionChangeFcn(handles.radio_boundaryView,[],handles);
      handles=guidata(h);
    case 'f'
      set(handles.panel_view,'SelectedObject',handles.radio_canvasFG);
      panel_view_SelectionChangeFcn(handles.radio_canvasFG,[],handles);
      handles=guidata(h);
    case 'b'
      set(handles.panel_view,'SelectedObject',handles.radio_canvasBG);
      panel_view_SelectionChangeFcn(handles.radio_canvasBG,[],handles);
      handles=guidata(h);
  end
end
% --- Executes on button press in button_reset.
function button_reset_Callback(hObject, eventdata, handles)
% hObject    handle to button_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles=stateTransition_function(handles,'started');
guidata(hObject,handles);

% --- Executes on button press in button_clearFrame.
function button_clearFrame_Callback(hObject, eventdata, handles)
% hObject    handle to button_clearFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.data.labelImg(:)=uint8(0);
handle.data.labelImg_orig(:)=uint8(0);
refreshCanvas(handles);
guidata(hObject,handles);

% --- Executes on button press in button_updateSeg.
function button_updateSeg_Callback(hObject, eventdata, handles)
% hObject    handle to button_updateSeg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.data.segmenterH.updateSeg(handles.data.labelImg_orig);
handles.data.curSeg=imresize(handles.data.segmenterH.seg, ...
                   [handles.data.hCanvas handles.data.wCanvas],'nearest');
curSeg=handles.data.curSeg;

[handles.data.segBoundaryMask,handles.data.segBoundaryColors]=...
  miscFns.getSegBoundary_twoColors(curSeg,handles.data.segBoundaryColor_outside, ...
                          handles.data.segBoundaryColor_inside...
                          ,handles.data.segBoundaryWidth,handles.data.segBoundaryWidth);
refreshCanvas(handles);
guidata(hObject,handles);

% --- Executes on button press in button_diagnose.
function button_diagnose_Callback(hObject, eventdata, handles)
% hObject    handle to button_diagnose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

varList={};
tryList={'labelImg','segmenterH'};
nameList={'labelImg','segH'};

for i=1:length(tryList)
  try
    eval(['handles.data.' tryList{i} ';']);
    cmd=[nameList{i} '=handles.data.' tryList{i} ';'];
    eval(cmd);
    varList{end+1}=nameList{i};
  catch
    continue;
  end
end

fprintf('\n------- Diagnosing -------\nVariables for you to analyse:\n');
for i=1:length(varList)
  fprintf([varList{i} '\n']);
end
keyboard;


% --- Executes on button press in button_reloadImage.
function button_reloadImage_Callback(hObject, eventdata, handles)
% hObject    handle to button_reloadImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[handles,allOk]=stateTransition_function(handles,'image_reload');
if(allOk)
  [handles,allOk]=stateTransition_function(handles,'imageLoaded');
end
guidata(hObject,handles);


% --- Executes on slider movement.
function slider_gamma_Callback(hObject, eventdata, handles)
% hObject    handle to slider_gamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.data.gamma=get(hObject,'Value');
set(handles.text_gamma,'string',sprintf('Gamma: %.2f',handles.data.gamma));
guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function slider_gamma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_gamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in button_popOut.
function button_popOut_Callback(hObject, eventdata, handles)
% hObject    handle to button_popOut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

frame=get(handles.data.labelImgHandle,'Cdata');
figure;imshow(frame);


% --- Executes on button press in chkbox_autoEdit.
function chkbox_autoEdit_Callback(hObject, eventdata, handles)
% hObject    handle to chkbox_autoEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chkbox_autoEdit


% --- Executes on button press in radio_brushBrush.
function radio_brushBrush_Callback(hObject, eventdata, handles)
% hObject    handle to radio_brushBrush (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radio_brushBrush


function textEdit_geoGamma_Callback(hObject, eventdata, handles)
% hObject    handle to textEdit_geoGamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of textEdit_geoGamma as text
%        str2double(get(hObject,'String')) returns contents of textEdit_geoGamma as a double
str=get(hObject,'string');
handles.data.geoGamma=str2num(str);
if(isempty(handles.data.geoGamma)),
  handles.data.geoGamma=1;
end
set(handles.text_geoGamma,'string',sprintf('Geo Gamma: %.5f',handles.data.geoGamma));
guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function textEdit_geoGamma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to textEdit_geoGamma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on key press with focus on textEdit_geoGamma and none of its controls.
function textEdit_geoGamma_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to textEdit_geoGamma (see GCBO)
% eventdata  structure with the following fields (see UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)



% --- Executes on selection change in popup_segMethod.
function popup_segMethod_Callback(hObject, eventdata, handles)
% hObject    handle to popup_segMethod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popup_segMethod contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popup_segMethod

vidMethodStrings=get(handles.popup_segMethod,'String');
handles.data.segMethod=vidMethodStrings{get(handles.popup_segMethod,'Value')};
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function popup_segMethod_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popup_segMethod (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
fprintf('Cleaning up ui\n');
handles=cleanUp(handles);

delete(hObject);


% --- Executes on button press in button_saveSeg.
function button_saveSeg_Callback(hObject, eventdata, handles)
% hObject    handle to button_saveSeg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname, filIndex] = uiputfile({'*.png'},'Save seg',handles.data.defSaveLocation_seg);
if(isequal(filename,0)), return; end;
handles.data.defSaveLocation_seg=[pathname filename];
seg=handles.data.curSeg;
imwrite(seg,[pathname filename]);
guidata(hObject,handles);


% --- Executes on scroll wheel click while the figure is in focus.
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	VerticalScrollCount: signed integer indicating direction and number of clicks
%	VerticalScrollAmount: number of lines scrolled for each click
% handles    structure with handles and user data (see GUIDATA)


