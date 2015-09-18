function varargout = ImageSegUI(varargin)
% IMAGESEGUI MATLAB code for ImageSegUI.fig
%      IMAGESEGUI, by itself, creates a new IMAGESEGUI or raises the existing
%      singleton*.
%
%      H = IMAGESEGUI returns the handle to a new IMAGESEGUI or the handle to
%      the existing singleton*.
%
%      IMAGESEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGESEGUI.M with the given input arguments.
%
%      IMAGESEGUI('Property','Value',...) creates a new IMAGESEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImageSegUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImageSegUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImageSegUI

% Last Modified by GUIDE v2.5 24-Aug-2015 15:29:02

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImageSegUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ImageSegUI_OutputFcn, ...
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
% global definitions


% --- Executes just before ImageSegUI is made visible.
function ImageSegUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImageSegUI (see VARARGIN)

% Choose default command line output for ImageSegUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ImageSegUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);
global slicSeg;
global mouse_state; 
global for_back_ground;
mouse_state=0;
for_back_ground=0;
slicSeg=SlicSegAlgorithm();


% --- Outputs from this function are returned to the command line.
function varargout = ImageSegUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_loadImage.
function pushbutton_loadImage_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_loadImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currentViewImageIndex;
global ILabel;
global slicSeg;
reset(handles.axes_image);
cla(handles.axes_image);
set(gcf,'WindowButtonDownFcn',{@mouse_down});
set(gcf,'WindowButtonMotionFcn',{@mouse_move});
set(gcf,'WindowButtonUpFcn',{@mouse_up});
% imgFolderName=uigetdir;
% defaultimgFolder='/Users/guotaiwang/Documents/MATLAB/ImageSeg/image16_14/img';
[startFileName,imgFolderName,FilterIndex] = uigetfile('*.png','select a file');
slicSeg.OpenImage(imgFolderName);
dirinfo=dir(fullfile(imgFolderName,'*.png'));
sliceNumber=length(dirinfo);
filenameLen=length(startFileName);
currentViewImageIndex=str2num(startFileName(1:filenameLen-4));

imgSize=slicSeg.Get('imageSize');
ILabel=uint8(zeros([imgSize(1), imgSize(2)]));
showResult(handles);

set(handles.text_currentslice,'String',['current slice number: ' num2str(currentViewImageIndex)]);
set(handles.text_totalslice,'String',['total slice number: ' num2str(sliceNumber)]);
set(handles.slider_imageIndex,'Min',1);
set(handles.slider_imageIndex,'Max',sliceNumber);
set(handles.slider_imageIndex,'Value',currentViewImageIndex);
set(handles.slider_imageIndex,'SliderStep',[1/(sliceNumber-1) 1]);

% --- Executes on button press in pushbutton_solectForground.
function pushbutton_solectForground_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_solectForground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global for_back_ground;
for_back_ground=1;

% --- Executes on button press in pushbutton_selectBackGound.
function pushbutton_selectBackGound_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_selectBackGound (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global for_back_ground;
for_back_ground=2;

% --- Executes on button press in pushbutton_segment.
function pushbutton_segment_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_segment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% segment_callback_Treebag(handles);
global slicSeg;
global currentViewImageIndex;
global ILabel;
slicSeg.Set('startIndex',currentViewImageIndex);
slicSeg.Set('seedImage',ILabel);
slicSeg.StartSliceSegmentation();
showResult(handles);

% --- Executes on button press in pushbutton_Propagate.
function pushbutton_Propagate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Propagate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% forward=true
global slicSeg;
% global processBar;
minSlice=str2num(get(handles.edit_min,'String'));
maxSlice=str2num(get(handles.edit_max,'String'));
slicSeg.Set('sliceRange',[minSlice,maxSlice]);

addlistener(slicSeg,'SegmentationProgress',@UpdateSegmentationProgressBar);
processBar = waitbar(0,'Please wait...');
slicSeg.SegmentationPropagate();
close(processBar) ;
showResult(handles);

function UpdateSegmentationProgressBar(eventSrc,eventData)
waitbar(eventData.OrgValue);



% --- Executes on button press in pushbutton_reload.
function pushbutton_reload_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global slicSeg;
global ILabel;
ILabel=uint8(zeros(size(ILabel)));
slicSeg.ResetSegmentationResult();
showResult(handles);

% --- Executes on slider movement.
function slider_imageIndex_Callback(hObject, eventdata, handles)
% hObject    handle to slider_imageIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global currentViewImageIndex;
currentViewImageIndex=round(get(handles.slider_imageIndex,'Value'));
set(handles.text_currentslice,'String',['current slice number: ' num2str(currentViewImageIndex)]);
showResult(handles);

% --- Executes during object creation, after setting all properties.
function slider_imageIndex_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_imageIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit_min_Callback(hObject, eventdata, handles)
% hObject    handle to edit_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_min as text
%        str2double(get(hObject,'String')) returns contents of edit_min as a double


% --- Executes during object creation, after setting all properties.
function edit_min_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_max_Callback(hObject, eventdata, handles)
% hObject    handle to edit_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_max as text
%        str2double(get(hObject,'String')) returns contents of edit_max as a double


% --- Executes during object creation, after setting all properties.
function edit_max_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
