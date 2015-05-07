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

% Last Modified by GUIDE v2.5 12-Mar-2015 10:17:37

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
global I;
global mouse_state;    %0--not pressed,1--pressed
global for_back_ground;%0--null,1--forground,2-background
global fileName;
global pathName;
global startImageIndex;
global currentImageIndex;
global lambda;
global simga;


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
global mouse_state; 
global for_back_ground;
global lambda;
global lambdaoff;
global sigma;
global sigmaoff;
mouse_state=0;
for_back_ground=0;
lambda=4.80;
sigma=3.5;
set(handles.slider1,'Value',lambda);
set(handles.text1,'String',['lambda=' num2str(lambda)]);
set(handles.slider2,'Value',sigma);
set(handles.text2,'String',['sigma=' num2str(sigma)]);


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
global I;
global ILabel;
global fileNames;
global imgFolderName;
global startImageIndex;
global currentImageIndex;
global ExistingTrainingSet;
global ExistingTrainingLabel;
global segSaveFolder;
reset(handles.axes_image);
reset(handles.axes_seg);
reset(handles.axes_gcseg);
reset(handles.axes_geosseg);
reset(handles.axes_segoffline);
cla(handles.axes_image);
cla(handles.axes_seg);
cla(handles.axes_gcseg);
cla(handles.axes_geosseg);
cla(handles.axes_segoffline);
set(gcf,'WindowButtonDownFcn',{@mouse_down});
set(gcf,'WindowButtonMotionFcn',{@mouse_move});
set(gcf,'WindowButtonUpFcn',{@mouse_up});
% imgFolderName=uigetdir;
[startFileName,imgFolderName,FilterIndex] = uigetfile('*.png');
dirinfo=dir(fullfile(imgFolderName,'*.png'));
fileNumber=length(dirinfo);
filenameLen=length(startFileName);
startImageIndex=str2num(startFileName(1:filenameLen-4));
currentImageIndex=startImageIndex;
fileNames={};
for i=1:fileNumber
    fileNames{i}=dirinfo(i).name;
end
segSaveFolder='';
% startImageIndex=floor(fileNumber/2);

longfilename=fullfile(imgFolderName,[num2str(currentImageIndex) '.png']);
I=imread(longfilename);
Isize=size(I);
ILabel=uint8(zeros(Isize));
axes(handles.axes_image);
imshow(I);
ExistingTrainingSet=[];
ExistingTrainingLabel=[];


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
segment_callback_Treebag(handles);


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global lambda;
slider_value = get(handles.slider1,'Value');
lambda=slider_value;
str=['lambda=' num2str(lambda)];
set(handles.text1,'String',str);
maxflow_callback(handles);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global sigma;
slider_value = get(handles.slider2,'Value');
sigma=slider_value;
str=['sigma=' num2str(sigma)];
set(handles.text2,'String',str);
maxflow_callback(handles);
% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton_Next.
function pushbutton_Next_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% forward=true
next_slice_callback(handles,true);

% --- Executes on button press in pushbutton_Previous.
function pushbutton_Previous_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Previous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% forward=false, segmentation will propagate backward
next_slice_callback(handles,false);

% --- Executes on button press in pushbutton_Retrain.
function pushbutton_Retrain_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Retrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
retrain_callback(handles);


% --- Executes on button press in pushbutton_gotostart.
function pushbutton_gotostart_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_gotostart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
goToStart_callback(handles);


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes on button press in pushbutton_reload.
function pushbutton_reload_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global I;
global ILabel;
global ExistingTrainingSet;
global ExistingTrainingLabel;
global currentImageIndex;
global startImageIndex;
global imgFolderName;
global segSaveFolder;
segSaveFolder='';
cla(handles.axes_seg);
cla(handles.axes_gcseg);
cla(handles.axes_geosseg);
cla(handles.axes_segoffline);
currentImageIndex=startImageIndex;
longfilename=fullfile(imgFolderName,[num2str(currentImageIndex) '.png']);
I=imread(longfilename);
Isize=size(I);
ILabel=uint8(zeros(Isize));
axes(handles.axes_image);
imshow(I);




% --- Executes on button press in checkbox_autosave.
function checkbox_autosave_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_autosave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_autosave



function edit_username_Callback(hObject, eventdata, handles)
% hObject    handle to edit_username (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_username as text
%        str2double(get(hObject,'String')) returns contents of edit_username as a double


% --- Executes during object creation, after setting all properties.
function edit_username_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_username (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_save.s
function pushbutton_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
saveFrame_callback(handles,true,true);


% --- Executes on button press in pushbutton_loadlabel.
function pushbutton_loadlabel_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_loadlabel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
loadLabel_callback(handles);


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global lambdaoff;
slider_value = get(handles.slider3,'Value');
lambdaoff=slider_value;
str=['lambda=' num2str(lambdaoff)];
set(handles.text10,'String',str);
maxflow_callback(handles);

% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global sigmaoff;
slider_value = get(handles.slider4,'Value');
sigmaoff=slider_value;
str=['sigma=' num2str(sigmaoff)];
set(handles.text11,'String',str);
maxflow_callback(handles);

% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider6_Callback(hObject, eventdata, handles)%lambda
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global lambda;
slider_value = get(handles.slider6,'Value');
lambda=slider_value;
str=['lambda=' num2str(lambda)];
set(handles.text12,'String',str);
maxflow_callback(handles,1);

% --- Executes during object creation, after setting all properties.
function slider6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles) %seg_with_lambda
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global lambda;
username=get(handles.edit_username,'String');
if(username=='lambda')
    lambdastr=get(handles.edit3,'String');
    disp(['lambda=' lambdastr]);
    lambda=str2num(lambdastr);
end
maxflow_callback(handles,1);


% --- Executes on button press in pushbutton_off_seg.
function pushbutton_off_seg_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_off_seg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
offline_segment_callback(handles);

% --- Executes on button press in pushbutton_offline_segment_save.
function pushbutton_offline_segment_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_offline_segment_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
offline_segment_save(handles);


% --- Executes on button press in pushbutton_propagate.
function pushbutton_propagate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_propagate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
offline_segment_propagation(handles);
