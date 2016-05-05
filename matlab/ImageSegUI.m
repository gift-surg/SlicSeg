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
global imageSegUIController
global glbHandles;

% Create a new UI controller which will manage the GUI state and mouse interaction
imageSegUIController = ImageSegUIController(gcf, handles.axes_image);

% Cache the handles so we can update the controls when the slice number
% changes
glbHandles = handles;

% Listen for slice number change callbacks
% addlistener(imageSegUIController, 'SliceNumberChanged', @UpdateSliceNumber);
addlistener(imageSegUIController, 'currentViewImageIndex', 'PostSet', @UpdateSliceNumber);


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

% Load the images
global imageSegUIController
imageSegUIController.selectAndLoad();

% Update the slice number and slider
maxSliceNumber = imageSegUIController.getMaxSliceNumber;
set(handles.text_totalslice,'String', ['Total number of slices: ' num2str(maxSliceNumber)]);
set(handles.slider_imageIndex,'Min', 1);
set(handles.slider_imageIndex,'Max', maxSliceNumber);
set(handles.slider_imageIndex,'SliderStep',[1/(maxSliceNumber-1) 1]);


% --- Executes on button press in pushbutton_solectForground.
function pushbutton_solectForground_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_solectForground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.selectForeground();


% --- Executes on button press in pushbutton_selectBackGound.
function pushbutton_selectBackGound_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_selectBackGound (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.selectBackground();


% --- Executes on button press in pushbutton_segment.
function pushbutton_segment_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_segment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.segment();


% --- Executes on button press in pushbutton_Propagate.
function pushbutton_Propagate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Propagate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
minSlice=str2int(get(handles.edit_min,'String'));
maxSlice=str2int(get(handles.edit_max,'String'));
imageSegUIController.propagate(minSlice, maxSlice);


% --- Executes on button press in pushbutton_reload.
function pushbutton_reload_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.reset();


% --- Executes on slider movement.
function slider_imageIndex_Callback(hObject, eventdata, handles)
% hObject    handle to slider_imageIndex (see GCBO)
% eventdata  reserved - to be defixned in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global imageSegUIController
imageSegUIController.currentViewImageIndex = round(get(handles.slider_imageIndex, 'Value'));


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


function UpdateSliceNumber(eventSrc, eventData)
    % Callback for when the controller's slice number has changed
    global glbHandles;
    currentViewImageIndex = eventData.AffectedObject.currentViewImageIndex;
    set(glbHandles.slider_imageIndex, 'Value', currentViewImageIndex);
    set(glbHandles.text_currentslice, 'String', ['Current slice number: ' num2str(currentViewImageIndex)]);



