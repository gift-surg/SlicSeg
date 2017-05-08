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
imageSegUIController = ImageSegUIController(handles.figure1, handles.axes_image);

% Cache the handles so we can update the controls when the slice number
% changes
glbHandles = handles;

% Listen for slice number change callbacks
addlistener(imageSegUIController, 'currentViewImageIndex', 'PostSet', @UpdateSliceNumber);
addlistener(imageSegUIController, 'guiState', 'PostSet', @UpdateGuiState);
addlistener(imageSegUIController, 'contrastMin', 'PostSet', @UpdateGuiContrastEdit);
addlistener(imageSegUIController, 'contrastMax', 'PostSet', @UpdateGuiContrastEdit);
UpdateGui(ImageSegUIState.NoImage);


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
set(handles.slider_imageIndex,'Value', max(1, min(maxSliceNumber, ceil(maxSliceNumber/2))));
set(handles.slider_imageIndex,'Min', 1);
set(handles.slider_imageIndex,'Max', maxSliceNumber);
set(handles.slider_imageIndex,'SliderStep',[1/max(1, maxSliceNumber-1) 1]);

% --- Executes on button press in pushbutton_reload.
function pushbutton_reload_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.reset();

% --- Executes on button press in pushbutton_reset_contrast.
function pushbutton_reset_contrast_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
contrastMin = str2num(get(handles.edit_contrastMin, 'String'));
contrastMax = str2num(get(handles.edit_contrastMax, 'String'));
imageSegUIController.resetContrast(contrastMin, contrastMax);


% --- Executes on button press in pushbutton_segment.
function pushbutton_segment_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_segment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.segment();


% --- Executes on button press in pushbutton_propagate.
function pushbutton_propagate_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Propagate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
minSlice=str2double(get(handles.edit_min,'String'));
maxSlice=str2double(get(handles.edit_max,'String'));
imageSegUIController.propagate(minSlice, maxSlice);

% --- Executes on button press in pushbutton_propagate.
function pushbutton_refine_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_Propagate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageSegUIController
imageSegUIController.refine();


% --- Executes on button press in pushbutton_loadImage.
function pushbutton_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_loadImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Save the segmented the images
global imageSegUIController
imageSegUIController.save();

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
    global imageSegUIController;
    global glbHandles;
    currentViewImageIndex = eventData.AffectedObject.currentViewImageIndex;
    maxSliceNumber = imageSegUIController.getMaxSliceNumber;
    set(glbHandles.slider_imageIndex, 'Value', currentViewImageIndex);
    set(glbHandles.text_currentslice, 'String', [num2str(currentViewImageIndex) '/' num2str(maxSliceNumber)]);

function UpdateGuiState(eventSrc, eventData)
    % Callback for when the controller has entered a different state
    UpdateGui(eventData.AffectedObject.guiState);

function UpdateGuiContrastEdit(eventSrc, eventData)
    global glbHandles,
    set(glbHandles.edit_contrastMin, 'String', num2str(eventData.AffectedObject.contrastMin));
    set(glbHandles.edit_contrastMax, 'String', num2str(eventData.AffectedObject.contrastMax));

function UpdateGui(newState)
    global glbHandles;
    loaded = 'off';
    scribbleProvided = 'off';
    startSliceSegmented = 'off';
    fullySegmented = 'off';
    if ~isempty(newState) && isa(newState, 'ImageSegUIState')
        switch newState
            case ImageSegUIState.NoImage
                loaded = 'off';
                scribbleProvided = 'off';
                startSliceSegmented = 'off';
                fullySegmented = 'off';
            case ImageSegUIState.ImageLoaded
                loaded = 'on';
                scribbleProvided = 'off';
                startSliceSegmented = 'off';
                fullySegmented = 'off';
            case ImageSegUIState.ScribblesProvided
                loaded = 'on';
                scribbleProvided = 'on';
                startSliceSegmented = 'off';
                fullySegmented = 'off';
            case ImageSegUIState.SliceSegmented
                loaded = 'on';
                scribbleProvided = 'on';
                startSliceSegmented = 'on';
                fullySegmented = 'off';
            case ImageSegUIState.FullySegmented
                loaded = 'on';
                scribbleProvided = 'on';
                startSliceSegmented = 'on';
                fullySegmented = 'on';
        end
    end
   
    set(glbHandles.slider_imageIndex, 'Enable', loaded);
    set(glbHandles.pushbutton_reload, 'Enable', loaded);
    set(glbHandles.edit_contrastMax, 'Enable', loaded);
    set(glbHandles.edit_contrastMin, 'Enable', loaded);
    set(glbHandles.pushbutton_reset_contrast, 'Enable', loaded);    
    set(glbHandles.pushbutton_segment, 'Enable', scribbleProvided);
    set(glbHandles.edit_max, 'Enable', startSliceSegmented);
    set(glbHandles.edit_min, 'Enable', startSliceSegmented);
    set(glbHandles.pushbutton_Propagate, 'Enable', startSliceSegmented);
    set(glbHandles.pushbutton_save, 'Enable', fullySegmented);
    
