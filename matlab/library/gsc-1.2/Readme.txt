Matlab implementation of the segmentation algorithms used in:
Geodesic Star Convexity for Interactive Image Segmentation
V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman
In Proceedings of CVPR 2010

This code has been tested on Matlab R2008b. It does not work with Matlab 2007b (and below) due to lack of support of certain object oriented programming features. It should work on any version of Matlab >= R2008b.

This package contains the following files and directories:
1) setup.m: Execute this file to setup up paths properly before running any code.
2) compile_mex.m: Compiles all the mex files needed by different segmentation methods.
3) data: Directory for images and saving brush strokes and segmentations.
4) packages: This directory contains different packages for implementing all the algorithms.
5) ui: This directory contains code for the user interface.

To start using the code, follow these steps:
1) Run setup.m to setup paths properly (ignore warning about random walker code
2) Run compile_mex.m to compile the mex files
3) Type ui to start the user interface.

To use the user interface (once you have loaded it by calling ui.m), follow these steps:
1) Click on 'Load Image' button on the top left.
2) Choose the image you wish to segment from the dialog box and press Open. The image should now display in the UI. (3 example images are provided in the directory data/. The image shown on the webpage is data/cows.jpg)
3) Choose the segmentation algorithm from the drop down menu on the right. The naming of the segmentation algorithms is described later in the Readme.
4) Click on the 'PreProcess' button (near the top left) to do any pre-processing needed for the algorithm.
5) Now you can make foreground and background brush strokes. The brush type can be changed by choosing the appropriate stroke from the panel on the left called 'Stroke Label'. You can also load saved brush strokes by clicking on the 'Load labels' button (on the right side of the ui). To load the brush strokes shown on the webpage, click 'Load labels' and select cows-labels.png.
6) Once you have made both foreground and background strokes, click on the 'Start' button to get the first segmentation.
7) The user interface automatically switches to 'Auto Brush' mode after the first segmentation is obtained. This mode is useful for editing the obtained segmentation. Note that brush stroke is set to 'Auto' in the 'Stroke Label' panel now.
8) In the 'Auto Brush' mode, you dont need to specify the label of every brush stroke made. You can just brush inside the incorrect region and the brush stroke label is inferred by flipping the labels of the current segmentation. You can still choose a specific brush by changing the brush type from 'Auto' to 'FG' or 'BG' from the 'Stroke Label' panel.
9) When you make additional brush strokes, the segmentation gets updated after every brush stroke. To disable this feature, uncheck the check box 'Auto Update' located on the bottom left of the ui. Once disabled, you'll need to click the 'Update seg' button for the edits to take effect.
10) You can use buttons on the top right to save the segmentation('Save seg') or save the brush strokes ('Save labels').
11) You can run a different segmentation algorithm on the same set of brush strokes by clicking on 'Reload image' and starting from Step 3 again. ('Reload image' resets everything but the brush strokes).
12) You can change certain parameters of the system from the ui itself. The brush stroke size and the coherence parameter can be changed using the sliders on the bottom right. The geodesic parameter can be changed by typing its value in the text box and pressing enter (the updated value should display in the text label). To change other settings, look into the file ui/segOptions.m.

To test the robot user:
1) Run robotUser.runExample (it will run the user on a single image and show the sequence of brushes and segmentations).
2) See the file packages/+robotUser/runExample.m on how to use the robot user.

To run segmentation code without user interface
1) See the file packages/+miscFns/runOffline.m for an example (you can also run miscFns.runOffline to see it in action).

Key for segmentation algorithm names used in the 'Seg. algorithm' dropbox (see paper for full details):
BJ -> vanilla boykov jolly segmentation
PP -> post-processing to choose segmentation connected to fg strokes on top of BJ
ESC -> boykov jolly with euclidean star-convexity constraint
ESCseq -> boykov jolly with sequential euclidean star-convexity constraint
GSC -> boykov jolly with geodesic star-convexity constraint
GSCseq -> boykov jolly with sequential geodesic star-convexity constraint
RW -> Random walker implementation of Leo Grady
SP-IG/LIG/SIG -> different variants of the shortest path algorithms of Bai and Sapiro. Refer to paper for details.

Note:
To run Random Walker(RW) code, you'll have to download the graph analysis toolbox and random walker code from: http://cns.bu.edu/~lgrady/software.html and add these to the path (there is code in setup.m to add them to the path)
