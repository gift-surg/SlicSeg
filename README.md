Slic-Seg
--------

Slic-Seg is open-source software for semi-automated interactive segmentation of medical images.

Slic-Seg is a minimally interactive online learning-based segmentation method. An online random forest is first trained on data coming from *scribbles* provided by the user in one single selected start slice. This then forms the basis for a slice-by-slice framework that segments subsequent slices before incorporating them into the training set on the fly. 

Slic-Seg was developed as part of the [GIFT-Surg][giftsurg] project. The algorithm and software were developed by Guotai Wang at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

Algorithm and code by Guotai Wang.
If you use this software, please cite [this paper][citation]. 


Disclaimer
----------

 * This is PROTOTYPE software and may lead to system instability and crashes which could result in data loss.


Software links
--------------

 - [Slic-Seg home page][SlicSegHome].
 - [GitHub mirror][githubhome].

License
-----------

Copyright (c) 2014-2016, [University College London][ucl].

Slic-Seg is available as free open-source software under a BSD 3-Clause License.
Other licenses may apply for dependencies:
 - [Maxflow][maxflow] by Michael Rubinstein uses the BSD 2-Clause License
 - [CoreMat][coremat] by Tom Doel uses the MIT License
 - [DicoMat][dicomat] by Tom Doel uses the BSD 3-Clause License



System requirements
-------------------

The current version of Slic-Seg requries:
 * Matlab
 * Matlab Image Processing Toolbox (if files are to be loaded from DICOM)
 * A Matlab-supported C++ compiler installed and configured to work with mex files [see here](http://uk.mathworks.com/help/matlab/matlab_external/what-you-need-to-build-mex-files.html)
 * [The CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) to be installed and configured to work with Matlab


How to use
------------

 * Before attempting to use Slic-Seg, please ensure you have C++ and CUDA compilers installed and correctly configured to work with Matlab
   * [What you need to build mex files](http://uk.mathworks.com/help/matlab/matlab_external/what-you-need-to-build-mex-files.html)
   * [CUDA Installation guides](http://docs.nvidia.com/cuda/index.html#installation-guides)
   * [Why can't MEX find a supported compiler in MATLAB R2015b after I upgraded to Xcode 7.0?](http://uk.mathworks.com/matlabcentral/answers/246507-why-can-t-mex-find-a-supported-compiler-in-matlab-r2015b-after-i-upgraded-to-xcode-7-0)

 * GPU computing may lead to system instability and data loss. Please back up any valuable data before using the software.
 
 * Switch to the `matlab` directory.
 
 * To launch the user interface, run `slicseg` on the command window
 
 * Alternatively,  run the test script to illustrate use of the algorithm without the user interface. To do this:
  * Run 'SlicSegAddPaths` to set up the paths for this session
  * Type `test` in the command window
 


 
 
How to use the user interface
----------------------------
  * Run `slicseg` to launch the user interface.
  * The mex and cuda files will automatically compile if they have not already been compiled. This will fail if you have not installed and correctly set up your mex and cuda compilers to work with Matlab.
  * Click `Load` to load Dicom or a series of png image from a directory you specify
  * Choose your starting slice (usually a slice in the middle of the object)
  * Draw **scribbles** (lines) over parts of the object you wish to segment. The left button selects the foreground (object) and the right button selects the background.
    * The `Background` button makes the left button select background, while the `Foreground` button makes the left button select foreground
  * Click `Segment` to segment the object on this slice, based on the scribbes you have entered
  * Select the range (start and end slices) over which the segmentation will propagate
  * Click `Propagate` to continue the segmentation over these slices
  * Click `Save` to save the segmentation
 
 
Issues
------
 
  * The most likely issues will be due to not having correctly set up your mex and cuda compilers.
  * If you get compilation errors, please fix your mex and cuda compiler setup, then run `CompileSlicSeg recompile` on the command window to force re-compilation.
  * OSX users, please check the [supported versions of XCode](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#abstract). Note that NVIDIA CUDA may not support the latest versions of XCode.
  * If you are using OSX and receive a `No supported compiler or SDK was found` error, and you have already installed XCode, please [follows these instructions](   * [Why can't MEX find a supported compiler in MATLAB R2015b after I upgraded to Xcode 7.0?](http://uk.mathworks.com/matlabcentral/answers/246507-why-can-t-mex-find-a-supported-compiler-in-matlab-r2015b-after-i-upgraded-to-xcode-7-0)
)

  * On linux, linking problems may occur due to Matlab adding an internal linking path before `mex` is called. If you get version problems when linking C++ files you can force Matlab to find specific library versions using LD_PRELOAD, for example:
    ```LD_PRELOAD=/path-to-desired-library/libstdc++.so.6 /path-to-matlab/bin/matlab```   
  

Funding
-------------

This work was supported through an Innovative Engineering for Health award by the [Wellcome Trust][wellcometrust] [WT101957], the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] [NS/A000027/1] and a [National Institute for Health Research][nihr] Biomedical Research Centre [UCLH][uclh]/UCL High Impact Initiative.



Supported Platforms
-----------------------------

Slic-Seg is a cross-platform Matlab/C++ library. We have tested Slic-Seg on the following platforms:

 - Linux
  - Ubuntu Desktop 14.04.3 LTS 64-bit
  - NVIDIA 12GB GTX TITAN X
  - CUDA 7.5
  - Matlab R2015b
 
 - MacOS X
  - OS X Yosemite 10.10.5
  - NVIDIA GeForce GT 750M 1024 MB
  - XCode 7.2.1
  - CUDA 7.5
 
 - Windows
  - Not yet tested


Please note that NVIDA CUDA drivers and Matlab have specific compatibility requirements regarding C++ compilers

[tig]: http://cmictig.cs.ucl.ac.uk
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[nihr]: http://www.nihr.ac.uk/research
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[maxflow]: http://uk.mathworks.com/matlabcentral/fileexchange/21310-maxflow
[coremat]: http://github.com/tomdoel/coremat
[dicomat]: http://github.com/tomdoel/dicomat
[citation]: http://www.sciencedirect.com/science/article/pii/S1361841516300287
[SlicSegHome]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/SlicSeg
[githubhome]: https://github.com/gift-surg/SlicSeg
