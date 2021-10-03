
# CV practice

TODO

- learn some python graphics (opencv) thing
    - load an image
    - mesh it
- load in Freecad, add 5mm z thickness (might be harder than it sounds)
- print out
- ???
- profit

Estimate: 1-3 months

# nifties

background subtraction
edge detection
object detection


# current projects

[x] put batman in spotlight - hello world
[x] tanchik - basic drill
[x] polzunok - set up a UI for manual image preprocessing
[x] Peshka! side project
[x] alpha/beta/gamma - play with manually using polzunok
[x] histogram
[x] 2D histogram
[ ] histogram equilization techniques article
[ ] thresholding
[ ] adaptive thresholding
[ ] FreeCAD - make a box with primitives?

[ ] PSNR - peak signal to noise ratio
[ ] histogram equalization < preprocessing
[ ] Kalman filter?
    - adaptive magic thingamagik

[ ] Computer Vision Modern approach 1st chapter




# background learning
https://scikit-image.org/docs/stable/user_guide/numpy_images.html
https://numpy.org/numpy-tutorials/content/tutorial-ma.html

https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html

## gamma


+-----------------------------------------------+
|gamma between 0 and 1 | encoding / compression |
|-----------------------------------------------|
|gamma > 1             | decoding / expansion   |
+-----------------------------------------------+

gamma of 2.2 is used to convert between true linear intensity
and human perception of linear. and i can never tell which is which.
I think humans see linear as crushed shadows? (linear ^ 2.2)
or overblown highlights?


sRGB - almost the same thing, but more linear near zero.

basically,
- images mostly come gamma-encoded to account for human vision
- decode in order to convert to linear space prior to processing
- encode back in the end



TODO - find native srgb conversion facility in opencv.


resize
blur
antialias

erosion/dilation
filtering

histogram equalization

http://blog.johnnovak.net/2016/09/21/what-every-coder-should-know-about-gamma/
https://github.com/opencv/opencv/issues/18813


completely unrelated thing, but

# Harris corner detector



# nifties

https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
https://wiki.freecadweb.org/Python_scripting_tutorial
https://numpy.org/doc/stable/reference/generated/numpy.clip.html?highlight=np%20clip
https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html

# FreeCAD

It's not clear how to pip-install it.

https://community.chocolatey.org/packages/freecad

https://github.com/FreeCAD/FreeCAD/releases

conda install maybe?

# cadquery

https://cadquery.readthedocs.io/en/latest/intro.html

https://github.com/CadQuery/cadquery


OpenCascade, CadQuery, what?

> The original version of CadQuery was built on the FreeCAD API. This was great because it allowed for fast development and easy cross-platform capability. However, we eventually started reaching the limits of the API for some advanced operations and selectors. This 2.0 version of CadQuery is based directly on a Python wrapper of the OCCT kernel.

install is through conda

(miniconda)[https://docs.conda.io/en/latest/miniconda.html]



