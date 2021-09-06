
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
[x] alpha/beta/gamma - play with manually using polzunok
[x] histogram
[ ] 2D histogram
[ ] histogram equalization < preprocessing
[ ] thresholding, adaptive thresholding
[ ] FreeCAD - make a box with primitives?




face
target


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



