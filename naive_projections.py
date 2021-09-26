# <<
# gcode stl?
object = [
   # X, Y, Z
    (3, 0, -2),
    (3, 0, -1),
    (3, 0, -10),
]

d = 2 # camera distance

projection = [ ((d/Z)*X, (d/Z)*Y) for X, Y, Z in object]
# TODO - sparse to dense
# imshow()

