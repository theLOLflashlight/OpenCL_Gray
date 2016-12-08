Andrew Meckling
Nav Bhatti

Converts an image to grayscale.

Our serial implementation runs faster than the others on release build.
This is most likely due to the fact that the serial version doesn't have 
to copy buffers between devices. (And all we do is convert to grayscale.)

Tried to implement a gaussain blur, but it only worked for blur radius of 
aprox 5 px. (Couldn't make the implementation robust to varying blur radii.)