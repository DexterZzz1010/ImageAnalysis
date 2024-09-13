### MATLAB ###

- Tested with MATLAB R2017b.
- `computeFLIP.m` and its subfunctions computes the FLIP map between two images. It's called from the `main.m` function,
  where you may load the reference and test images. Input images are supposed to be in sRGB space and in the [0,1] range.
- The default test and reference images are found in the `images` folder.
- The FLIP output is saved to the `images` folder.