## For contributors
### If you want to add a function or an operation to the library follow the next steps.
* In the src/internal/functions add to the internal_functions.h file the function you want to add.
  It needs to inherit from the Function class, and overrides the forward and the backward methods as in the examples provided.
* Create a cpp file following the naming convention of the library, and implement the function there.
* Add the function you created into the src/functions.cpp file and in the include/CaberNet/functions.h file.
