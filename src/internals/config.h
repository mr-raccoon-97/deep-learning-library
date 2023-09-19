/**
Note: This file is used to configure the internal library.
Eigen is used as the default backend for the library, and should be included only
in the cpp files, not in the h or hpp files. 

This is for making the implementation of the operations independent of the internal
library. The internal library should be able to use any backend, and the user should
be able to choose the backend when compiling the library.
**/

#ifndef INTERNAL_CONFIG_H
#define INTERNAL_CONFIG_H

#define USE_EIGEN_BACKEND true

#if defined(USE_EIGEN_BACKEND)
    #include <eigen3/Eigen/Dense>
#endif // USE_EIGEN_BACKEND

#endif //INTERNAL_CONFIG_H