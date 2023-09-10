# LineDetectionInPointCloud_CUDA
Transformada de Hough para detección de líneas en nubes de puntos en GPU usando CUDA

Este repositorio contiene dos carpetas.

CUDA_HOUGH contiene una implementación de la transformada de Hough en CUDA. Para poder ejecutar el código, es necesario instalar CUDA Toolkit, en concreto la versión 12 o posteriores, ya que no está verificado que el código funcione para versiones anteriores. También es neceario instalar OpenCV.

POINT_DATA_CLOUD_PROYECTION contiene un script de Python con el que se puede filtrar y proyectar una nube de puntos sobre un grid de resolución variable para obtener una imagen .png. Para poder ejecutar el código es necesario instalar previamente las librerías de Python pdal, numpy, matplotlib y PIL. Es preferible usar un entorno de anaconda con esas librerías instaladas.
