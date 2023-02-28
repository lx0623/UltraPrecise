#ifndef VIO_INCLUDED
#define VIO_INCLUDED
#include "violite.h"
Vio *create_and_init_vio(int connFd) ;
Vio *mysql_socket_vio_new(int mysql_socket, enum enum_vio_type type, uint flags);

#endif