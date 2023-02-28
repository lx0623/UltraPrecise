#include <string.h>
#include "./include/violite.h"
#include "server/mysql/include/mysql_def.h"

using namespace mysql;
/* Create a new VIO for socket or TCP/IP connection. */

static void vio_init(Vio *vio, enum enum_vio_type type,
                     my_socket sd, uint flags);
Vio *mysql_socket_vio_new(int connFd, enum enum_vio_type type, uint flags)
{
  Vio *vio;
  my_socket sd= connFd;
  if ( ( vio = (Vio*) malloc(sizeof(*vio)) ) )
  {
    vio_init(vio, type, sd, flags);
  }
  DBUG_RETURN(vio);
}


static my_bool has_no_data(Vio *vio )
{
  return FALSE;
}
/*
 * Helper to fill most of the Vio* with defaults.
 */

static void vio_init(Vio *vio, enum enum_vio_type type,
                     my_socket sd, uint flags)
{
  // DBUG_ENTER("vio_init");
  // DBUG_PRINT("enter", ("type: %d  sd: %d  flags: %d", type, sd, flags));

  memset(vio, 0, sizeof(*vio));
  vio->type= type;
  vio->mysql_socket= sd;
  vio->localhost= flags & VIO_LOCALHOST;
  vio->read_timeout= vio->write_timeout= -1;
  if ((flags & VIO_BUFFERED_READ) &&
      !(vio->read_buffer= (char*)malloc(VIO_READ_BUFFER_SIZE)))
    flags&= ~VIO_BUFFERED_READ;
    #ifdef HAVE_OPENSSL
  if (type == VIO_TYPE_SSL)
  {
    vio->viodelete	=vio_ssl_delete;
    vio->vioerrno	=vio_errno;
    vio->read		=vio_ssl_read;
    vio->write		=vio_ssl_write;
    vio->fastsend	=vio_fastsend;
    vio->viokeepalive	=vio_keepalive;
    vio->should_retry	=vio_should_retry;
    vio->was_timeout    =vio_was_timeout;
    vio->vioshutdown	=vio_ssl_shutdown;
    vio->peer_addr	=vio_peer_addr;
    vio->io_wait        =vio_io_wait;
    vio->is_connected   =vio_is_connected;
    vio->has_data       =vio_ssl_has_data;
    vio->timeout        =vio_socket_timeout;
    DBUG_VOID_RETURN;
  }
#endif /* HAVE_OPENSSL */
  vio->viodelete        =vio_delete;
  vio->vioerrno         =vio_errno;
  vio->read=            (flags & VIO_BUFFERED_READ) ? vio_read_buff : vio_read;
  vio->write            =vio_write;
  vio->fastsend         =vio_fastsend;
  vio->viokeepalive     =vio_keepalive;
  vio->should_retry     =vio_should_retry;
  vio->was_timeout      =vio_was_timeout;
  vio->vioshutdown      =vio_shutdown;
  vio->peer_addr        =vio_peer_addr;
  vio->io_wait          =vio_io_wait;
  vio->is_connected     =vio_is_connected;
  vio->timeout          =vio_socket_timeout;
  vio->has_data=        (flags & VIO_BUFFERED_READ) ?
                            vio_buff_has_data : has_no_data;
  DBUG_VOID_RETURN;
}

void vio_delete(Vio* vio)
{
  if (!vio)
    return; /* It must be safe to delete null pointers. */

  if (vio->inactive == FALSE)
    vio->vioshutdown(vio);
  free(vio->read_buffer);
  free(vio);
}