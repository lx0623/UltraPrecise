#ifndef VIOLITE_INCLUDED
#define VIOLITE_INCLUDED 
 #include <netinet/in.h>
 #include "my_global.h"
 #include "mysql_com.h"

typedef struct st_vio Vio;
enum enum_vio_type
{
  /**
    Type of the connection is unknown.
  */
  NO_VIO_TYPE= 0,
  /**
    Used in case of TCP/IP connections.
  */
  VIO_TYPE_TCPIP= 1,
  /**
    Used for Unix Domain socket connections. Unix only.
  */
  VIO_TYPE_SOCKET= 2,
  /**
    Used for named pipe connections. Windows only.
  */
  VIO_TYPE_NAMEDPIPE= 3,
  /**
    Used in case of SSL connections.
  */
  VIO_TYPE_SSL= 4,
  /**
    Used for shared memory connections. Windows only.
  */
  VIO_TYPE_SHARED_MEMORY= 5,
  /**
    Used internally by the prepared statements
  */
  VIO_TYPE_LOCAL= 6,
  /**
    Implicitly used by plugins that doesn't support any other VIO_TYPE.
  */
  VIO_TYPE_PLUGIN= 7,

  FIRST_VIO_TYPE= VIO_TYPE_TCPIP,
  /*
    If a new type is added, please update LAST_VIO_TYPE. In addition, please
    change get_vio_type_name() in vio/vio.c to return correct name for it.
  */
  LAST_VIO_TYPE= VIO_TYPE_PLUGIN
};

/**
  VIO I/O events.
*/
enum enum_vio_io_event
{
  VIO_IO_EVENT_READ,
  VIO_IO_EVENT_WRITE,
  VIO_IO_EVENT_CONNECT
};

#define VIO_LOCALHOST 1                         /* a localhost connection */
#define VIO_BUFFERED_READ 2                     /* use buffered read */
#define VIO_READ_BUFFER_SIZE 16384              /* size of read buffer */
#define VIO_DESCRIPTION_SIZE 30                 /* size of description */
void    vio_delete(Vio* vio);
int vio_shutdown(Vio* vio);
my_bool vio_reset(Vio* vio, enum enum_vio_type type,
                  my_socket sd, void *ssl, uint flags);
size_t  vio_read(Vio *vio, uchar *	buf, size_t size);
size_t  vio_read_buff(Vio *vio, uchar * buf, size_t size);
size_t  vio_write(Vio *vio, const uchar * buf, size_t size);
/* setsockopt TCP_NODELAY at IPPROTO_TCP level, when possible */
int vio_fastsend(Vio *vio);
/* setsockopt SO_KEEPALIVE at SOL_SOCKET level, when possible */
int vio_keepalive(Vio *vio, my_bool	onoff);
/* Whenever we should retry the last read/write operation. */
my_bool vio_should_retry(Vio *vio);
/* Check that operation was timed out */
my_bool vio_was_timeout(Vio *vio);
/* Short text description of the socket for those, who are curious.. */
const char* vio_description(Vio *vio);
/* Return the type of the connection */
enum enum_vio_type vio_type(Vio* vio);
/* Return last error number */
int	vio_errno(Vio*vio);
/* Get socket number */
my_socket vio_fd(Vio*vio);
/* Remote peer's address and name in text form */
my_bool vio_peer_addr(Vio *vio, char *buf, uint16 *port, size_t buflen);
/* Wait for an I/O event notification. */
int vio_io_wait(Vio *vio, enum enum_vio_io_event event, int timeout);
my_bool vio_is_connected(Vio *vio);
int vio_socket_timeout(Vio *vio,
                       uint which ,
                       my_bool old_mode);
my_bool vio_buff_has_data(Vio *vio);
void my_net_set_read_timeout(NET *net, uint timeout);
void my_net_set_write_timeout(NET *net, uint timeout);
#ifndef DBUG_OFF
ssize_t vio_pending(Vio *vio);
#endif
/* Set timeout for a network operation. */
int vio_timeout(Vio *vio, uint which, int timeout_sec);
/* Connect to a peer. */
my_bool vio_socket_connect(Vio *vio, struct sockaddr *addr, socklen_t len,
                           int timeout);

my_bool vio_get_normalized_ip_string(const struct sockaddr *addr, size_t addr_length,
                                     char *ip_string, size_t ip_string_size);

my_bool vio_is_no_name_error(int err_code);

int vio_getnameinfo(const struct sockaddr *sa,
                    char *hostname, size_t hostname_size,
                    char *port, size_t port_size,
                    int flags);

/**
  Convert a vio type to a printable string.
  @param vio_type the type
  @param[out] str the string
  @param[out] len the string length
*/
void get_vio_type_name(enum enum_vio_type vio_type, const char ** str, int * len);


/* HFTODO - hide this if we don't want client in embedded server */
/* This structure is for every connection on both sides */
struct st_vio
{
  int  mysql_socket;           /* Instrumented socket */
  // MYSQL_SOCKET  mysql_socket;           /* Instrumented socket */
  int       localhost;              /* Are we from localhost? */
  struct sockaddr_storage   local;      /* Local internet address */
  struct sockaddr_storage   remote;     /* Remote internet address */
  size_t addrLen;                       /* Length of remote address */
  enum enum_vio_type    type;           /* Type of connection */
  int               inactive; /* Connection inactive (has been shutdown) */
  char                  desc[VIO_DESCRIPTION_SIZE]; /* Description string. This
                                                      member MUST NOT be
                                                      used directly, but only
                                                      via function
                                                      "vio_description" */
  char                  *read_buffer;   /* buffer for vio_read_buff */
  char                  *read_pos;      /* start of unfetched data in the
                                           read buffer */
  char                  *read_end;      /* end of unfetched data */
  int                   read_timeout;   /* Timeout value (ms) for read ops. */
  int                   write_timeout;  /* Timeout value (ms) for write ops. */
  
  /* 
     VIO vtable interface to be implemented by VIO's like SSL, Socket,
     Named Pipe, etc.
  */
  
  /* 
     viodelete is responsible for cleaning up the VIO object bygg freeing 
     internal buffers, closing descriptors, handles. 
  */
  void    (*viodelete)(Vio*);
  int     (*vioerrno)(Vio*);
  size_t  (*read)(Vio*, uchar *, size_t);
  size_t  (*write)(Vio*, const uchar *, size_t);
  int     (*timeout)(Vio*, uint, my_bool);
  int     (*viokeepalive)(Vio*, my_bool);
  int     (*fastsend)(Vio*);
  my_bool (*peer_addr)(Vio*, char *, uint16*, size_t);
  void    (*in_addr)(Vio*, struct sockaddr_storage*);
  my_bool (*should_retry)(Vio*);
  my_bool (*was_timeout)(Vio*);
  /* 
     vioshutdown is resposnible to shutdown/close the channel, so that no 
     further communications can take place, however any related buffers,
     descriptors, handles can remain valid after a shutdown.
  */
  int     (*vioshutdown)(Vio*);
  my_bool (*is_connected)(Vio*);
  my_bool (*has_data) (Vio*);
  int (*io_wait)(Vio*, enum enum_vio_io_event, int);
  my_bool (*connect)(Vio*, struct sockaddr *, socklen_t, int);
#ifdef _WIN32
  OVERLAPPED overlapped;
  HANDLE hPipe;
#endif
#ifdef HAVE_OPENSSL
  void    *ssl_arg;
#endif
#if defined (_WIN32) && !defined (EMBEDDED_LIBRARY)
  HANDLE  handle_file_map;
  char    *handle_map;
  HANDLE  event_server_wrote;
  HANDLE  event_server_read;
  HANDLE  event_client_wrote;
  HANDLE  event_client_read;
  HANDLE  event_conn_closed;
  size_t  shared_memory_remain;
  char    *shared_memory_pos;
#endif /* _WIN32 && !EMBEDDED_LIBRARY */
};
#endif