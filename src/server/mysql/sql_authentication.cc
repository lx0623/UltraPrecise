#include <string.h>
#include <glog/logging.h>

#include "./include/mysqld.h"
#include "./include/sql_string.h"
#include "./include/protocol_classic.h"
#include "./include/mysql_com.h"
#include "./include/mysql_version.h"
#include "./include/my_command.h"
#include "./include/plugin_auth_common.h"
#include "./include/m_string.h"
#include "./include/sql_authentication.h"
#include "./include/crypt_genhash_impl.h"
#include "./include/my_byteorder.h"
#include "./include/violite.h"
#include "server/mysql/include/derror.h"
#include "server/mysql/include/password.h"
#include "server/mysql/include/sql_auth_cache.h"

char server_version[SERVER_VERSION_LENGTH];

/** Size of the header fields of an authentication packet. */
#define AUTH_PACKET_HEADER_SIZE_PROTO_41    32
#define AUTH_PACKET_HEADER_SIZE_PROTO_40    5

bool mysql_change_db(THD *thd, const string &new_db_name, bool force_switch);
bool net_send_error(THD *thd, uint sql_errno, const char *err);

/****************************************************************************
   AUTHENTICATION CODE
   including initial connect handshake, invoking appropriate plugins,
   client-server plugin negotiation, COM_CHANGE_USER, and native
   MySQL authentication plugins.
****************************************************************************/

LEX_CSTRING native_password_plugin_name= {
  C_STRING_WITH_LEN( "mysql_native_password" )
};
  
LEX_CSTRING sha256_password_plugin_name= {
  C_STRING_WITH_LEN("sha256_password")
};
// 
// LEX_CSTRING validate_password_plugin_name= {
//   C_STRING_WITH_LEN("validate_password")
// };

LEX_CSTRING default_auth_plugin_name = native_password_plugin_name;

size_t copy_and_convert(char *to, size_t to_length,
                        const CHARSET_INFO *to_cs,
                        const char *from, size_t from_length,
                        const CHARSET_INFO *from_cs, uint *errors);
bool thd_init_client_charset(THD *thd, uint cs_number);

static size_t parse_client_handshake_packet(MPVIO_EXT *mpvio,
                                            uchar **buff, size_t pkt_len);
static bool send_server_handshake_packet(MPVIO_EXT *mpvio,
                                         const char *data, uint data_len);
Vio *mysql_socket_vio_new(int mysql_socket, enum enum_vio_type type, uint flags);

static int server_mpvio_read_packet(MYSQL_PLUGIN_VIO *param, uchar **buf);
static int server_mpvio_write_packet(MYSQL_PLUGIN_VIO *param,
                                     const uchar *packet, int packet_len);

                                     bool

Proc_charset_adapter::init_client_charset(uint cs_number)
{
  if (thd_init_client_charset(thd, cs_number))
    return true;
  // thd->update_charset();
  // return thd->is_error();
  return false;
}

const CHARSET_INFO *
Proc_charset_adapter::charset()
{
  return thd->charset();
}

/**
  sends a "change plugin" packet, requesting a client to restart authentication
  using a different authentication plugin

  Packet format:
   
    Bytes       Content
    -----       ----
    1           byte with the value 254
    n           client plugin to use, \0-terminated
    n           plugin provided data

  @retval 0 ok
  @retval 1 error
*/
static bool send_plugin_request_packet(MPVIO_EXT *mpvio,
                                       const uchar *data, uint data_len)
{
  DBUG_ASSERT(mpvio->packets_written == 1);
  DBUG_ASSERT(mpvio->packets_read == 1);
  static uchar switch_plugin_request_buf[]= { 254 };

  // DBUG_ENTER("send_plugin_request_packet");
  mpvio->status= MPVIO_EXT::FAILURE; // the status is no longer RESTART

  const char *client_auth_plugin= mpvio->plugin.data();
    // ((st_mysql_auth *) (plugin_decl(mpvio->plugin)->info))->client_auth_plugin;

  DBUG_ASSERT(client_auth_plugin);

  /*
    If we're dealing with an older client we can't just send a change plugin
    packet to re-initiate the authentication handshake, because the client 
    won't understand it. The good thing is that we don't need to : the old client
    expects us to just check the user credentials here, which we can do by just reading
    the cached data that are placed there by parse_com_change_user_packet() 
    In this case we just do nothing and behave as if normal authentication
    should continue.
  */
  if (!(mpvio->protocol->has_client_capability(CLIENT_PLUGIN_AUTH)))
  {
    // DBUG_PRINT("info", ("old client sent a COM_CHANGE_USER"));
    DBUG_ASSERT(mpvio->cached_client_reply.pkt);
    /* get the status back so the read can process the cached result */
    mpvio->status= MPVIO_EXT::RESTART; 
    DBUG_RETURN(0);
  }

  // DBUG_PRINT("info", ("requesting client to use the %s plugin", 
  //                     client_auth_plugin));
  DBUG_RETURN(net_write_command(mpvio->protocol->get_net(),
                                switch_plugin_request_buf[0],
                                (uchar*) client_auth_plugin,
                                strlen(client_auth_plugin) + 1,
                                (uchar*) data, data_len));
}

bool MPVIO_EXT::can_authenticate()
{
  return (acl_user && acl_user->can_authenticate);
}

void
server_mpvio_initialize(THD* thd, MPVIO_EXT *mpvio, Proc_charset_adapter *charset_adapter)
{
  // LEX_CSTRING sctx_host_or_ip= thd->security_context()->host_or_ip();

  // memset(mpvio, 0, sizeof(MPVIO_EXT));
  mpvio->read_packet= server_mpvio_read_packet;
  mpvio->write_packet= server_mpvio_write_packet;
  // mpvio->info= server_mpvio_info;
  mpvio->auth_info.host_or_ip= thd->peer_host.data();
  mpvio->auth_info.host_or_ip_length= thd->peer_host.length();

// #if defined(HAVE_OPENSSL) && !defined(EMBEDDED_LIBRARY)
//   Vio *vio= thd->get_protocol_classic()->get_vio();
//   if (vio->ssl_arg)
//     mpvio->vio_is_encrypted= 1;
//   else
// #endif /* HAVE_OPENSSL && !EMBEDDED_LIBRARY */
  // mpvio->mem_root= thd->mem_root;
  mpvio->scramble= thd->scramble;
  // mpvio->rand= &thd->rand;
  mpvio->thread_id= thd->thread_id();
  mpvio->server_status= &thd->server_status;
  mpvio->protocol= thd->get_protocol_classic();
  // mpvio->ip= (char *) thd->security_context()->ip().str;
  // mpvio->host= (char *) thd->security_context()->host().str;
  mpvio->charset_adapter= charset_adapter;
}

/**
  MySQL Server Password Authentication Plugin

  In the MySQL authentication protocol:
  1. the server sends the random scramble to the client
  2. client sends the encrypted password back to the server
  3. the server checks the password.
*/
// int native_password_authenticate(THD* thd)
// {
//   MPVIO_EXT mpvio;
//   Proc_charset_adapter charset_adapter(thd);
//   server_mpvio_initialize(thd, &mpvio, &charset_adapter);
//   thd->set_db("");
// 
//   // LEX_CSTRING auth_plugin_name = native_password_plugin_name;
//   mpvio.plugin = native_password_plugin_name.str;
// 
//   thd->get_protocol_classic()->get_net()->read_pos=
//     thd->get_protocol_classic()->get_raw_packet();
// 
//   int old_status = mpvio.status;
// 
//   /* scramble - random string sent to client on handshake */
//   generate_user_salt(mpvio.scramble, SCRAMBLE_LENGTH + 1);
//   if (mpvio.write_packet(&mpvio, (uchar *)mpvio.scramble, SCRAMBLE_LENGTH + 1)) {
//       DBUG_RETURN(CR_AUTH_HANDSHAKE);
//   }
// 
//   uchar *pkt;
//   int pkt_len;
//   /* read the reply with the encrypted password */
//   if ((pkt_len= mpvio.read_packet(&mpvio, &pkt)) < 0)
//     DBUG_RETURN(CR_AUTH_HANDSHAKE);
// 
//   if (NULL != mpvio.auth_info.user_name)
//   {
//     thd->set_user_name(mpvio.auth_info.user_name, mpvio.auth_info.user_name_length);
//     free(mpvio.auth_info.user_name);
//     mpvio.auth_info.user_name = NULL;
//   }
//   if (mpvio.cached_server_packet.pkt) {
//       free(mpvio.cached_server_packet.pkt);
//       mpvio.cached_server_packet.pkt = NULL;
//   }
//     /* Change a database if necessary */
//     if (!mpvio.db.empty()) {
//         if (mysql_change_db(thd, mpvio.db, false)) {
//             DBUG_RETURN(1);
//         }
//     }
// 
//   // if (pkt_len == 0) /* no password */
//     // DBUG_RETURN(mpvio->acl_user->salt_len != 0 ?
//     //             CR_AUTH_USER_CREDENTIALS : CR_OK);
//     return CR_OK;
//   // return 0;
//   //  DBUG_RETURN(CR_AUTH_HANDSHAKE);
// }

/**
  MySQL Server Password Authentication Plugin

  In the MySQL authentication protocol:
  1. the server sends the random scramble to the client
  2. client sends the encrypted password back to the server
  3. the server checks the password.
*/
static int native_password_authenticate(MYSQL_PLUGIN_VIO *vio,
                                        MYSQL_SERVER_AUTH_INFO *info)
{
  uchar *pkt;
  int pkt_len;
  MPVIO_EXT *mpvio= (MPVIO_EXT *) vio;

  DBUG_ENTER("native_password_authenticate");

  /* generate the scramble, or reuse the old one */
  if (mpvio->scramble[SCRAMBLE_LENGTH])
    generate_user_salt(mpvio->scramble, SCRAMBLE_LENGTH + 1);

  /* send it to the client */
  if (mpvio->write_packet(mpvio, (uchar*) mpvio->scramble, SCRAMBLE_LENGTH + 1))
    DBUG_RETURN(CR_AUTH_HANDSHAKE);

  /* reply and authenticate */

  /*
    <digression>
      This is more complex than it looks.

      The plugin (we) may be called right after the client was connected -
      and will need to send a scramble, read reply, authenticate.

      Or the plugin may be called after another plugin has sent a scramble,
      and read the reply. If the client has used the correct client-plugin,
      we won't need to read anything here from the client, the client
      has already sent a reply with everything we need for authentication.

      Or the plugin may be called after another plugin has sent a scramble,
      and read the reply, but the client has used the wrong client-plugin.
      We'll need to sent a "switch to another plugin" packet to the
      client and read the reply. "Use the short scramble" packet is a special
      case of "switch to another plugin" packet.

      Or, perhaps, the plugin may be called after another plugin has
      done the handshake but did not send a useful scramble. We'll need
      to send a scramble (and perhaps a "switch to another plugin" packet)
      and read the reply.

      Besides, a client may be an old one, that doesn't understand plugins.
      Or doesn't even understand 4.0 scramble.

      And we want to keep the same protocol on the wire  unless non-native
      plugins are involved.

      Anyway, it still looks simple from a plugin point of view:
      "send the scramble, read the reply and authenticate"
      All the magic is transparently handled by the server.
    </digression>
  */

  /* read the reply with the encrypted password */
  if ((pkt_len= mpvio->read_packet(mpvio, &pkt)) < 0)
    DBUG_RETURN(CR_AUTH_HANDSHAKE);
  // DBUG_PRINT("info", ("reply read : pkt_len=%d", pkt_len));

#ifdef NO_EMBEDDED_ACCESS_CHECKS
  DBUG_RETURN(CR_OK);
#endif /* NO_EMBEDDED_ACCESS_CHECKS */

  // DBUG_EXECUTE_IF("native_password_bad_reply",
  //                 {
  //                   /* This should cause a HANDSHAKE ERROR */
  //                   pkt_len= 12;
  //                 }
  //                 );
  // if (mysql_native_password_proxy_users)
  // {
  //   *info->authenticated_as= PROXY_FLAG;
	// DBUG_PRINT("info", ("mysql_native_authentication_proxy_users is enabled, setting authenticated_as to NULL"));
  // }
  if (pkt_len == 0) /* no password */
    DBUG_RETURN(mpvio->acl_user->salt_len != 0 ?
                CR_AUTH_USER_CREDENTIALS : CR_OK);

  info->password_used= PASSWORD_USED_YES;
  if (pkt_len == SCRAMBLE_LENGTH)
  {
    if (!mpvio->acl_user->salt_len)
      DBUG_RETURN(CR_AUTH_USER_CREDENTIALS);

    DBUG_RETURN(check_scramble(pkt, mpvio->scramble, mpvio->acl_user->salt) ?
                CR_AUTH_USER_CREDENTIALS : CR_OK);
  }

  my_error(ER_HANDSHAKE_ERROR, MYF(0));
  DBUG_RETURN(CR_AUTH_HANDSHAKE);
}

static int do_auth_once(THD *thd, const LEX_CSTRING &auth_plugin_name,
                        MPVIO_EXT *mpvio)
{
  DBUG_ENTER("do_auth_once");
  int res= CR_OK, old_status= MPVIO_EXT::FAILURE;
  // bool unlock_plugin= false;
  // plugin_ref plugin= NULL;

//   if (auth_plugin_name.str == native_password_plugin_name.str)
//     plugin= native_password_plugin;
// #ifndef EMBEDDED_LIBRARY
//   else
//   {
//     if ((plugin= my_plugin_lock_by_name(thd, auth_plugin_name,
//                                         MYSQL_AUTHENTICATION_PLUGIN)))
//       unlock_plugin= true;
//   }
// #endif /* EMBEDDED_LIBRARY */

  mpvio->plugin= auth_plugin_name.str;
  old_status= mpvio->status;

   if (auth_plugin_name.str != native_password_plugin_name.str)
   {
      my_error(ER_NOT_SUPPORTED_AUTH_MODE, MYF(0));
      res= CR_ERROR;
   }
   else
   {
    res = native_password_authenticate( mpvio, &mpvio->auth_info );
   }
  
  /*
    If the status was MPVIO_EXT::RESTART before the authenticate_user() call
    it can never be MPVIO_EXT::RESTART after the call, because any call
    to write_packet() or read_packet() will reset the status.

    But (!) if a plugin never called a read_packet() or write_packet(), the
    status will stay unchanged. We'll fix it, by resetting the status here.
  */
  if (old_status == MPVIO_EXT::RESTART && mpvio->status == MPVIO_EXT::RESTART)
    mpvio->status= MPVIO_EXT::FAILURE; // reset to the default

  DBUG_RETURN(res);
}

/**
  a helper function to report an access denied error in all the proper places
*/
static void login_failed_error(MPVIO_EXT *mpvio, int passwd_used)
{
  THD *thd= current_thd;

  if (thd->is_error())
  {
    // sql_print_information("%s", thd->get_stmt_da()->message_text());
  }
  else if (passwd_used == 2)
  {
    my_error(ER_ACCESS_DENIED_NO_PASSWORD_ERROR, MYF(0),
             mpvio->auth_info.user_name,
             mpvio->auth_info.host_or_ip);
    /*
      Log access denied messages to the error log when log-warnings = 2
      so that the overhead of the general query log is not required to track
      failed connections.
    */
  }
  else
  {
    my_error(ER_ACCESS_DENIED_ERROR, MYF(0),
             mpvio->auth_info.user_name,
             mpvio->auth_info.host_or_ip,
             passwd_used ? ER(ER_YES) : ER(ER_NO));
    /*
      Log access denied messages to the error log when log-warnings = 2
      so that the overhead of the general query log is not required to track
      failed connections.
    */
  }
}

static bool parse_com_change_user_packet(MPVIO_EXT *mpvio, size_t packet_length);
static bool find_mpvio_user(MPVIO_EXT *mpvio);
static bool read_client_connect_attrs(char **ptr, size_t *max_bytes_available,
                          const CHARSET_INFO *from_cs);

/**
  Perform the handshake, authorize the client and update thd sctx variables.

  @param thd                     thread handle
  @param command                 the command to be executed, it can be either a
                                 COM_CHANGE_USER or COM_CONNECT (if
                                 it's a new connection)

  @retval 0  success, thd is updated.
  @retval 1  error
*/
int
acl_authenticate(THD *thd, enum_server_command command)
{
  int res= CR_OK;
  MPVIO_EXT mpvio;
  LEX_CSTRING auth_plugin_name= default_auth_plugin_name;
  // Thd_charset_adapter charset_adapter(thd);

  DBUG_ENTER("acl_authenticate");
  // compile_time_assert(MYSQL_USERNAME_LENGTH == USERNAME_LENGTH);
  DBUG_ASSERT(command == COM_CONNECT || command == COM_CHANGE_USER);

  Proc_charset_adapter charset_adapter(thd);
  server_mpvio_initialize(thd, &mpvio, &charset_adapter);

  /*
    Clear thd->db as it points to something, that will be freed when
    connection is closed. We don't want to accidentally free a wrong
    pointer if connect failed.
  */
  thd->set_db("");

  /* acl_authenticate() takes the data from net->read_pos */
  thd->get_protocol_classic()->get_net()->read_pos=
    thd->get_protocol_classic()->get_raw_packet();
  // DBUG_PRINT("info", ("com_change_user_pkt_len=%u",
  //   mpvio.protocol->get_packet_length()));

  if (command == COM_CHANGE_USER)
  {
    mpvio.packets_written++; // pretend that a server handshake packet was sent
    mpvio.packets_read++;    // take COM_CHANGE_USER packet into account

    if (parse_com_change_user_packet(&mpvio,
                                     mpvio.protocol->get_packet_length()))
    {
      login_failed_error(&mpvio, mpvio.auth_info.password_used);
      DBUG_RETURN(1);
    }

    DBUG_ASSERT(mpvio.status == MPVIO_EXT::RESTART ||
                mpvio.status == MPVIO_EXT::SUCCESS);
  }
  else
  {
    /* mark the thd as having no scramble yet */
    mpvio.scramble[SCRAMBLE_LENGTH]= 1;
    
    /*
     perform the first authentication attempt, with the default plugin.
     This sends the server handshake packet, reads the client reply
     with a user name, and performs the authentication if everyone has used
     the correct plugin.
    */

    res= do_auth_once(thd, auth_plugin_name, &mpvio);
  }

  /*
   retry the authentication, if - after receiving the user name -
   we found that we need to switch to a non-default plugin
  */
  if (mpvio.status == MPVIO_EXT::RESTART)
  {
    DBUG_ASSERT(mpvio.acl_user);
    DBUG_ASSERT(command == COM_CHANGE_USER ||
                my_strcasecmp(system_charset_info, auth_plugin_name.str,
                              mpvio.acl_user->plugin.c_str()));
    res= do_auth_once(thd, auth_plugin_name, &mpvio);
    if (res <= CR_OK)
    {
      if (auth_plugin_name.str == native_password_plugin_name.str)
        thd->variables.old_passwords= 0;
      if (auth_plugin_name.str == sha256_password_plugin_name.str)
        thd->variables.old_passwords= 2;
    }
  }

  if (res == CR_OK &&
      (!mpvio.can_authenticate() || thd->is_error()))
  {
    res= CR_ERROR;
  }

  if (thd->is_error())
    DBUG_RETURN(1);

  if (res > CR_OK && mpvio.status != MPVIO_EXT::SUCCESS)
  {
    DBUG_ASSERT(mpvio.status == MPVIO_EXT::FAILURE);
    login_failed_error(&mpvio, mpvio.auth_info.password_used);
    DBUG_RETURN (1);
  }

  if (NULL != mpvio.auth_info.user_name)
  {
    thd->set_user_name(mpvio.auth_info.user_name, mpvio.auth_info.user_name_length);
    free(mpvio.auth_info.user_name);
    mpvio.auth_info.user_name = NULL;
  }
  if (mpvio.cached_server_packet.pkt) {
      free(mpvio.cached_server_packet.pkt);
      mpvio.cached_server_packet.pkt = NULL;
  }
    /* Change a database if necessary */
    if (!mpvio.db.empty()) {
        if (mysql_change_db(thd, mpvio.db, false)) {
            DBUG_RETURN(1);
        }
    }
    
  my_ok(thd);
  /* Ready to handle queries */
  DBUG_RETURN(0);

}

/**
  Make sure that when sending plugin supplied data to the client they
  are not considered a special out-of-band command, like e.g. 
  \255 (error) or \254 (change user request packet) or \0 (OK).
  To avoid this the server will send all plugin data packets "wrapped" 
  in a command \1.
  Note that the client will continue sending its replies unrwapped.
*/

static inline int 
wrap_plguin_data_into_proper_command(NET *net, 
                                     const uchar *packet, int packet_len)
{
  return net_write_command(net, 1, (uchar *) "", 0, packet, packet_len);
}

/**
  vio->write_packet() callback method for server authentication plugins

  This function is called by a server authentication plugin, when it wants
  to send data to the client.

  It transparently wraps the data into a handshake packet,
  and handles plugin negotiation with the client. If necessary,
  it escapes the plugin data, if it starts with a mysql protocol packet byte.
*/
static int server_mpvio_write_packet(MYSQL_PLUGIN_VIO *param,
                                     const uchar *packet, int packet_len)
{
  MPVIO_EXT *mpvio= (MPVIO_EXT *) param;
  int res;
  Protocol_classic *protocol = mpvio->protocol;

  // DBUG_ENTER("server_mpvio_write_packet");
  /* 
    Reset cached_client_reply if not an old client doing mysql_change_user, 
    as this is where the password from COM_CHANGE_USER is stored.
  */
  if (!((!(protocol->has_client_capability(CLIENT_PLUGIN_AUTH))) &&
        mpvio->status == MPVIO_EXT::RESTART/* &&
        mpvio->cached_client_reply.plugin == 
        ((st_mysql_auth *) (plugin_decl(mpvio->plugin)->info))->client_auth_plugin*/
        ))
    mpvio->cached_client_reply.pkt= 0;
  /* for the 1st packet we wrap plugin data into the handshake packet */
  if (mpvio->packets_written == 0)
    res = send_server_handshake_packet(mpvio, (char *)packet, packet_len);
  else if (mpvio->status == MPVIO_EXT::RESTART)
    res = send_plugin_request_packet(mpvio, packet, packet_len);
  else
    res= wrap_plguin_data_into_proper_command(protocol->get_net(),
                                              packet, packet_len);
  mpvio->packets_written++;
  DBUG_RETURN(res);
}

/**
  sends a server handshake initialization packet, the very first packet
  after the connection was established

  Packet format:

    Bytes       Content
    -----       ----
    1           protocol version (always 10)
    n           server version string, \0-terminated
    4           thread id
    8           first 8 bytes of the plugin provided data (scramble)
    1           \0 byte, terminating the first part of a scramble
    2           server capabilities (two lower bytes)
    1           server character set
    2           server status
    2           server capabilities (two upper bytes)
    1           length of the scramble
    10          reserved, always 0
    n           rest of the plugin provided data (at least 12 bytes)
    1           \0 byte, terminating the second part of a scramble

  @retval 0 ok
  @retval 1 error
*/
static bool
send_server_handshake_packet(MPVIO_EXT *mpvio,
                             const char *data, uint data_len)
{
  // DBUG_ASSERT(mpvio->status == MPVIO_EXT::FAILURE);
  DBUG_ASSERT(data_len <= 255);
  Protocol_classic *protocol= mpvio->protocol;

  char *buff= (char *) alloca(1 + SERVER_VERSION_LENGTH + data_len + 64);
  char scramble_buf[SCRAMBLE_LENGTH];
  char *end= buff;

  // DBUG_ENTER("send_server_handshake_packet");
  *end++= PROTOCOL_VERSION;

  protocol->set_client_capabilities(CLIENT_BASIC_FLAGS);

  // if (opt_using_transactions)
     protocol->add_client_capability(CLIENT_TRANSACTIONS);

  protocol->add_client_capability(CAN_CLIENT_COMPRESS);

  // if (ssl_acceptor_fd)
  // {
  //   protocol->add_client_capability(CLIENT_SSL);
  //   protocol->add_client_capability(CLIENT_SSL_VERIFY_SERVER_CERT);
  // }

  if (data_len)
  {
    if (NULL != mpvio->cached_server_packet.pkt)
    {
      free(mpvio->cached_server_packet.pkt);
    }
    mpvio->cached_server_packet.pkt= (char*) strndup(data, data_len);
    mpvio->cached_server_packet.pkt_len= data_len;
  }

  if (data_len < SCRAMBLE_LENGTH)
  {
    if (data_len)
    {
      /*
        the first packet *must* have at least 20 bytes of a scramble.
        if a plugin provided less, we pad it to 20 with zeros
      */
      memcpy(scramble_buf, data, data_len);
      memset(scramble_buf + data_len, 0, SCRAMBLE_LENGTH - data_len);
      data= scramble_buf;
    }
    else
    {
      /*
        if the default plugin does not provide the data for the scramble at
        all, we generate a scramble internally anyway, just in case the
        user account (that will be known only later) uses a
        native_password_plugin (which needs a scramble). If we don't send a
        scramble now - wasting 20 bytes in the packet -
        native_password_plugin will have to send it in a separate packet,
        adding one more round trip.
      */
      generate_user_salt(mpvio->scramble, SCRAMBLE_LENGTH + 1);
      data= mpvio->scramble;
    }
    data_len= SCRAMBLE_LENGTH;
  }

  stpcpy(server_version, MYSQL_SERVER_VERSION);
  end= my_stpnmov(end, server_version, SERVER_VERSION_LENGTH) + 1;

  // DBUG_ASSERT(sizeof(my_thread_id) == 4);
  int4store((uchar*) end, mpvio->thread_id);
  end+= 4;

  /*
    Old clients does not understand long scrambles, but can ignore packet
    tail: that's why first part of the scramble is placed here, and second
    part at the end of packet.
  */
  end= (char*) memcpy(end, data, AUTH_PLUGIN_DATA_PART_1_LENGTH);
  end+= AUTH_PLUGIN_DATA_PART_1_LENGTH;
  *end++= 0;
 
  int2store(end, static_cast<uint16>(protocol->get_client_capabilities()));
  /* write server characteristics: up to 16 bytes allowed */
  end[2]= (char) default_charset_info->number;
  // end[2]= (char) default_charset_info->number;
  int2store(end + 3, SERVER_STATUS_AUTOCOMMIT);
  // int2store(end + 3, mpvio->server_status[0]);
  int2store(end + 5, protocol->get_client_capabilities() >> 16);
  end[7]= data_len;
  // DBUG_EXECUTE_IF("poison_srv_handshake_scramble_len", end[7]= -100;);
  memset(end + 8, 0, 10);
  end+= 18;
  /* write scramble tail */
  end= (char*) memcpy(end, data + AUTH_PLUGIN_DATA_PART_1_LENGTH,
                      data_len - AUTH_PLUGIN_DATA_PART_1_LENGTH);
  end+= data_len - AUTH_PLUGIN_DATA_PART_1_LENGTH;
  end= strmake(end, default_auth_plugin_name.str,
                    default_auth_plugin_name.length);
  // end= strmake(end, plugin_name(mpvio->plugin)->str,
  //                   plugin_name(mpvio->plugin)->length);

  int res= my_net_write(protocol->get_net(), (uchar*) buff, (size_t) (end - buff + 1)) ||
           net_flush(protocol->get_net());
  // int res= protocol->write((uchar*) buff, (size_t) (end - buff + 1)) ||
  //          protocol->flush_net();
  DBUG_RETURN (res);
}

/**
  vio->read_packet() callback method for server authentication plugins

  This function is called by a server authentication plugin, when it wants
  to read data from the client.

  It transparently extracts the client plugin data, if embedded into
  a client authentication handshake packet, and handles plugin negotiation
  with the client, if necessary.

  RETURN
    -1          Protocol failure
    >= 0        Success and also the packet length
*/
static int server_mpvio_read_packet(MYSQL_PLUGIN_VIO *param, uchar **buf)
{
  MPVIO_EXT *mpvio= (MPVIO_EXT *) param;
  Protocol_classic *protocol = mpvio->protocol;
  size_t pkt_len;

  // DBUG_ENTER("server_mpvio_read_packet");
  if (mpvio->packets_written == 0)
  {
    /*
      plugin wants to read the data without sending anything first.
      send an empty packet to force a server handshake packet to be sent
    */
    if (mpvio->write_packet(mpvio, 0, 0))
      pkt_len= packet_error;
    else
    {
      protocol->read_packet();
      pkt_len= protocol->get_packet_length();
    }
  }
  else if (mpvio->cached_client_reply.pkt)
  {
    DBUG_ASSERT(mpvio->status == MPVIO_EXT::RESTART);
    DBUG_ASSERT(mpvio->packets_read > 0);
    /*
      if the have the data cached from the last server_mpvio_read_packet
      (which can be the case if it's a restarted authentication)
      and a client has used the correct plugin, then we can return the
      cached data straight away and avoid one round trip.
    */
    const char *client_auth_plugin= mpvio->plugin.data();
      // ((st_mysql_auth *) (plugin_decl(mpvio->plugin)->info))->client_auth_plugin;
    if (client_auth_plugin == 0 ||
        my_strcasecmp(system_charset_info, mpvio->cached_client_reply.plugin,
                      client_auth_plugin) == 0)
    {
      mpvio->status= MPVIO_EXT::FAILURE;
      *buf= (uchar*) mpvio->cached_client_reply.pkt;
      mpvio->cached_client_reply.pkt= 0;
      mpvio->packets_read++;
      DBUG_RETURN ((int) mpvio->cached_client_reply.pkt_len);
    }

    /* older clients don't support change of client plugin request */
    if (!(protocol->has_client_capability(CLIENT_PLUGIN_AUTH)))
    {
      mpvio->status= MPVIO_EXT::FAILURE;
      pkt_len= packet_error;
      goto err;
    }

    /*
      But if the client has used the wrong plugin, the cached data are
      useless. Furthermore, we have to send a "change plugin" request
      to the client.
    */
    if (mpvio->write_packet(mpvio, 0, 0))
      pkt_len= packet_error;
    else
    {
      protocol->read_packet();
      pkt_len= protocol->get_packet_length();
    }
  }
  else
  {
    protocol->read_packet();
    pkt_len= protocol->get_packet_length();
  }

  if (pkt_len == packet_error)
    goto err;

  mpvio->packets_read++;

  /*
    the 1st packet has the plugin data wrapped into the client authentication
    handshake packet
  */
  if (mpvio->packets_read == 1)
  {
    pkt_len = parse_client_handshake_packet(mpvio, buf, pkt_len);
    if (pkt_len == packet_error)
      goto err;
  }
  else
    *buf= protocol->get_net()->read_pos;

  DBUG_RETURN((int)pkt_len);

err:
  if (mpvio->status == MPVIO_EXT::FAILURE)
  {
    my_error(ER_HANDSHAKE_ERROR, MYF(0));
  }
  DBUG_RETURN(-1);
}

static bool parse_com_change_user_packet(MPVIO_EXT *mpvio, size_t packet_length)
{
  Protocol_classic *protocol = mpvio->protocol;
  char *user= (char*) protocol->get_net()->read_pos;
  char *end= user + packet_length;
  /* Safe because there is always a trailing \0 at the end of the packet */
  char *passwd= strend(user) + 1;
  size_t user_len= passwd - user - 1;
  char *db= passwd;
  char db_buff[NAME_LEN + 1];                 // buffer to store db in utf8
  char user_buff[USERNAME_LENGTH + 1];        // buffer to store user in utf8
  uint dummy_errors;

  DBUG_ENTER ("parse_com_change_user_packet");
  if (passwd >= end)
  {
    my_message(ER_UNKNOWN_COM_ERROR, ER(ER_UNKNOWN_COM_ERROR), MYF(0));
    DBUG_RETURN (1);
  }

  /*
    Clients send the size (1 byte) + string (not null-terminated).

    Cast *passwd to an unsigned char, so that it doesn't extend the sign for
    *passwd > 127 and become 2**32-127+ after casting to uint.
  */
  size_t passwd_len= (uchar) (*passwd++);

  db+= passwd_len + 1;
  /*
    Database name is always NUL-terminated, so in case of empty database
    the packet must contain at least the trailing '\0'.
  */
  if (db >= end)
  {
    my_message(ER_UNKNOWN_COM_ERROR, ER(ER_UNKNOWN_COM_ERROR), MYF(0));
    DBUG_RETURN (1);
  }

  size_t db_len= strlen(db);

  char *ptr= db + db_len + 1;

  if (ptr + 1 < end)
  {
    if (mpvio->charset_adapter->init_client_charset(uint2korr(ptr)))
      DBUG_RETURN(1);
  }

  /* Convert database and user names to utf8 */
  db_len= copy_and_convert(db_buff, sizeof(db_buff) - 1, system_charset_info,
                           db, db_len, mpvio->charset_adapter->charset(),
                           &dummy_errors);
  db_buff[db_len]= 0;

  user_len= copy_and_convert(user_buff, sizeof(user_buff) - 1,
                                  system_charset_info, user, user_len,
                                  mpvio->charset_adapter->charset(),
                                  &dummy_errors);
  user_buff[user_len]= 0;

  /* we should not free mpvio->user here: it's saved by dispatch_command() */
  if (!(mpvio->auth_info.user_name= strndup(user_buff, user_len)))
    DBUG_RETURN(1);
  mpvio->auth_info.user_name_length= user_len;
  if (db_len > 0) {
      mpvio->db.assign(db_buff, db_len);
  }
  // if (make_lex_string_root(mpvio->mem_root, 
  //                          &mpvio->db, db_buff, db_len, 0) == 0)
  //   DBUG_RETURN(1); /* The error is set by make_lex_string(). */

  // if (!initialized)
  // {
  //   // if mysqld's been started with --skip-grant-tables option
  //   strmake(mpvio->auth_info.authenticated_as, 
  //           mpvio->auth_info.user_name, USERNAME_LENGTH);

  //   mpvio->status= MPVIO_EXT::SUCCESS;
  //   DBUG_RETURN(0);
  // }

  if (find_mpvio_user(mpvio))
  {
    DBUG_RETURN(1);
  }

  const char *client_plugin;
  if (protocol->has_client_capability(CLIENT_PLUGIN_AUTH))
  {
    client_plugin= ptr + 2;
    if (client_plugin >= end)
    {
      my_message(ER_UNKNOWN_COM_ERROR, ER(ER_UNKNOWN_COM_ERROR), MYF(0));
      DBUG_RETURN(1);
    }
  }
  else
    client_plugin= native_password_plugin_name.str;

  size_t bytes_remaining_in_packet= end - ptr;

  if (protocol->has_client_capability(CLIENT_CONNECT_ATTRS) &&
      read_client_connect_attrs(&ptr, &bytes_remaining_in_packet,
                                mpvio->charset_adapter->charset()))
    DBUG_RETURN(MY_TEST(packet_error));

  // DBUG_PRINT("info", ("client_plugin=%s, restart", client_plugin));
  /* 
    Remember the data part of the packet, to present it to plugin in 
    read_packet() 
  */
  mpvio->cached_client_reply.pkt= passwd;
  mpvio->cached_client_reply.pkt_len= passwd_len;
  mpvio->cached_client_reply.plugin= client_plugin;
  mpvio->status= MPVIO_EXT::RESTART;

  DBUG_RETURN (0);
}

#ifndef EMBEDDED_LIBRARY
/** Get a string according to the protocol of the underlying buffer. */
typedef char * (*get_proto_string_func_t) (char **, size_t *, size_t *);

/**
  Get a string formatted according to the 4.1 version of the MySQL protocol.

  @param buffer[in, out]    Pointer to the user-supplied buffer to be scanned.
  @param max_bytes_available[in, out]  Limit the bytes to scan.
  @param string_length[out] The number of characters scanned not including
                            the null character.

  @remark Strings are always null character terminated in this version of the
          protocol.

  @remark The string_length does not include the terminating null character.
          However, after the call, the buffer is increased by string_length+1
          bytes, beyond the null character if there still available bytes to
          scan.

  @return pointer to beginning of the string scanned.
    @retval NULL The buffer content is malformed
*/

static
char *get_41_protocol_string(char **buffer,
                             size_t *max_bytes_available,
                             size_t *string_length)
{
  char *str= (char *)memchr(*buffer, '\0', *max_bytes_available);

  if (str == NULL)
    return NULL;

  *string_length= (size_t)(str - *buffer);
  *max_bytes_available-= *string_length + 1;
  str= *buffer;
  *buffer += *string_length + 1;

  return str;
}


/**
  Get a string formatted according to the 4.0 version of the MySQL protocol.

  @param buffer[in, out]    Pointer to the user-supplied buffer to be scanned.
  @param max_bytes_available[in, out]  Limit the bytes to scan.
  @param string_length[out] The number of characters scanned not including
                            the null character.

  @remark If there are not enough bytes left after the current position of
          the buffer to satisfy the current string, the string is considered
          to be empty and a pointer to empty_c_string is returned.

  @remark A string at the end of the packet is not null terminated.

  @return Pointer to beginning of the string scanned, or a pointer to a empty
          string.
*/
static
char *get_40_protocol_string(char **buffer,
                             size_t *max_bytes_available,
                             size_t *string_length)
{
  char *str;
  size_t len;

  /* No bytes to scan left, treat string as empty. */
  if ((*max_bytes_available) == 0)
  {
    *string_length= 0;
    return empty_c_string;
  }

  str= (char *) memchr(*buffer, '\0', *max_bytes_available);

  /*
    If the string was not null terminated by the client,
    the remainder of the packet is the string. Otherwise,
    advance the buffer past the end of the null terminated
    string.
  */
  if (str == NULL)
    len= *string_length= *max_bytes_available;
  else
    len= (*string_length= (size_t)(str - *buffer)) + 1;

  str= *buffer;
  *buffer+= len;
  *max_bytes_available-= len;

  return str;
}

/**
  Get a length encoded string from a user-supplied buffer.

  @param buffer[in, out] The buffer to scan; updates position after scan.
  @param max_bytes_available[in, out] Limit the number of bytes to scan
  @param string_length[out] Number of characters scanned

  @remark In case the length is zero, then the total size of the string is
    considered to be 1 byte; the size byte.

  @return pointer to first byte after the header in buffer.
    @retval NULL The buffer content is malformed
*/

static
char *get_56_lenc_string(char **buffer,
                         size_t *max_bytes_available,
                         size_t *string_length)
{
  static char empty_string[1]= { '\0' };
  char *begin= *buffer;
  uchar *pos= (uchar *)begin;
  size_t required_length= 9;


  if (*max_bytes_available == 0)
    return NULL;

  /*
    If the length encoded string has the length 0
    the total size of the string is only one byte long (the size byte)
  */
  if (*begin == 0)
  {
    *string_length= 0;
    --*max_bytes_available;
    ++*buffer;
    /*
      Return a pointer to the \0 character so the return value will be
      an empty string.
    */
    return empty_string;
  }

  /* Make sure we have enough bytes available for net_field_length_ll */
  // DBUG_EXECUTE_IF("buffer_too_short_3",
  //                 *pos= 252; *max_bytes_available= 2;
  // );
  // DBUG_EXECUTE_IF("buffer_too_short_4",
  //                 *pos= 253; *max_bytes_available= 3;
  // );
  // DBUG_EXECUTE_IF("buffer_too_short_9",
  //                 *pos= 254; *max_bytes_available= 8;
  // );

  if (*pos <= 251)
    required_length= 1;
  if (*pos == 252)
    required_length= 3;
  if (*pos == 253)
    required_length= 4;

  if (*max_bytes_available < required_length)
    return NULL;

  *string_length= (size_t)net_field_length_ll((uchar **)buffer);

  // DBUG_EXECUTE_IF("sha256_password_scramble_too_long",
  //                 *string_length= SIZE_T_MAX;
  // );

  size_t len_len= (size_t)(*buffer - begin);

  DBUG_ASSERT((*max_bytes_available >= len_len) &&
              (len_len == required_length));
  
  if (*string_length > *max_bytes_available - len_len)
    return NULL;

  *max_bytes_available -= *string_length;
  *max_bytes_available -= len_len;
  *buffer += *string_length;
  return (char *)(begin + len_len);
}


/**
  Get a length encoded string from a user-supplied buffer.

  @param buffer[in, out] The buffer to scan; updates position after scan.
  @param max_bytes_available[in, out] Limit the number of bytes to scan
  @param string_length[out] Number of characters scanned

  @remark In case the length is zero, then the total size of the string is
    considered to be 1 byte; the size byte.

  @note the maximum size of the string is 255 because the header is always 
    1 byte.
  @return pointer to first byte after the header in buffer.
    @retval NULL The buffer content is malformed
*/

static
char *get_41_lenc_string(char **buffer,
                         size_t *max_bytes_available,
                         size_t *string_length)
{
 if (*max_bytes_available == 0)
    return NULL;

  /* Do double cast to prevent overflow from signed / unsigned conversion */
  size_t str_len= (size_t)(unsigned char)**buffer;

  /*
    If the length encoded string has the length 0
    the total size of the string is only one byte long (the size byte)
  */
  if (str_len == 0)
  {
    ++*buffer;
    *string_length= 0;
    /*
      Return a pointer to the 0 character so the return value will be
      an empty string.
    */
    return *buffer-1;
  }

  if (str_len >= *max_bytes_available)
    return NULL;

  char *str= *buffer+1;
  *string_length= str_len;
  *max_bytes_available-= *string_length + 1;
  *buffer+= *string_length + 1;
  return str;
}
#endif /* EMBEDDED LIBRARY */

static bool
read_client_connect_attrs(char **ptr, size_t *max_bytes_available,
                          const CHARSET_INFO *from_cs)
{
  size_t length, length_length;
  char *ptr_save;
  /* not enough bytes to hold the length */
  if (*max_bytes_available < 1)
    return true;

  /* read the length */
  ptr_save= *ptr;
  length= static_cast<size_t>(net_field_length_ll((uchar **) ptr));
  length_length= *ptr - ptr_save;
  if (*max_bytes_available < length_length)
    return true;

  *max_bytes_available-= length_length;

  /* length says there're more data than can fit into the packet */
  if (length > *max_bytes_available)
    return true;

  /* impose an artificial length limit of 64k */
  if (length > 65535)
    return true;

  return false;
}

/**
  When authentication is attempted using an unknown username a dummy user
  account with no authentication capabilites is assigned to the connection.
  This is done increase the cost of enumerating user accounts based on
  authentication protocol.
*/

ACL_USER_SPTR decoy_user( const string &username )
{
  auto user = std::make_shared< ACL_USER >();
  user->can_authenticate= false;
  user->user= username;
  user->auth_string= "";
  // user->ssl_cipher= empty_c_string;
  // user->x509_issuer= empty_c_string;
  // user->x509_subject= empty_c_string;
  user->salt_len= 0;
  // user->password_last_changed.time_type= MYSQL_TIMESTAMP_ERROR;
  user->password_lifetime= 0;
  user->use_default_password_lifetime= true;
  user->account_locked= false;

  /*
    For now the common default account is used. Improvements might involve
    mapping a consistent hash of a username to a range of plugins.
  */
  user->plugin= default_auth_plugin_name.str;
  return user;
}

/**
   Finds acl entry in user database for authentication purposes.
   
   Finds a user and copies it into mpvio. Reports an authentication
   failure if a user is not found.

   @note find_acl_user is not the same, because it doesn't take into
   account the case when user is not empty, but acl_user->user is empty

   @retval 0    found
   @retval 1    not found
*/
static bool find_mpvio_user(MPVIO_EXT *mpvio)
{
  DBUG_ENTER("find_mpvio_user");
  // DBUG_PRINT("info", ("entry: %s", mpvio->auth_info.user_name));
  DBUG_ASSERT(mpvio->acl_user == 0);
  std::lock_guard< mutex > lock( acl_users_mutex );
  for ( auto& it : acl_users )
  {
    auto& acl_user_tmp = it.second;
    if ((acl_user_tmp->user.empty() || 
         !strcmp(mpvio->auth_info.user_name, acl_user_tmp->user.data()))/* &&
        acl_user_tmp->host.compare_hostname(mpvio->host, mpvio->ip)*/)
    {
      mpvio->acl_user= acl_user_tmp; //->copy(mpvio->mem_root);
      mpvio->acl_user_plugin = acl_user_tmp->plugin;

      /*
        When setting mpvio->acl_user_plugin we can save memory allocation if
        this is a built in plugin.
      */
      // if (auth_plugin_is_built_in(acl_user_tmp->plugin.str))
      //   mpvio->acl_user_plugin= mpvio->acl_user->plugin;
      // else
      //   make_lex_string_root(mpvio->mem_root, 
      //                        &mpvio->acl_user_plugin, 
      //                        acl_user_tmp->plugin.str, 
      //                        acl_user_tmp->plugin.length, 0);
      break;
    }
  }

  if (!mpvio->acl_user)
  {
    /*
      Pretend the user exists; let the plugin decide how to handle
      bad credentials.
    */
    // LEX_STRING usr= { mpvio->auth_info.user_name,
    //                   mpvio->auth_info.user_name_length };
    mpvio->acl_user= decoy_user( mpvio->auth_info.user_name );
    mpvio->acl_user_plugin= mpvio->acl_user->plugin;
  }

  if (my_strcasecmp(system_charset_info, mpvio->acl_user->plugin.data(),
                    native_password_plugin_name.str) != 0 &&
      !(mpvio->protocol->has_client_capability(CLIENT_PLUGIN_AUTH)))
  {
    /* user account requires non-default plugin and the client is too old */
    DBUG_ASSERT(my_strcasecmp(system_charset_info, mpvio->acl_user->plugin.data(),
                              native_password_plugin_name.str));
    // my_error(ER_NOT_SUPPORTED_AUTH_MODE, MYF(0));
    ARIES_EXCEPTION( ER_NOT_SUPPORTED_AUTH_MODE );
    // query_logger.general_log_print(current_thd, COM_CONNECT,
    //                                ER(ER_NOT_SUPPORTED_AUTH_MODE));
    DBUG_RETURN (1);
  }

  mpvio->auth_info.auth_string= mpvio->acl_user->auth_string;
  // mpvio->auth_info.auth_string_length= 
  //   (unsigned long) mpvio->acl_user->auth_string.length;
  strmake(mpvio->auth_info.authenticated_as, !mpvio->acl_user->user.empty() ?
          mpvio->acl_user->user.data() : "", USERNAME_LENGTH);
  // DBUG_PRINT("info", ("exit: user=%s, auth_string=%s, authenticated as=%s"
  //                     ", plugin=%s",
  //                     mpvio->auth_info.user_name,
  //                     mpvio->auth_info.auth_string,
  //                     mpvio->auth_info.authenticated_as,
  //                     mpvio->acl_user->plugin.str));
  DBUG_RETURN(0);
}


/* the packet format is described in send_client_reply_packet() */
static size_t parse_client_handshake_packet(MPVIO_EXT *mpvio,
                                            uchar **buff, size_t pkt_len)
{
  Protocol_classic *protocol = mpvio->protocol;
  char *end;
  bool packet_has_required_size= false;
  DBUG_ASSERT(mpvio->status == MPVIO_EXT::FAILURE);

  uint charset_code= 0;
  end= (char *)protocol->get_net()->read_pos;
  /*
    In order to safely scan a head for '\0' string terminators
    we must keep track of how many bytes remain in the allocated
    buffer or we might read past the end of the buffer.
  */
  size_t bytes_remaining_in_packet= pkt_len;
  
  /*
    Peek ahead on the client capability packet and determine which version of
    the protocol should be used.
  */
  if (bytes_remaining_in_packet < 2)
    return packet_error;
    
  protocol->set_client_capabilities(uint2korr(end));

  /*
    JConnector only sends server capabilities before starting SSL
    negotiation.  The below code is patch for this.
  */
  if (bytes_remaining_in_packet == 4 &&
      protocol->has_client_capability(CLIENT_SSL))
  {
    protocol->set_client_capabilities(uint4korr(end));
    mpvio->max_client_packet_length= 0xfffff;
    charset_code= global_system_variables.character_set_client->number;
    goto skip_to_ssl;
  }

  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
    packet_has_required_size= bytes_remaining_in_packet >= 
      AUTH_PACKET_HEADER_SIZE_PROTO_41;
  else
    packet_has_required_size= bytes_remaining_in_packet >=
      AUTH_PACKET_HEADER_SIZE_PROTO_40;
  
  if (!packet_has_required_size)
    return packet_error;
  
  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
  {
    protocol->set_client_capabilities(uint4korr(end));
    mpvio->max_client_packet_length= uint4korr(end + 4);
    charset_code= (uint)(uchar)*(end + 8);
    /*
      Skip 23 remaining filler bytes which have no particular meaning.
    */
    end+= AUTH_PACKET_HEADER_SIZE_PROTO_41;
    bytes_remaining_in_packet-= AUTH_PACKET_HEADER_SIZE_PROTO_41;
  }
  else
  {
    protocol->set_client_capabilities(uint2korr(end));
    mpvio->max_client_packet_length= uint3korr(end + 2);
    end+= AUTH_PACKET_HEADER_SIZE_PROTO_40;
    bytes_remaining_in_packet-= AUTH_PACKET_HEADER_SIZE_PROTO_40;
    /**
      Old clients didn't have their own charset. Instead the assumption
      was that they used what ever the server used.
    */
    charset_code= global_system_variables.character_set_client->number;
  }

skip_to_ssl:
#if defined(HAVE_OPENSSL)
  DBUG_PRINT("info", ("client capabilities: %lu",
                      protocol->get_client_capabilities()));
  
  /*
    If client requested SSL then we must stop parsing, try to switch to SSL,
    and wait for the client to send a new handshake packet.
    The client isn't expected to send any more bytes until SSL is initialized.
  */
  if (protocol->has_client_capability(CLIENT_SSL))
  {
    unsigned long errptr;
#if !defined(DBUG_OFF)
    uint ssl_charset_code= 0;
#endif

    /* Do the SSL layering. */
    if (!ssl_acceptor_fd)
      return packet_error;

    DBUG_PRINT("info", ("IO layer change in progress..."));
    if (sslaccept(ssl_acceptor_fd, protocol->get_vio(),
                  protocol->get_net()->read_timeout, &errptr))
    {
      DBUG_PRINT("error", ("Failed to accept new SSL connection"));
      return packet_error;
    }

    DBUG_PRINT("info", ("Reading user information over SSL layer"));
    int rc= protocol->read_packet();
    pkt_len= protocol->get_packet_length();
    if (rc)
    {
      DBUG_PRINT("error", ("Failed to read user information (pkt_len= %lu)",
                           static_cast<ulong>(pkt_len)));
      return packet_error;
    }
    /* mark vio as encrypted */
    mpvio->vio_is_encrypted= 1;
  
    /*
      A new packet was read and the statistics reflecting the remaining bytes
      in the packet must be updated.
    */
    bytes_remaining_in_packet= pkt_len;

    /*
      After the SSL handshake is performed the client resends the handshake
      packet but because of legacy reasons we chose not to parse the packet
      fields a second time and instead only assert the length of the packet.
    */
    if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
    {
      packet_has_required_size= bytes_remaining_in_packet >= 
        AUTH_PACKET_HEADER_SIZE_PROTO_41;
#if !defined(DBUG_OFF)
      ssl_charset_code=
        (uint)(uchar)*((char *)protocol->get_net()->read_pos + 8);
      DBUG_PRINT("info", ("client_character_set: %u", ssl_charset_code));
#endif
      end= (char *)protocol->get_net()->read_pos
        + AUTH_PACKET_HEADER_SIZE_PROTO_41;
      bytes_remaining_in_packet -= AUTH_PACKET_HEADER_SIZE_PROTO_41;
    }
    else
    {
      packet_has_required_size= bytes_remaining_in_packet >= 
        AUTH_PACKET_HEADER_SIZE_PROTO_40;
      end= (char *)protocol->get_net()->read_pos
        + AUTH_PACKET_HEADER_SIZE_PROTO_40;
      bytes_remaining_in_packet -= AUTH_PACKET_HEADER_SIZE_PROTO_40;
#if !defined(DBUG_OFF)
      /**
        Old clients didn't have their own charset. Instead the assumption
        was that they used what ever the server used.
      */
      ssl_charset_code= global_system_variables.character_set_client->number;
#endif
    }
    DBUG_ASSERT(charset_code == ssl_charset_code);
    if (!packet_has_required_size)
      return packet_error;
  }
#endif /* HAVE_OPENSSL */

  LOG(INFO) << "client_character_set: " << charset_code;
  if (mpvio->charset_adapter->init_client_charset(charset_code))
    return packet_error;

  if ((protocol->has_client_capability(CLIENT_TRANSACTIONS)) /* &&
      opt_using_transactions */ )
    protocol->get_net()->return_status= mpvio->server_status;

  /*
    The 4.0 and 4.1 versions of the protocol differ on how strings
    are terminated. In the 4.0 version, if a string is at the end
    of the packet, the string is not null terminated. Do not assume
    that the returned string is always null terminated.
  */
  get_proto_string_func_t get_string;

  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
    get_string= get_41_protocol_string;
  else
    get_string= get_40_protocol_string;

  /*
    When the ability to change default plugin require that the initial password
   field can be of arbitrary size. However, the 41 client-server protocol limits
   the length of the auth-data-field sent from client to server to 255 bytes
   (CLIENT_SECURE_CONNECTION). The solution is to change the type of the field
   to a true length encoded string and indicate the protocol change with a new
   client capability flag: CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA.
  */
  get_proto_string_func_t get_length_encoded_string;

  if (protocol->has_client_capability(CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA))
    get_length_encoded_string= get_56_lenc_string;
  else
    get_length_encoded_string= get_41_lenc_string;

  /*
    In order to safely scan a head for '\0' string terminators
    we must keep track of how many bytes remain in the allocated
    buffer or we might read past the end of the buffer.
  */
  bytes_remaining_in_packet=
    pkt_len - (end - (char *)protocol->get_net()->read_pos);

  size_t user_len;
  char *user= get_string(&end, &bytes_remaining_in_packet, &user_len);
  if (user == NULL)
    return packet_error;

  /*
    Old clients send a null-terminated string as password; new clients send
    the size (1 byte) + string (not null-terminated). Hence in case of empty
    password both send '\0'.
  */
  size_t passwd_len= 0;
  char *passwd= NULL;

  passwd= get_length_encoded_string(&end, &bytes_remaining_in_packet,
                                    &passwd_len);
  if (passwd == NULL)
  {
    return packet_error;
  }

  size_t db_len= 0;
  char *db= NULL;

  if (protocol->has_client_capability(CLIENT_CONNECT_WITH_DB))
  {
    db= get_string(&end, &bytes_remaining_in_packet, &db_len);
    if (db == NULL)
      return packet_error;
  }

  /*
    Set the default for the password supplied flag for non-existing users
    as the default plugin (native passsword authentication) would do it
    for compatibility reasons.
  */
  if (passwd_len)
    mpvio->auth_info.password_used= PASSWORD_USED_YES;

  size_t client_plugin_len= 0;
  const char *client_plugin= get_string(&end, &bytes_remaining_in_packet,
                                  &client_plugin_len);
  if (client_plugin == NULL)
    client_plugin= &empty_c_string[0];

  if ((protocol->has_client_capability(CLIENT_CONNECT_ATTRS)) &&
      read_client_connect_attrs(&end, &bytes_remaining_in_packet,
                                mpvio->charset_adapter->charset()))
    return packet_error;

  char db_buff[NAME_LEN + 1];           // buffer to store db in utf8
  char user_buff[USERNAME_LENGTH + 1];  // buffer to store user in utf8
  uint dummy_errors;


  /*
    Copy and convert the user and database names to the character set used
    by the server. Since 4.1 all database names are stored in UTF-8. Also,
    ensure that the names are properly null-terminated as this is relied
    upon later.
  */
  if (db)
  {
    db_len= copy_and_convert(db_buff, sizeof(db_buff) - 1, system_charset_info,
                             db, db_len, mpvio->charset_adapter->charset(),
                             &dummy_errors);
    db_buff[db_len]= '\0';
    db= db_buff;
  }

  user_len= copy_and_convert(user_buff, sizeof(user_buff) - 1,
                             system_charset_info, user, user_len,
                             mpvio->charset_adapter->charset(),
                             &dummy_errors);
  user_buff[user_len]= '\0';
  user= user_buff;

  /* If username starts and ends in "'", chop them off */
  if (user_len > 1 && user[0] == '\'' && user[user_len - 1] == '\'')
  {
    user[user_len - 1]= 0;
    user++;
    user_len-= 2;
  }

  // LOG(INFO) << "User " << user << ", db " << protocol->m_thd->db();

  if (db_len > 0) {
      mpvio->db.assign(db, db_len);
  }
  if (mpvio->auth_info.user_name)
    free(mpvio->auth_info.user_name);
  if (!(mpvio->auth_info.user_name= strndup(user, user_len)))
    return packet_error; /* The error is set by my_strdup(). */
  mpvio->auth_info.user_name_length= user_len;

  // if (!initialized)
  // {
  //   // if mysqld's been started with --skip-grant-tables option
  //   mpvio->status= MPVIO_EXT::SUCCESS;
  //   return packet_error;
  // }

  if (find_mpvio_user(mpvio))
    return packet_error;

  if (!(protocol->has_client_capability(CLIENT_PLUGIN_AUTH)))
  {
    /* An old client is connecting */
    client_plugin= native_password_plugin_name.str;
  }
  
  /*
    if the acl_user needs a different plugin to authenticate
    (specified in GRANT ... AUTHENTICATED VIA plugin_name ..)
    we need to restart the authentication in the server.
    But perhaps the client has already used the correct plugin -
    in that case the authentication on the client may not need to be
    restarted and a server auth plugin will read the data that the client
    has just send. Cache them to return in the next server_mpvio_read_packet().
  */
  // if (my_strcasecmp(system_charset_info, mpvio->acl_user_plugin.str,
  //                   plugin_name(mpvio->plugin)->str) != 0)
  // {
  //   mpvio->cached_client_reply.pkt= passwd;
  //   mpvio->cached_client_reply.pkt_len= passwd_len;
  //   mpvio->cached_client_reply.plugin= client_plugin;
  //   mpvio->status= MPVIO_EXT::RESTART;
  //   return packet_error;
  // }

  /*
    ok, we don't need to restart the authentication on the server.
    but if the client used the wrong plugin, we need to restart
    the authentication on the client. Do it here, the server plugin
    doesn't need to know.
  */
  const char *client_auth_plugin= mpvio->plugin.data();
  //   ((st_mysql_auth *) (plugin_decl(mpvio->plugin)->info))->client_auth_plugin;

  if (client_auth_plugin &&
      my_strcasecmp(system_charset_info, client_plugin, client_auth_plugin))
  {
    mpvio->cached_client_reply.plugin= client_plugin;
    if (send_plugin_request_packet(mpvio,
                                   (uchar*) mpvio->cached_server_packet.pkt,
                                   mpvio->cached_server_packet.pkt_len))
      return packet_error;

    mpvio->protocol->read_packet();
    passwd_len= protocol->get_packet_length();
    passwd= (char *)protocol->get_net()->read_pos;
  }

  *buff= (uchar *) passwd;
  return passwd_len;
}

int generate_native_password(char *outbuf, unsigned int *buflen,
                             const char *inbuf, unsigned int inbuflen)
{
  // if (my_validate_password_policy(inbuf, inbuflen))
  //   return 1;
  /* for empty passwords */
  if (inbuflen == 0)
  {
    *buflen= 0;
    return 0;
  }
  char *buffer= (char*)malloc( SCRAMBLED_PASSWORD_CHAR_LENGTH+1 );
  if (buffer == NULL)
    return 1;
  my_make_scrambled_password_sha1(buffer, inbuf, inbuflen);
  /*
    if buffer specified by server is smaller than the buffer given
    by plugin then return error
  */
  if (*buflen < strlen(buffer))
  {
    free(buffer);
    return 1;
  }
  *buflen= SCRAMBLED_PASSWORD_CHAR_LENGTH;
  memcpy(outbuf, buffer, *buflen);
  free(buffer);
  return 0;
}

int set_native_salt(const char* password, unsigned int password_len,
                    unsigned char* salt, unsigned char *salt_len)
{
  /* for empty passwords salt_len is 0 */
  if (password_len == 0)
    *salt_len= 0;
  else
  {
    if (password_len == SCRAMBLED_PASSWORD_CHAR_LENGTH)
    {
      get_salt_from_password(salt, password);
      *salt_len= SCRAMBLE_LENGTH;
    }
  }
  return 0;
}