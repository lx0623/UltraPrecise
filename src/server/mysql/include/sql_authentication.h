#ifndef SQL_AUTHENTICATION_INCLUDED
#define SQL_AUTHENTICATION_INCLUDED

/* Copyright (c) 2000, 2015, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software Foundation,
   51 Franklin Street, Suite 500, Boston, MA 02110-1335 USA */

#include "my_global.h"                  // NO_EMBEDDED_ACCESS_CHECKS
#include "plugin_auth_common.h"
#include "m_string.h"
#include "mysql_com.h"
#include "plugin_auth.h"
#include "sql_class.h"
#include "protocol_classic.h"
#include "sql_auth_cache.h"
// #include "my_thread_local.h"            // my_thread_id
// #include <mysql/plugin_auth.h>          // MYSQL_SERVER_AUTH_INFO
// #include "sql_plugin_ref.h"             // plugin_ref

/* Forward declarations */
typedef struct charset_info_st CHARSET_INFO;
typedef struct st_net NET;

int acl_authenticate(THD *thd, enum_server_command command);
int native_password_authenticate(THD* thd);
int set_default_auth_plugin(char *plugin_name, size_t plugin_name_length);
int generate_native_password(char *outbuf, unsigned int *buflen,
                             const char *inbuf, unsigned int inbuflen);
int set_native_salt(const char* password, unsigned int password_len,
                    unsigned char* salt, unsigned char *salt_len);

/* Classes */

class Proc_charset_adapter
{
  THD *thd;
public:
  Proc_charset_adapter(THD *thd_arg) : thd (thd_arg) {}
  bool init_client_charset(uint cs_number);

  const CHARSET_INFO *charset();
};

class ACL_USER;
/**
  The internal version of what plugins know as MYSQL_PLUGIN_VIO,
  basically the context of the authentication session
*/
struct MPVIO_EXT : public MYSQL_PLUGIN_VIO
{
  MPVIO_EXT()
  : packets_read( 0 ),
    packets_written( 0 ),
    status( FAILURE ),
    scramble( 0 ),
    rand( 0 ),
    thread_id( 0 ),
    server_status( 0 ),
    protocol( 0 ),
    max_client_packet_length( 0 ),
    // ip( 0 ),
    // host( 0 ),
    charset_adapter( 0 ),
    vio_is_encrypted( 0 )
  {
  }

  MYSQL_SERVER_AUTH_INFO auth_info;
  ACL_USER_SPTR acl_user;
  // plugin_ref plugin;        ///< what plugin we're under
  std::string plugin;

  string db;            ///< db name from the handshake packet
  /** when restarting a plugin this caches the last client reply */
  struct {
    const char *plugin = 0, *pkt = 0;     ///< pointers into NET::buff
    uint pkt_len;
  } cached_client_reply;
  /** this caches the first plugin packet for restart request on the client */
  struct {
    char *pkt = 0;
    uint pkt_len = 0;
  } cached_server_packet;
  int packets_read, packets_written; ///< counters for send/received packets
  /** when plugin returns a failure this tells us what really happened */
  enum { SUCCESS, FAILURE, RESTART } status;

  /* encapsulation members */
  char *scramble;
  // MEM_ROOT *mem_root;
  struct  rand_struct *rand;
  uint32 thread_id;
  uint      *server_status;
  Protocol_classic *protocol;
  ulong max_client_packet_length;
  // char *ip;
  // char *host;
  Proc_charset_adapter *charset_adapter;
  std::string acl_user_plugin;
  int vio_is_encrypted;
  bool can_authenticate();
};

#if defined(HAVE_OPENSSL)
#ifndef HAVE_YASSL
typedef struct rsa_st RSA;
class Rsa_authentication_keys
{
private:
  RSA *m_public_key;
  RSA *m_private_key;
  int m_cipher_len;
  char *m_pem_public_key;

  void get_key_file_path(char *key, String *key_file_path);
  bool read_key_file(RSA **key_ptr, bool is_priv_key, char **key_text_buffer);

public:
  Rsa_authentication_keys();
  ~Rsa_authentication_keys()
  {
  }

  void free_memory();
  void *allocate_pem_buffer(size_t buffer_len);
  RSA *get_private_key()
  {
    return m_private_key;
  }

  RSA *get_public_key()
  {
    return m_public_key;
  }

  int get_cipher_length();
  bool read_rsa_keys();
  const char *get_public_key_as_pem(void)
  {
    return m_pem_public_key;
  }
  
};

#endif /* HAVE_YASSL */
#endif /* HAVE_OPENSSL */

/* Data Structures */

extern LEX_CSTRING native_password_plugin_name;
extern LEX_CSTRING sha256_password_plugin_name;
extern LEX_CSTRING validate_password_plugin_name;
extern LEX_CSTRING default_auth_plugin_name;

#endif /* SQL_AUTHENTICATION_INCLUDED */
