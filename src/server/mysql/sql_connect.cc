/*
   Copyright (c) 2007, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA
*/

/*
  Functions to authenticate and handle requests for a connection
*/
#ifndef EMBEDDED_LIBRARY
#include "./include/mysqld.h"
#include "./include/sql_class.h"
#include "./include/sql_connect.h"
#include "./include/violite.h"
#include "./include/sql_authentication.h"
#include "server/mysql/include/mysql_def.h"
using namespace mysql;

int check_connection(THD *thd);
bool net_send_error(THD *thd, uint sql_errno, const char *err);
/*
  Autenticate user, with error reporting

  SYNOPSIS
   login_connection()
   thd        Thread handler

  NOTES
    Connection is not closed in case of errors

  RETURN
    0    ok
    1    error
*/


bool login_connection(THD *thd)
{
  int error;
  // DBUG_ENTER("login_connection");
  // DBUG_PRINT("info", ("login_connection called by thread %u",
  //                     thd->thread_id()));

  /* Use "connect_timeout" value during connection phase */
  thd->get_protocol_classic()->set_read_timeout(/* connect_timeout */30);
  thd->get_protocol_classic()->set_write_timeout(/* connect_timeout */ 60);

  error= check_connection(thd);
  thd->send_statement_status();

  if (error)
  {           // Wrong permissions
    DBUG_RETURN(1);
  }
  /* Connect completed, set read/write timeouts back to default */
  // thd->get_protocol_classic()->set_read_timeout(
  //   thd->variables.net_read_timeout);
  // thd->get_protocol_classic()->set_write_timeout(
  //   thd->variables.net_write_timeout);
  DBUG_RETURN(0);
}

bool thd_prepare_connection(THD *thd)
{
  bool rc;
  rc= login_connection(thd);

  if (rc)
    return rc;

  // prepare_new_connection_state(thd);
  return FALSE;
}

bool is_supported_parser_charset(const CHARSET_INFO *cs)
{
  return (cs->mbminlen == 1);
}

/**
  Set thread character set variables from the given ID

  @param  thd         thread handle
  @param  cs_number   character set and collation ID

  @retval  0  OK; character_set_client, collation_connection and
              character_set_results are set to the new value,
              or to the default global values.

  @retval  1  error, e.g. the given ID is not supported by parser.
              Corresponding SQL error is sent.
*/

bool thd_init_client_charset(THD *thd, uint cs_number)
{
  // CHARSET_INFO *cs;
  /*
   Use server character set and collation if
   - opt_character_set_client_handshake is not set
   - client has not specified a character set
   - client character set is the same as the servers
   - client character set doesn't exists in server
  */
  // if (!opt_character_set_client_handshake ||
  //     !(cs= get_charset(cs_number, MYF(0))) ||
  //     !my_strcasecmp(&my_charset_latin1,
  //                    global_system_variables.character_set_client->name,
  //                    cs->name))
  // {
    if (!is_supported_parser_charset(
      global_system_variables.character_set_client))
    {
      /* Disallow non-supported parser character sets: UCS2, UTF16, UTF32 */
      LOG(ERROR) << "character_set_client not supported " << global_system_variables.character_set_client->csname;
      return true;
    }    
    thd->variables.character_set_client=
      global_system_variables.character_set_client;
    thd->variables.collation_connection=
      global_system_variables.collation_connection;
    thd->variables.character_set_results=
      global_system_variables.character_set_results;
  // }
  // else
  // {
  //   if (!is_supported_parser_charset(cs))
  //   {
  //     /* Disallow non-supported parser character sets: UCS2, UTF16, UTF32 */
  //     my_error(ER_WRONG_VALUE_FOR_VAR, MYF(0), "character_set_client",
  //              cs->csname);
  //     return true;
  //   }
  //   thd->variables.character_set_results=
  //     thd->variables.collation_connection=
  //     thd->variables.character_set_client= cs;
  // }
  return false;
}

/*
  Perform handshake, authorize client and update thd ACL variables.

  SYNOPSIS
    check_connection()
    thd  thread handle

  RETURN
     0  success, thd is updated.
     1  error
*/

int check_connection(THD *thd)
{
  // uint connect_errors= 0;
  int auth_rc;
  NET *net= thd->get_protocol_classic()->get_net();
  // DBUG_PRINT("info",
  //            ("New connection received on %s", vio_description(net->vio)));

  thd->set_active_vio(net->vio);

  // if (!thd->m_main_security_ctx.host().length)     // If TCP/IP connection
  // {
    my_bool peer_rc;
    char ip[NI_MAXHOST];
    // LEX_CSTRING main_sctx_ip;

    peer_rc= vio_peer_addr(net->vio, ip, &thd->peer_port, NI_MAXHOST);
    if (peer_rc)
    {
      /*
        Since we can not even get the peer IP address,
        there is nothing to show in the host_cache,
        so increment the global status variable for peer address errors.
      */
      // connection_errors_peer_addr++;
      my_error(ER_BAD_HOST_ERROR, MYF(0));
      return 1;
    }
    thd->peer_host.assign(ip, strlen(ip));
    // thd->m_main_security_ctx.assign_ip(ip, strlen(ip));
    // main_sctx_ip= thd->m_main_security_ctx.ip();
    if (!(strlen(ip)))
    {
      /*
        No error accounting per IP in host_cache,
        this is treated as a global server OOM error.
        TODO: remove the need for my_strdup.
      */
      // connection_errors_internal++;
      return 1; /* The error is set by my_strdup(). */
    }
  vio_keepalive(net->vio, TRUE);

  // if (thd->get_protocol_classic()->get_packet()->alloc(
  //     thd->variables.net_buffer_length))
  // {
  //   /*
  //     Important note:
  //     net_buffer_length is a SESSION variable,
  //     so it may be tempting to account OOM conditions per IP in the HOST_CACHE,
  //     in case some clients are more demanding than others ...
  //     However, this session variable is *not* initialized with a per client
  //     value during the initial connection, it is initialized from the
  //     GLOBAL net_buffer_length variable from the server.
  //     Hence, there is no reason to account on OOM conditions per client IP,
  //     we count failures in the global server status instead.
  //   */
  //   connection_errors_internal++;
  //   return 1; /* The error is set by alloc(). */
  // }

  // if (mysql_audit_notify(thd,
  //                       AUDIT_EVENT(MYSQL_AUDIT_CONNECTION_PRE_AUTHENTICATE)))
  // {
  //   return 1;
  // }

  auth_rc= acl_authenticate(thd, COM_CONNECT);
  // auth_rc = native_password_authenticate(thd);

  // if (mysql_audit_notify(thd, AUDIT_EVENT(MYSQL_AUDIT_CONNECTION_CONNECT)))
  // {
  //   return 1;
  // }

  /*
  if (auth_rc == 0 && connect_errors != 0)
  {
      A client connection from this IP was successful,
      after some previous failures.
      Reset the connection error counter.
    // reset_host_connect_errors(thd->m_main_security_ctx.ip().str);
  }
  */

  if (auth_rc <= CR_OK) {
      my_ok(thd);
  }

  return auth_rc;
}

/**
  Close a connection.

  @param thd        Thread handle.
  @param sql_errno  The error code to send before disconnect.
  @param server_shutdown Argument passed to the THD's disconnect method.
  @param generate_event  Generate Audit API disconnect event.

  @note
    For the connection that is doing shutdown, this is called twice
*/

void close_connection(THD *thd, uint sql_errno,
                      bool server_shutdown, bool generate_event)
{
  // DBUG_ENTER("close_connection");

  if (sql_errno)
    net_send_error(thd, sql_errno, ER(sql_errno));

  thd->disconnect(server_shutdown);

  // MYSQL_CONNECTION_DONE((int) sql_errno, thd->thd_id());

  // if (MYSQL_CONNECTION_DONE_ENABLED())
  // {
  //   sleep(0); /* Workaround to avoid tailcall optimisation */
  // }

  // if (generate_event)
  //   mysql_audit_notify(thd,
  //                      AUDIT_EVENT(MYSQL_AUDIT_CONNECTION_DISCONNECT),
  //                      sql_errno);

  DBUG_VOID_RETURN;
}

bool connection_alive(THD *thd)
{
  NET *net= thd->get_protocol_classic()->get_net();
  if (!net->error &&
      net->vio != 0 &&
      !(thd->killed == THD::KILL_CONNECTION))
    return true;
  return false;
}

#endif