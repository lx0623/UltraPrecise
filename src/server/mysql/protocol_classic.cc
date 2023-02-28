/* Copyright (c) 2000, 2018, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

/**
@file

Low level functions for storing data to be send to the MySQL client.
The actual communication is handled by the net_xxx functions in net_serv.cc
*/

#include "./include/sql_class.h"
#include "./include/m_string.h"
#include "./include/protocol_classic.h"
#include "./include/my_byteorder.h"
#include "./include/mysqld.h"
#include "./include/my_command.h"
#include "./include/com_data.h"
#include "./include/mysql_com.h"
#include "server/mysql/include/mysql_def.h"
#include "./include/my_time.h"
using namespace mysql;
#include <algorithm>
using std::min;
static uchar eof_buff[1]= { (uchar) 254 };      /* Marker for end of fields */
size_t convert_error_message(char *to, size_t to_length,
                             const CHARSET_INFO *to_cs,
                             const char *from, size_t from_length,
                             const CHARSET_INFO *from_cs, uint *errors);

bool net_send_error_packet(THD *thd, uint sql_errno, const char *err,
                           const char* sqlstate);
bool net_send_error_packet(NET* net, uint sql_errno, const char *err,
                           const char* sqlstate, bool bootstrap,
                           ulong client_capabilities,
                           const CHARSET_INFO* character_set_results);
bool net_send_eof(THD *thd, uint server_status, uint statement_warn_count);
my_bool
net_write_command(NET *net,uchar command,
      const uchar *header, size_t head_len,
      const uchar *packet, size_t len);
/**
  @brief Stores the given string in length-encoded format into the specified
         buffer.

  @param to     [IN]        Buffer to store the given string in.
  @param from   [IN]        The give string to be stored.
  @param length [IN]        Length of the above string.

  @return                   void.
*/

void store_lenenc_string(String &to, const uchar *from, size_t length)
{
  uchar lenBuff[9];
  memset(lenBuff, 0, 9);
  uchar* pos = net_store_length(lenBuff, length);
  to.append(lenBuff, pos - lenBuff);
  to.append(from, length);
}

/**
  Format EOF packet according to the current protocol and
  write it to the network output buffer.

  @param thd The thread handler
  @param net The network handler
  @param server_status The server status
  @param statement_warn_count The number of warnings


  @return
    @retval FALSE The message was sent successfully
    @retval TRUE An error occurred and the messages wasn't sent properly
*/

static bool write_eof_packet(THD *thd, NET *net,
                             uint server_status,
                             uint statement_warn_count)
{
  bool error;
  Protocol_classic *protocol= thd->get_protocol_classic();
  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
  {
    uchar buff[5];
    /*
      Don't send warn count during SP execution, as the warn_list
      is cleared between substatements, and mysqltest gets confused
    */
    uint tmp= min(statement_warn_count, 65535U);
    buff[0]= 254;
    int2store(buff+1, tmp);
    /*
      The following test should never be true, but it's better to do it
      because if 'is_fatal_error' is set the server is not going to execute
      other queries (see the if test in dispatch_command / COM_QUERY)
    */
    // if (thd->is_fatal_error)
    //   server_status&= ~SERVER_MORE_RESULTS_EXISTS;
    int2store(buff + 3, server_status);
    error= my_net_write(net, buff, 5);
  }
  else
    error= my_net_write(net, eof_buff, 1);

  return error;
}

Protocol_classic::Protocol_classic() : send_metadata(false), bad_packet(false) {
  packet = new basic_string<uchar>();
  convert = new basic_string<uchar>();
}
Protocol_classic::~Protocol_classic() {
    if (packet) {
        delete packet;
        packet = nullptr;
    }
    if (convert) {
        delete convert;
        convert = nullptr;
    }
}

/*****************************************************************************
  Protocol_classic functions
*****************************************************************************/

void Protocol_classic::init(THD *thd)
{
  m_thd= thd;
#ifndef DBUG_OFF
  // field_types= 0;
#endif
}

bool Protocol_classic::connection_alive()
{
  return m_thd->net.vio != NULL;
}

/**
  Finish the result set with EOF packet, as is expected by the client,
  if there is an error evaluating the next row and a continue handler
  for the error.
*/

void Protocol_classic::end_partial_result_set()
{
  net_send_eof(m_thd, m_thd->server_status,
               0 /* no warnings, we're inside SP */);
}
bool Protocol_text::store_date(aries_acc::AriesDate *date)
{
#ifndef DBUG_OFF
    // field_types check is needed because of the embedded protocol
    // DBUG_ASSERT(send_metadata || field_types == 0 ||
    //             field_types[field_pos] == MYSQL_TYPE_DATE);
    field_pos++;
#endif
    char buff[MAX_DATE_STRING_REP_LENGTH];
    size_t length= my_date_to_str(date, buff);
    return net_store_data((uchar *) buff, length);
}
bool Protocol_text::store_time(aries_acc::AriesTime *tm, uint decimals)
{
#ifndef DBUG_OFF
    // field_types check is needed because of the embedded protocol
    // DBUG_ASSERT(send_metadata || field_types == 0 ||
    //             field_types[field_pos] == MYSQL_TYPE_TIME);
    field_pos++;
#endif
    char buff[MAX_DATE_STRING_REP_LENGTH];
    size_t length= mysql_time_to_str(tm, buff, decimals);
    return net_store_data((uchar *) buff, length);
}
/**
  @todo
  Second_part format ("%06") needs to change when
  we support 0-6 decimals for time.
*/

bool Protocol_text::store(AriesDatetime *datetime, uint decimals)
{
// #ifndef DBUG_OFF
//     // field_types check is needed because of the embedded protocol
//     DBUG_ASSERT(send_metadata || field_types == 0 ||
//                 is_temporal_type_with_date_and_time(field_types[field_pos]));
     field_pos++;
// #endif
    char buff[MAX_DATE_STRING_REP_LENGTH];
    size_t length= mysql_datetime_to_str(datetime, buff, decimals);
    return net_store_data((uchar *) buff, length);
}
bool Protocol_text::store(const char *from, size_t length,
                          const CHARSET_INFO *fromcs,
                          const CHARSET_INFO *tocs)
{
  if(!send_metadata) field_pos++;
  return store_string_aux(from, length, fromcs, tocs);
}

void Protocol_text::start_row()
{
  packet->clear();
}

bool Protocol_text::store_short(longlong from)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //   field_types[field_pos] == MYSQL_TYPE_YEAR ||
  //   field_types[field_pos] == MYSQL_TYPE_SHORT);
  field_pos++;
#endif
  char buff[20];
  return net_store_data((uchar *) buff,
    (size_t) (int10_to_str((int) from, buff, -10) - buff));
}

bool Protocol_text::store_null()
{
#ifndef DBUG_OFF
  field_pos++;
#endif
  char buff[1];
  buff[0]= (char)251;
  packet->append((uchar*)buff, sizeof(buff)/*, PACKET_BUFFER_EXTRA_ALLOC*/);
  return false;
}

bool Protocol_text::store_tiny(longlong from)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //             field_types[field_pos] == MYSQL_TYPE_TINY);
  field_pos++;
#endif
  char buff[20];
  return net_store_data((uchar *) buff,
    (size_t) (int10_to_str((int) from, buff, -10) - buff));
}


bool Protocol_text::store_long(longlong from)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //   field_types[field_pos] == MYSQL_TYPE_INT24 ||
  //   field_types[field_pos] == MYSQL_TYPE_LONG);
  field_pos++;
#endif
  char buff[20];
  return net_store_data((uchar *) buff,
    (size_t) (int10_to_str((long int)from, buff,
                           (from < 0) ? -10 : 10) - buff));
}


bool Protocol_text::store_longlong(longlong from, bool unsigned_flag)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //   field_types[field_pos] == MYSQL_TYPE_LONGLONG);
  field_pos++;
#endif
  char buff[22];
  return net_store_data((uchar *) buff,
    (size_t) (longlong10_to_str(from, buff,
                                unsigned_flag ? 10 : -10)-
                                buff));
}

bool Protocol_text::store(float from, uint32 decimals/*, String *buffer*/)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //   field_types[field_pos] == MYSQL_TYPE_FLOAT);
  field_pos++;
#endif
  // buffer->set_real((double) from, decimals, m_thd->charset());
  char buff[FLOATING_POINT_BUFFER];
  int len = set_real((double)from, decimals, buff, FLOATING_POINT_BUFFER);
  return net_store_data((uchar *)buff, len);
  // return net_store_data((uchar *) buffer->data(), buffer->length());
}

bool Protocol_text::store(double from, uint32 decimals/*, String *buffer*/)
{
#ifndef DBUG_OFF
  // field_types check is needed because of the embedded protocol
  // DBUG_ASSERT(send_metadata || field_types == 0 ||
  //   field_types[field_pos] == MYSQL_TYPE_DOUBLE);
  field_pos++;
#endif
  // buffer->set_real(from, decimals, m_thd->charset());
  char buff[FLOATING_POINT_BUFFER];
  int len = set_real(from, decimals, buff, FLOATING_POINT_BUFFER);
  return net_store_data((uchar *)buff, len);
  // return net_store_data((uchar *) buffer->data(), buffer->length());
}

Vio *Protocol_classic::get_vio()
{
  return m_thd->net.vio;
}

void Protocol_classic::set_vio(Vio *vio)
{
  m_thd->net.vio= vio;
}

// NET interaction functions
bool Protocol_classic::init_net(Vio *vio)
{
  return my_net_init(&m_thd->net, vio);
}
void Protocol_classic::end_net()
{
    DBUG_ASSERT(m_thd->net.buff);
    net_end(&m_thd->net);
    m_thd->net.vio= NULL;
}
bool Protocol_classic::flush_net()
{
    return net_flush(&m_thd->net);
}

bool Protocol_classic::write(const uchar *ptr, size_t len)
{
    return my_net_write(&m_thd->net, ptr, len);
}
NET *Protocol_classic::get_net()
{
  return &m_thd->net;
}
void Protocol_classic::set_read_timeout(ulong read_timeout)
{
  my_net_set_read_timeout(&m_thd->net, read_timeout);
}


void Protocol_classic::set_write_timeout(ulong write_timeout)
{
  my_net_set_write_timeout(&m_thd->net, write_timeout);
}


int Protocol_classic::shutdown(bool server_shutdown)
{
  return m_thd->net.vio ? vio_shutdown(m_thd->net.vio) : 0;
}

uint Protocol_classic::get_rw_status()
{
  return m_thd->net.reading_or_writing;
}

char *strend(const char *s);
bool Protocol_classic::parse_packet(union COM_DATA *data,
                                    enum_server_command cmd)
{
  switch(cmd)
  {
  case COM_INIT_DB:
  {
    data->com_init_db.db_name= reinterpret_cast<const char*>(raw_packet);
    data->com_init_db.length= packet_length;
    break;
  }
  case COM_REFRESH:
  {
    if (packet_length < 1)
      goto malformed;
    data->com_refresh.options= raw_packet[0];
    break;
  }
  case COM_SHUTDOWN:
  {
    data->com_shutdown.level= packet_length == 0 ?
      SHUTDOWN_DEFAULT : (enum mysql_enum_shutdown_level) raw_packet[0];
    break;
  }
  case COM_PROCESS_KILL:
  {
    if (packet_length < 4)
      goto malformed;
    data->com_kill.id= (ulong) uint4korr(raw_packet);
    break;
  }
  case COM_SET_OPTION:
  {
    if (packet_length < 2)
      goto malformed;
    data->com_set_option.opt_command= uint2korr(raw_packet);
    break;
  }
  case COM_STMT_EXECUTE:
  {
    if (packet_length < 9)
      goto malformed;
    data->com_stmt_execute.stmt_id= uint4korr(raw_packet);
    data->com_stmt_execute.flags= (ulong) raw_packet[4];
    /* stmt_id + 5 bytes of flags */
    /*
      FIXME: params have to be parsed into an array/structure
      by protocol too
    */
    data->com_stmt_execute.params= raw_packet + 9;
    data->com_stmt_execute.params_length= packet_length - 9;
    break;
  }
  case COM_STMT_FETCH:
  {
    if (packet_length < 8)
      goto malformed;
    data->com_stmt_fetch.stmt_id= uint4korr(raw_packet);
    data->com_stmt_fetch.num_rows= uint4korr(raw_packet + 4);
    break;
  }
  case COM_STMT_SEND_LONG_DATA:
  {
    if (packet_length < MYSQL_LONG_DATA_HEADER)
      goto malformed;
    data->com_stmt_send_long_data.stmt_id= uint4korr(raw_packet);
    data->com_stmt_send_long_data.param_number= uint2korr(raw_packet + 4);
    data->com_stmt_send_long_data.longdata= raw_packet + 6;
    data->com_stmt_send_long_data.length= packet_length - 6;
    break;
  }
  case COM_STMT_PREPARE:
  {
    data->com_stmt_prepare.query= reinterpret_cast<const char*>(raw_packet);
    data->com_stmt_prepare.length= packet_length;
    break;
  }
  case COM_STMT_CLOSE:
  {
    if (packet_length < 4)
      goto malformed;

    data->com_stmt_close.stmt_id= uint4korr(raw_packet);
    break;
  }
  case COM_STMT_RESET:
  {
    if (packet_length < 4)
      goto malformed;

    data->com_stmt_reset.stmt_id= uint4korr(raw_packet);
    break;
  }
  case COM_QUERY:
  {
    data->com_query.query= reinterpret_cast<const char*>(raw_packet);
    data->com_query.length= packet_length;
    break;
  }
  case COM_FIELD_LIST:
  {
    /*
      We have name + wildcard in packet, separated by endzero
    */
    data->com_field_list.table_name= raw_packet;
    uint len= data->com_field_list.table_name_length=
        strend((char *)raw_packet) - (char *)raw_packet;
    if (len >= packet_length || len > NAME_LEN)
      goto malformed;
    data->com_field_list.query= raw_packet + len + 1;
    data->com_field_list.query_length= packet_length - len;
    break;
  }
  default:
    break;
  }

  return false;

malformed:
  my_error(ER_MALFORMED_PACKET, MYF(0));
  LOG(ERROR) << "ER_MALFORMED_PACKET";
  bad_packet= true;
  return true;
}

int Protocol_classic::get_command(COM_DATA *com_data, enum_server_command *cmd)
{
  // read packet from the network
  if(int rc= read_packet())
  {
    return rc;
  }

  /*
    'packet_length' contains length of data, as it was stored in packet
    header. In case of malformed header, my_net_read returns zero.
    If packet_length is not zero, my_net_read ensures that the returned
    number of bytes was actually read from network.
    There is also an extra safety measure in my_net_read:
    it sets packet[packet_length]= 0, but only for non-zero packets.
  */
  if (packet_length == 0)                       /* safety */
  {
    /* Initialize with COM_SLEEP packet */
    raw_packet[0]= (uchar) COM_SLEEP;
    packet_length= 1;
  }
  /* Do not rely on my_net_read, extra safety against programming errors. */
  raw_packet[packet_length]= '\0';                  /* safety */

  *cmd= (enum enum_server_command) raw_packet[0];

  if (*cmd >= COM_END)
    *cmd= COM_END;				// Wrong command

  DBUG_ASSERT(packet_length);
  // Skip 'command'
  packet_length--;
  raw_packet++;

  return parse_packet(com_data, *cmd);
}

/**
  Faster net_store_length when we know that length is less than 65536.
  We keep a separate version for that range because it's widely used in
  libmysql.

  uint is used as agrument type because of MySQL type conventions:
    - uint for 0..65536
    - ulong for 0..4294967296
    - ulonglong for bigger numbers.
*/

static uchar *net_store_length_fast(uchar *packet, size_t length)
{
  if (length < 251)
  {
    *packet=(uchar) length;
    return packet+1;
  }
  *packet++=252;
  int2store(packet,(uint) length);
  return packet+2;
}

/****************************************************************************
  Functions used by the protocol functions (like net_send_ok) to store
  strings and numbers in the header result packet.
****************************************************************************/

/* The following will only be used for short strings < 65K */

uchar *net_store_data(uchar *to, const uchar *from, size_t length)
{
  to=net_store_length_fast(to,length);
  memcpy(to,from,length);
  return to+length;
}

/**
  Return OK to the client.

  The OK packet has the following structure:

  Here 'n' denotes the length of state change information.

  Bytes                Name
  -----                ----
  1                    [00] or [FE] the OK header
                       [FE] is used as header for result set rows
  1-9 (lenenc-int)     affected rows
  1-9 (lenenc-int)     last-insert-id

  if capabilities & CLIENT_PROTOCOL_41 {
    2                  status_flags; Copy of thd->server_status; Can be used
                       by client to check if we are inside a transaction.
    2                  warnings (New in 4.1 protocol)
  } elseif capabilities & CLIENT_TRANSACTIONS {
    2                  status_flags
  }

  if capabilities & CLIENT_ACCEPTS_SERVER_STATUS_CHANGE_INFO {
    1-9(lenenc_str)    info (message); Stored as length of the message string +
                       message.
    if n > 0 {
      1-9 (lenenc_int) total length of session state change
                       information to follow (= n)
      n                session state change information
    }
  }
  else {
    string[EOF]          info (message); Stored as packed length (1-9 bytes) +
                         message. Is not stored if no message.
  }

  @param thd                     Thread handler
  @param server_status           The server status
  @param statement_warn_count    Total number of warnings
  @param affected_rows           Number of rows changed by statement
  @param id                      Auto_increment id for first row (if used)
  @param message                 Message to send to the client
                                 (Used by mysql_status)
  @param eof_indentifier         when true [FE] will be set in OK header
                                 else [00] will be used

  @return
    @retval FALSE The message was successfully sent
    @retval TRUE An error occurred and the messages wasn't sent properly
*/

bool net_send_ok(THD *thd,
                 uint server_status, uint statement_warn_count,
                 ulonglong affected_rows, ulonglong id, const char *message,
                 bool eof_identifier)
{
  Protocol *protocol= thd->get_protocol();
  NET *net= thd->get_protocol_classic()->get_net();
  uchar buff[MYSQL_ERRMSG_SIZE + 10];
  uchar *pos, *start;

  /*
    To be used to manage the data storage in case session state change
    information is present.
  */
  // String store;
  bool state_changed= false;

  bool error= FALSE;
  // DBUG_ENTER("net_send_ok");

  if (! net->vio)	// hack for re-parsing queries
  {
    // DBUG_PRINT("info", ("vio present: NO"));
    DBUG_RETURN(FALSE);
  }

  start= buff;

  /*
    Use 0xFE packet header if eof_identifier is true
    unless we are talking to old client
  */
  if (eof_identifier &&
      (protocol->has_client_capability(CLIENT_DEPRECATE_EOF)))
    buff[0]= 254;
  else
    buff[0]= 0;

  /* affected rows */
  pos= net_store_length(buff + 1, affected_rows);

  /* last insert id */
  pos= net_store_length(pos, id);

  if (protocol->has_client_capability(CLIENT_SESSION_TRACK) &&
      thd->session_tracker.enabled_any() &&
      thd->session_tracker.changed_any())
  {
    server_status |= SERVER_SESSION_STATE_CHANGED;
    state_changed= true;
  }

  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
  {
    // DBUG_PRINT("info",
    //     ("affected_rows: %lu  id: %lu  status: %u  warning_count: %u",
    //         (ulong) affected_rows,
    //         (ulong) id,
    //         (uint) (server_status & 0xffff),
    //         (uint) statement_warn_count));
    /* server status */
    int2store(pos, server_status);
    pos+= 2;

    /* warning count: we can only return up to 65535 warnings in two bytes. */
    uint tmp= min(statement_warn_count, 65535U);
    int2store(pos, tmp);
    pos+= 2;
  }
  else if (net->return_status)			// For 4.0 protocol
  {
    int2store(pos, server_status);
    pos+=2;
  }

  // thd->get_stmt_da()->set_overwrite_status(true);

  if (protocol->has_client_capability(CLIENT_SESSION_TRACK))
  {
    /* the info field */
    if (state_changed || (message && message[0]))
      pos= net_store_data(pos, (uchar*) message, message ? strlen(message) : 0);
    /* session state change information */
    if (/*unlikely*/(state_changed))
    {
      /*
        First append the fields collected so far. In case of malloc, memory
        for message is also allocated here.
      */
      // store.append((const char *)start, (pos - start), MYSQL_ERRMSG_SIZE);

      String store;
      /* .. and then the state change information. */
      thd->session_tracker.store(thd, store);

      memcpy(pos, store.c_str(), store.length());
      pos+= store.length();
    }
  }
  else if (message && message[0])
  {
    /* the info field, if there is a message to store */
    pos= net_store_data(pos, (uchar*) message, strlen(message));
  }

  /* OK packet length will be restricted to 16777215 bytes */
  if (((size_t) (pos - start)) > MAX_PACKET_LENGTH)
  {
    net->error= 1;
    net->last_errno= ER_NET_OK_PACKET_TOO_LARGE;
    // my_error(ER_NET_OK_PACKET_TOO_LARGE, MYF(0));
    fprintf(stderr, "OK packet too large\n");
    // DBUG_PRINT("info", ("OK packet too large"));
    DBUG_RETURN(1);
  }
  error= my_net_write(net, start, (size_t) (pos - start));
  if (!error)
    error= net_flush(net);

  // thd->get_stmt_da()->set_overwrite_status(false);
  // DBUG_PRINT("info", ("OK sent, so no more error sending allowed"));

  DBUG_RETURN(error);
}

bool net_send_ok(THD *thd,
                 uint server_status, uint statement_warn_count,
                 ulonglong affected_rows, ulonglong id, const char *message,
                 const char* session_state_info, int session_state_info_len,
                 bool eof_identifier)
{
  Protocol *protocol= thd->get_protocol();
  NET *net= thd->get_protocol_classic()->get_net();
  uchar buff[MYSQL_ERRMSG_SIZE + 10];
  uchar *pos, *start;

  /*
    To be used to manage the data storage in case session state change
    information is present.
  */
  // String store;
  bool state_changed= false;

  bool error= FALSE;
  // DBUG_ENTER("net_send_ok");

  if (! net->vio)	// hack for re-parsing queries
  {
    // DBUG_PRINT("info", ("vio present: NO"));
    DBUG_RETURN(FALSE);
  }

  start= buff;

  /*
    Use 0xFE packet header if eof_identifier is true
    unless we are talking to old client
  */
  if (eof_identifier &&
      (protocol->has_client_capability(CLIENT_DEPRECATE_EOF)))
    buff[0]= 254;
  else
    buff[0]= 0;

  /* affected rows */
  pos= net_store_length(buff + 1, affected_rows);

  /* last insert id */
  pos= net_store_length(pos, id);

  if (protocol->has_client_capability(CLIENT_SESSION_TRACK) && NULL != session_state_info)
  {
    server_status |= SERVER_SESSION_STATE_CHANGED;
    state_changed= true;
  }

  if (protocol->has_client_capability(CLIENT_PROTOCOL_41))
  {
    // DBUG_PRINT("info",
    //     ("affected_rows: %lu  id: %lu  status: %u  warning_count: %u",
    //         (ulong) affected_rows,
    //         (ulong) id,
    //         (uint) (server_status & 0xffff),
    //         (uint) statement_warn_count));
    /* server status */
    int2store(pos, server_status);
    pos+= 2;

    /* warning count: we can only return up to 65535 warnings in two bytes. */
    uint tmp= min(statement_warn_count, 65535U);
    int2store(pos, tmp);
    pos+= 2;
  }
  else if (net->return_status)			// For 4.0 protocol
  {
    int2store(pos, server_status);
    pos+=2;
  }

  // thd->get_stmt_da()->set_overwrite_status(true);

  if (protocol->has_client_capability(CLIENT_SESSION_TRACK))
  {
    /* the info field */
    if (state_changed || (message && message[0]))
      pos= net_store_data(pos, (uchar*) message, message ? strlen(message) : 0);
    /* session state change information */
    if (state_changed)
    {
      // memcpy(pos, session_state_info, strlen(session_state_info));
      // pos+= strlen(session_state_info);
      pos= net_store_data(pos, (uchar*) session_state_info, session_state_info_len);
    }
  }
  else if (message && message[0])
  {
    /* the info field, if there is a message to store */
    pos= net_store_data(pos, (uchar*) message, strlen(message));
  }

  /* OK packet length will be restricted to 16777215 bytes */
  if (((size_t) (pos - start)) > MAX_PACKET_LENGTH)
  {
    net->error= 1;
    net->last_errno= ER_NET_OK_PACKET_TOO_LARGE;
    // my_error(ER_NET_OK_PACKET_TOO_LARGE, MYF(0));
    fprintf(stderr, "OK packet too large\n");
    // DBUG_PRINT("info", ("OK packet too large"));
    DBUG_RETURN(1);
  }
  error= my_net_write(net, start, (size_t) (pos - start));
  if (!error)
    error= net_flush(net);

  // thd->get_stmt_da()->set_overwrite_status(false);
  // DBUG_PRINT("info", ("OK sent, so no more error sending allowed"));

  DBUG_RETURN(error);
}
int Protocol_classic::read_packet()
{
  int ret;
  if ((packet_length= my_net_read(&m_thd->net)) &&
      packet_length != packet_error)
  {
    DBUG_ASSERT(!m_thd->net.error);
    bad_packet= false;
    raw_packet= m_thd->net.read_pos;
    return 0;
  }
  else if (m_thd->net.error == 3)
    ret= 1;
  else
    ret= -1;
  bad_packet= true;
  return ret;
}

bool Protocol_classic::net_store_data(const uchar *from, size_t length)
{
  /*
     The +9 comes from that strings of length longer than 16M require
     9 bytes to be stored (see net_store_length).
  */
  // if (packet_length+9+length > packet->alloced_length() &&
  //     packet->mem_realloc(packet_length+9+length))
  //   return 1;
  ulonglong tmp;
  uchar *pos = net_store_length((uchar *)&tmp, length);
  size_t tmpLen = (size_t) (pos - (uchar *) &tmp);
  packet->append((uchar*)&tmp, tmpLen);
  packet->append(from, length);
  //memcpy(to,from,length);
  // packet->length((uint) (to+length-(uchar *) packet->ptr()));
  return 0;
}

/**
  Auxilary function to convert string to the given character set
  and store in network buffer.
*/

bool Protocol_classic::store_string_aux(const char *from, size_t length,
                                        const CHARSET_INFO *fromcs,
                                        const CHARSET_INFO *tocs)
{
  /* 'tocs' is set 0 when client issues SET character_set_results=NULL */
  // if (tocs && !my_charset_same(fromcs, tocs) &&
  //     fromcs != &my_charset_bin &&
  //     tocs != &my_charset_bin)
  // {
  //   /* Store with conversion */
  //   return net_store_data((uchar *) from, length, fromcs, tocs);
  // }
  /* Store without conversion */
  return net_store_data((uchar *) from, length);
}

bool
Protocol_classic::start_result_metadata(uint num_cols, uint flags)
{
  ulonglong tmp;
  send_metadata= true;
  // field_count= num_cols;
  sending_flags = flags;
  if (flags & Protocol::SEND_NUM_ROWS)
  {
    uchar *pos = net_store_length((uchar *)&tmp, num_cols);
    my_net_write(&m_thd->net, (uchar *)&tmp, (size_t)(pos - ((uchar *)&tmp)));
  }
  return false;
}

static inline uint32
char_to_byte_length_safe(uint32 char_length_arg, uint32 mbmaxlen_arg)
{
   ulonglong tmp= ((ulonglong) char_length_arg) * mbmaxlen_arg;
   return (tmp > UINT_MAX32) ? (uint32) UINT_MAX32 : (uint32) tmp;
}

bool Protocol_classic::end_row()
{
  // DBUG_ENTER("Protocol_classic::end_row");
  if (m_thd->get_protocol()->connection_alive())
      return my_net_write(&m_thd->net, (uchar *)packet->c_str(),
                          packet->length());
  return 0;
}

bool
Protocol_classic::end_result_metadata()
{
  // DBUG_ENTER("Protocol_classic::end_result_metadata");
  // DBUG_PRINT("info", ("num_cols %u, flags %u", field_count, sending_flags));
  send_metadata= false;
  if (sending_flags & SEND_EOF)
  {
    /* if it is new client do not send EOF packet */
    if (!(has_client_capability(CLIENT_DEPRECATE_EOF)))
    {
      /*
        Mark the end of meta-data result set, and store m_thd->server_status,
        to show that there is no cursor.
        Send no warning information, as it will be sent at statement end.
      */
      if (write_eof_packet(m_thd, &m_thd->net, /* m_thd->server_status */0,
            /* m_thd->get_stmt_da()->current_statement_cond_count()*/0))
      {
        DBUG_RETURN(true);
      }
    }
  }
  DBUG_RETURN(false);
}

/**
  Send eof (= end of result set) to the client.

  The eof packet has the following structure:

  - 254           : Marker (1 byte)
  - warning_count : Stored in 2 bytes; New in 4.1 protocol
  - status_flag   : Stored in 2 bytes;
  For flags like SERVER_MORE_RESULTS_EXISTS.

  Note that the warning count will not be sent if 'no_flush' is set as
  we don't want to report the warning count until all data is sent to the
  client.

  @param thd                    Thread handler
  @param server_status          The server status
  @param statement_warn_count   Total number of warnings

  @return
    @retval FALSE The message was successfully sent
    @retval TRUE An error occurred and the message wasn't sent properly
*/

bool
net_send_eof(THD *thd, uint server_status, uint statement_warn_count)
{
  NET *net= thd->get_protocol_classic()->get_net();
  bool error= FALSE;
  // DBUG_ENTER("net_send_eof");
  /* Set to TRUE if no active vio, to work well in case of --init-file */
  if (net->vio != 0)
  {
    // thd->get_stmt_da()->set_overwrite_status(true);
    error= write_eof_packet(thd, net, server_status, statement_warn_count);
    if (!error)
      error= net_flush(net);
    // thd->get_stmt_da()->set_overwrite_status(false);
    // DBUG_PRINT("info", ("EOF sent, so no more error sending allowed"));
  }
  DBUG_RETURN(error);
}

bool Protocol_classic::flush()
{
    bool error;
    m_thd->get_stmt_da()->set_overwrite_status(true);
    error= net_flush(&m_thd->net);
    m_thd->get_stmt_da()->set_overwrite_status(false);
    return error;
}
/**
  A default implementation of "EOF" packet response to the client.

  Binary and text protocol do not differ in their EOF packet format.
*/

bool Protocol_classic::send_eof(uint server_status, uint statement_warn_count)
{
  // DBUG_ENTER("Protocol_classic::send_eof");
  bool retval;
  /*
    Normally end of statement reply is signaled by OK packet, but in case
    of binlog dump request an EOF packet is sent instead. Also, old clients
    expect EOF packet instead of OK
  */
  if (has_client_capability(CLIENT_DEPRECATE_EOF)/* &&
      (m_thd->get_command() != COM_BINLOG_DUMP &&
       m_thd->get_command() != COM_BINLOG_DUMP_GTID)*/)
    retval= net_send_ok(m_thd, server_status, statement_warn_count, 0, 0, NULL,
                        true);
  else
    retval= net_send_eof(m_thd, server_status, statement_warn_count);
  DBUG_RETURN(retval);
}

/**
  A default implementation of "ERROR" packet response to the client.

  Binary and text protocol do not differ in ERROR packet format.
*/

bool Protocol_classic::send_error(uint sql_errno, const char *err_msg,
                                  const char *sql_state)
{
    // DBUG_ENTER("Protocol_classic::send_error");
    const bool retval= net_send_error_packet(m_thd, sql_errno, err_msg, sql_state);
    DBUG_RETURN(retval);
}


bool Protocol_classic::send_field_metadata(Send_field *field,
                                           const CHARSET_INFO *item_charset)
{
  // DBUG_ENTER("Protocol_classic::send_field_metadata");
  const CHARSET_INFO *cs= default_charset_info;
  // const CHARSET_INFO *thd_charset= m_thd->variables.character_set_results;
  const CHARSET_INFO *thd_charset= default_charset_info;

  /* Keep things compatible for old clients */
  if (field->type == MYSQL_TYPE_VARCHAR)
    field->type= MYSQL_TYPE_VAR_STRING;

  send_metadata= true;
  if (has_client_capability(CLIENT_PROTOCOL_41))
  {
    if (store(STRING_WITH_LEN("def"), cs) ||
        store(field->db_name.data(), strlen(field->db_name.data()), cs) ||
        store(field->table_name.data(), strlen(field->table_name.data()), cs) ||
        store(field->org_table_name.data(), strlen(field->org_table_name.data()), cs) ||
        store(field->col_name.data(), strlen(field->col_name.data()), cs) ||
        store(field->org_col_name.data(), strlen(field->org_col_name.data()), cs))// ||
        // packet->mem_realloc(packet->length() + 12))
    {
      send_metadata= false;
      return true;
    }
    /* Store fixed length fields */
    uchar tmp = 12;
    packet->append(&tmp, 1);
    // *pos++= 12;        // Length of packed fields
    /* inject a NULL to test the client */
    // DBUG_EXECUTE_IF("poison_rs_fields", pos[-1]= (char) 0xfb;);
    // if (item_charset == &my_charset_bin || thd_charset == NULL)
    // {
    //   /* No conversion */
    //   int2store(pos, item_charset->number);
    //   int4store(pos + 2, field->length);
    // }
    // else
    // {
      /* With conversion */
      uint32 field_length, max_length;
      uchar tmpCharset[2];
      int2store(tmpCharset, cs->number);
      packet->append(tmpCharset, 2);
      // int2store(pos, 8);
      // int2store(pos, thd_charset->number);
      /*
        For TEXT/BLOB columns, field_length describes the maximum data
        length in bytes. There is no limit to the number of characters
        that a TEXT column can store, as long as the data fits into
        the designated space.
        For the rest of textual columns, field_length is evaluated as
        char_count * mbmaxlen, where character count is taken from the
        definition of the column. In other words, the maximum number
        of characters here is limited by the column definition.

        When one has a LONG TEXT column with a single-byte
        character set, and the connection character set is multi-byte, the
        client may get fields longer than UINT_MAX32, due to
        <character set column> -> <character set connection> conversion.
        In that case column max length does not fit into the 4 bytes
        reserved for it in the protocol.
      */
      max_length= (field->type >= MYSQL_TYPE_TINY_BLOB &&
                   field->type <= MYSQL_TYPE_BLOB) ?
                   field->length / item_charset->mbminlen :
                   field->length / item_charset->mbmaxlen;
      field_length= char_to_byte_length_safe(max_length, thd_charset->mbmaxlen);
      uchar tmpFieldLen[4];
      int4store(tmpFieldLen, field_length);
      packet->append(tmpFieldLen, 4);
      // int4store(pos + 2, field_length);
    // }
    // pos[6]= field->type;
    packet->append((uchar*)&field->type, 1);
    // int2store(pos + 7, field->flags);
    uchar tmpFlags[2];
    int2store(tmpFlags, field->flags);
    packet->append(tmpFlags, 2);
    // pos[9]= (char) field->decimals;
    // pos[10]= 0;        // For the future
    // pos[11]= 0;        // For the future
    packet->append((uchar*)&field->decimals, 1);
    tmp = 0;
    packet->append((uchar*)&tmp, 1);
    packet->append((uchar*)&tmp, 1);
  }
  else
  {
    if (store(field->table_name.data(), strlen(field->table_name.data()), cs) ||
        store(field->col_name.data(), strlen(field->col_name.data()), cs)) // ||
        // packet->mem_realloc(packet->length() + 10))
    {
      send_metadata= false;
      return true;
    }
    // pos= (char *) packet->data() + packet->length();
    // pos[0]= 3;
    uchar tmp = 3;
    packet->append(&tmp, 1);

    // int3store(pos + 1, field->length);
    uchar tmpLen[3];
    int3store(tmpLen, field->length);
    packet->append(tmpLen, 3);

    // pos[4]= 1;
    tmp = 1;
    packet->append(&tmp, 1);

    // pos[5]= field->type;
    packet->append((uchar*)&field->type, 1);

    // pos[6]= 3;
    tmp = 3;
    packet->append(&tmp, 1);
    
    // int2store(pos + 7, field->flags);
    // pos[9]= (char) field->decimals;
    uchar tmpFlags[2];
    int2store(tmpFlags, field->flags);
    packet->append(tmpFlags, 2);
    packet->append((uchar*)&field->decimals, 1);
  }
  // packet->length((uint) (pos - packet->ptr()));

#ifndef DBUG_OFF
  // TODO: this should be protocol-dependent, as it records incorrect type
  // for binary protocol
  // Text protocol sends fields as varchar
  // field_types[count++]= field->field ? MYSQL_TYPE_VAR_STRING : field->type;
#endif
  DBUG_RETURN(false);
}

/**
  A default implementation of "OK" packet response to the client.

  Currently this implementation is re-used by both network-oriented
  protocols -- the binary and text one. They do not differ
  in their OK packet format, which allows for a significant simplification
  on client side.
*/

bool
Protocol_classic::send_ok(uint server_status, uint statement_warn_count,
                          ulonglong affected_rows, ulonglong last_insert_id,
                          const char *message)
{
  // DBUG_ENTER("Protocol_classic::send_ok");
  const bool retval=
    net_send_ok(m_thd, server_status, statement_warn_count,
                affected_rows, last_insert_id, message, false);
  DBUG_RETURN(retval);
}

/**
  A default implementation of "OK" packet response to the client.

  Currently this implementation is re-used by both network-oriented
  protocols -- the binary and text one. They do not differ
  in their OK packet format, which allows for a significant simplification
  on client side.
*/

bool
Protocol_classic::send_ok(uint server_status, uint statement_warn_count,
                          ulonglong affected_rows, ulonglong last_insert_id,
                          const char *message, const char *session_state_info, int session_state_info_len)
{
  // DBUG_ENTER("Protocol_classic::send_ok");
  const bool retval=
    net_send_ok(m_thd, server_status, statement_warn_count,
                affected_rows, last_insert_id, message, session_state_info, session_state_info_len, false);
  DBUG_RETURN(retval);
}

/**
  Send a error string to client.

  Design note:

  net_printf_error and net_send_error are low-level functions
  that shall be used only when a new connection is being
  established or at server startup.

  For SIGNAL/RESIGNAL and GET DIAGNOSTICS functionality it's
  critical that every error that can be intercepted is issued in one
  place only, my_message_sql.

  @param thd Thread handler
  @param sql_errno The error code to send
  @param err A pointer to the error message

  @return
    @retval FALSE The message was sent to the client
    @retval TRUE An error occurred and the message wasn't sent properly
*/

bool net_send_error(THD *thd, uint sql_errno, const char *err)
{
  bool error;

  DBUG_ASSERT(sql_errno);
  DBUG_ASSERT(err);

  /*
    It's one case when we can push an error even though there
    is an OK or EOF already.
  */
  thd->get_stmt_da()->set_overwrite_status(true);

  /* Abort multi-result sets */
  thd->server_status&= ~SERVER_MORE_RESULTS_EXISTS;

  error= net_send_error_packet(thd, sql_errno, err,
    mysql_errno_to_sqlstate(sql_errno));

  thd->get_stmt_da()->set_overwrite_status(false);

  DBUG_RETURN(error);
}

/**
  Send a error string to client using net struct.
  This is used initial connection handling code.

  @param net        Low-level net struct
  @param sql_errno  The error code to send
  @param err        A pointer to the error message

  @return
    @retval FALSE The message was sent to the client
    @retval TRUE  An error occurred and the message wasn't sent properly
*/

bool net_send_error(NET *net, uint sql_errno, const char *err)
{
  DBUG_ASSERT(sql_errno && err);

  bool error=
    net_send_error_packet(net, sql_errno, err,
                          mysql_errno_to_sqlstate(sql_errno), false, 0,
                          global_system_variables.character_set_results);

  DBUG_RETURN(error);
}

/**
  @param thd          Thread handler
  @param sql_errno    The error code to send
  @param err          A pointer to the error message
  @param sqlstate     SQL state

  @return
   @retval FALSE The message was successfully sent
   @retval TRUE  An error occurred and the messages wasn't sent properly
*/

bool net_send_error_packet(THD *thd, uint sql_errno, const char *err,
                           const char* sqlstate)
{
  return net_send_error_packet(thd->get_protocol_classic()->get_net(),
                               sql_errno, err, sqlstate, false,
                               thd->get_protocol()->get_client_capabilities(),
                               thd->variables.character_set_results);
}


/**
  @param net                    Low-level NET struct
  @param sql_errno              The error code to send
  @param err                    A pointer to the error message
  @param sqlstate               SQL state
  @param bootstrap              Server is started in bootstrap mode
  @param client_capabilities    Client capabilities flag
  @param character_set_results  Char set info

  @return
   @retval FALSE The message was successfully sent
   @retval TRUE  An error occurred and the messages wasn't sent properly
*/

bool net_send_error_packet(NET* net, uint sql_errno, const char *err,
                           const char* sqlstate, bool bootstrap,
                           ulong client_capabilities,
                           const CHARSET_INFO* character_set_results)
{
  uint length;
  /*
    buff[]: sql_errno:2 + ('#':1 + SQLSTATE_LENGTH:5) + MYSQL_ERRMSG_SIZE:512
  */
  // uint error;
  // char converted_err[MYSQL_ERRMSG_SIZE];
  char buff[2+1+SQLSTATE_LENGTH+MYSQL_ERRMSG_SIZE], *pos;

  // DBUG_ENTER("net_send_error_packet");

  if (net->vio == 0)
  {
    if (bootstrap)
    {
      /* In bootstrap it's ok to print on stderr */
      // my_message_local(ERROR_LEVEL, "%d  %s", sql_errno, err);
    }
    DBUG_RETURN(FALSE);
  }

  int2store(buff,sql_errno);
  pos= buff+2;
  if (client_capabilities & CLIENT_PROTOCOL_41)
  {
    /* The first # is to make the protocol backward compatible */
    buff[2]= '#';
    pos= stpcpy(buff+3, sqlstate);
  }

  // convert_error_message(converted_err, sizeof(converted_err),
  //                       character_set_results, err,
  //                       strlen(err), system_charset_info, &error);
  /* Converted error message is always null-terminated. */
  length= (uint) (strmake(pos, err, MYSQL_ERRMSG_SIZE - 1) - buff);

  DBUG_RETURN(net_write_command(net,(uchar) 255, (uchar *) "", 0,
              (uchar *) buff, length));
}

/****************************************************************************
  Functions to handle the binary protocol used with prepared statements

  Data format:

    [ok:1]                            reserved ok packet
    [null_field:(field_count+7+2)/8]  reserved to send null data. The size is
                                      calculated using:
                                      bit_fields= (field_count+7+2)/8;
                                      2 bits are reserved for identifying type
                                      of package.
    [[length]data]                    data field (the length applies only for
                                      string/binary/time/timestamp fields and
                                      rest of them are not sent as they have
                                      the default length that client understands
                                      based on the field type
    [..]..[[length]data]              data
****************************************************************************/
bool Protocol_binary::start_result_metadata(uint num_cols, uint flags/*,
                                            const CHARSET_INFO *result_cs*/)
{
  bit_fields= (num_cols + 9) / 8;
  return Protocol_classic::start_result_metadata(num_cols, flags/*, result_cs*/);
}

void Protocol_binary::start_row()
{
  if (send_metadata)
    return Protocol_text::start_row();
  packet->clear();
  uchar* header = new uchar[bit_fields + 1];
  memset(header, 0, bit_fields + 1);
  packet->append(header, bit_fields + 1);
  delete[] header;
  field_pos=0;
}

bool Protocol_binary::store_date(aries_acc::AriesDate *date)
{
    if(send_metadata)
        return Protocol_text::store_date(date);
#ifndef DBUG_OFF
    // field_types check is needed because of the embedded protocol
    // DBUG_ASSERT(field_types == 0 ||
    //             field_types[field_pos] == MYSQL_TYPE_DATE ||
    //             field_types[field_pos] == MYSQL_TYPE_VAR_STRING);
#endif
    auto dtPtr = std::make_shared<aries_acc::AriesDatetime>(*date);
    return Protocol_binary::store(dtPtr.get(), 0);
}
bool Protocol_binary::store_time(aries_acc::AriesTime *tm, uint precision)
{
    if(send_metadata)
        return Protocol_text::store_time(tm, precision);
    char buff[13], *pos;
    size_t length;
#ifndef DBUG_OFF
    // field_types check is needed because of the embedded protocol
    // DBUG_ASSERT(field_types == 0 ||
    //             field_types[field_pos] == MYSQL_TYPE_TIME ||
    //             field_types[field_pos] == MYSQL_TYPE_VAR_STRING);
#endif
    field_pos++;
    uint days = 0;
    pos= buff+1;
    pos[0]= tm->sign < 0 ? 1 : 0;
    if (tm->hour >= 24)
    {
        /* Fix if we come from Item::send */
        days= tm->hour/24;
        tm->hour-= days*24;
        // tm->day+= days;
    }
    int4store(pos+1, days);
    pos[5]= (uchar) tm->hour;
    pos[6]= (uchar) tm->minute;
    pos[7]= (uchar) tm->second;
    int4store(pos+8, tm->second_part);
    if (tm->second_part)
        length=12;
    else if (tm->hour || tm->minute || tm->second || days)
        length=8;
    else
        length=0;
    buff[0]=(char) length;			// Length is stored first
    packet->append((uchar*)buff, length+1);
    return true;
}
bool Protocol_binary::store(aries_acc::AriesDatetime *tm, uint precision)
{
    if(send_metadata)
        return Protocol_text::store(tm, precision);

#ifndef DBUG_OFF
    // field_types check is needed because of the embedded protocol
    // DBUG_ASSERT(field_types == 0 ||
    //             field_types[field_pos] == MYSQL_TYPE_DATE ||
    //             is_temporal_type_with_date_and_time(field_types[field_pos]) ||
    //             field_types[field_pos] == MYSQL_TYPE_VAR_STRING);
#endif
    char buff[12],*pos;
    size_t length;
    field_pos++;
    pos= buff+1;

    int2store(pos, tm->year);
    pos[2]= (uchar) tm->month;
    pos[3]= (uchar) tm->day;
    pos[4]= (uchar) tm->hour;
    pos[5]= (uchar) tm->minute;
    pos[6]= (uchar) tm->second;
    int4store(pos+7, tm->second_part);
    if (tm->second_part)
        length=11;
    else if (tm->hour || tm->minute || tm->second)
        length=7;
    else if (tm->year || tm->month || tm->day)
        length=4;
    else
        length=0;
    buff[0]=(char) length;			// Length is stored first
    packet->append((uchar*)buff, length+1);
    return false;
}

bool Protocol_binary::store(const char *from, size_t length,
                            const CHARSET_INFO *fromcs,
                            const CHARSET_INFO *tocs)
{
  if(send_metadata)
    return Protocol_text::store(from, length, fromcs, tocs);
  field_pos++;
  return store_string_aux(from, length, fromcs, tocs);
}

bool Protocol_binary::store_null()
{
  if(send_metadata)
    return Protocol_text::store_null();
  uint offset= (field_pos+2)/8+1, bit= (1 << ((field_pos+2) & 7));
  /* Room for this as it's allocated in prepare_for_send */
  char *to= (char *) packet->data()+offset;
  *to= (char) ((uchar) *to | (uchar) bit);
  field_pos++;
  return 0;
}

bool Protocol_binary::store_tiny(longlong from)
{
  if(send_metadata)
    return Protocol_text::store_tiny(from);
  char buff[1];
  field_pos++;
  buff[0]= (uchar) from;
  packet->append((uchar*)buff, sizeof(buff)/*, PACKET_BUFFER_EXTRA_ALLOC*/);
  return false;
}


bool Protocol_binary::store_short(longlong from)
{
  if(send_metadata)
    return Protocol_text::store_short(from);
  field_pos++;
  char to[2];
  int2store(to, (int) from);
  packet->append((uchar*)to, 2);
  return 0;
}


bool Protocol_binary::store_long(longlong from)
{
  if(send_metadata)
    return Protocol_text::store_long(from);
  field_pos++;
  char to[4];
  int4store(to, static_cast<uint32>(from));
  packet->append((uchar*)to, 4);
  return 0;
}


bool Protocol_binary::store_longlong(longlong from, bool unsigned_flag)
{
  if(send_metadata)
    return Protocol_text::store_longlong(from, unsigned_flag);
  field_pos++;
  char to[8];
  int8store(to, from);
  packet->append((uchar*)to, 8);
  return 0;
}

bool Protocol_binary::store(float from, uint32 decimals/*, String *buffer*/)
{
  if(send_metadata)
    return Protocol_text::store(from, decimals);
  field_pos++;
  char to[4];
  float4store(to, from);
  packet->append((uchar*)to, 4);
  return 0;
}

bool Protocol_binary::store(double from, uint32 decimals/*, String *buffer*/)
{
  if(send_metadata)
    return Protocol_text::store(from, decimals);
  field_pos++;
  char to[8];
  float8store(to, from);
  packet->append((uchar*)to, 8);
  return 0;
}

bool net_request_file(NET *net, const string& fname) {
    DBUG_ENTER("net_request_file");
    DBUG_RETURN(net_write_command(net, 251, (uchar *)fname.data(), fname.length(),
                                  (uchar *)"", 0));
}
