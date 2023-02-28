#ifndef PROTOCOL_CLASSIC_INCLUDED
#define PROTOCOL_CLASSIC_INCLUDED

/* Copyright (c) 2002, 2018, Oracle and/or its affiliates. All rights reserved.

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
#include "my_global.h"
#include "m_ctype.h"
#include "mysql_com.h"
#include "my_command.h"
#include "com_data.h"
#include "binary_log_types.h"
#include "protocol.h"                           /* Protocol */
#include "mysqld.h"

class THD;
class Send_field;

bool net_request_file(NET *net, const std::string& fname);

class Protocol_classic : public Protocol
{
public:
    /**
      Flags available to alter the way the messages are sent to the client
    */
    enum
    {
        SEND_NUM_ROWS = 1,
        SEND_DEFAULTS = 2,
        SEND_EOF = 4
    };

private:
    ulong m_client_capabilities;

protected:
    basic_string<uchar>* packet;
    basic_string<uchar>* convert;
    uint sending_flags;
    ulong packet_length;
    uchar *raw_packet;
    bool send_metadata;
    uint field_pos;

public:
    Protocol_classic();
    virtual ~Protocol_classic();
    void init(THD *thd);
    bool bad_packet;
    THD *m_thd;
    CHARSET_INFO *result_cs = default_charset_info;
    // virtual void start_row();
    virtual uint get_rw_status();
    String* get_packet() { return packet; }
    ulong get_client_capabilities() { return m_client_capabilities; }
    void set_client_capabilities(ulong client_capabilities)
    {
        this->m_client_capabilities = client_capabilities;
    }
    /* Adds  client capability */
    void add_client_capability(ulong client_capability)
    {
        m_client_capabilities |= client_capability;
    }
    /* Removes a client capability*/
    void remove_client_capability(unsigned long capability)
    {
        m_client_capabilities &= ~capability;
    }
    /* Returns true if the client has the capability and false otherwise*/
    virtual bool has_client_capability(unsigned long client_capability)
    {
        return (bool) (m_client_capabilities & client_capability);
    }

    void set_read_timeout(ulong read_timeout);
    void set_write_timeout(ulong write_timeout);
    bool init_net(Vio *vio);
    /* Deinitialize NET */
    void end_net();
    /* Flush NET buffer */
    bool flush_net();
    /* Write data to NET buffer */
    bool write(const uchar *ptr, size_t len);
    /* return VIO */
    Vio *get_vio();
    /* Set VIO */
    void set_vio(Vio *vio);
    virtual int read_packet();
    virtual bool send_ok(uint server_status, uint statement_warn_count,
                         ulonglong affected_rows, ulonglong last_insert_id,
                         const char *message);
    bool
    send_ok(uint server_status, uint statement_warn_count,
            ulonglong affected_rows, ulonglong last_insert_id,
            const char *message, const char *sesson_state_info, int session_state_info_len);
    bool send_eof(uint server_status, uint statement_warn_count);
    bool send_error(uint sql_errno, const char *err_msg,
                    const char *sql_state);

    /* return packet length */
    ulong get_packet_length() { return packet_length; }
    /* Return raw packet buffer */
    uchar *get_raw_packet() { return raw_packet; }
    NET* get_net();

    bool
    start_result_metadata(uint num_cols, uint flags);

    bool send_field_metadata(Send_field *field,
                             const CHARSET_INFO *item_charset);

    virtual bool flush();

public:
    int get_command(COM_DATA *, enum_server_command *);
    bool end_row();
    virtual void abort_row(){};
    virtual void end_partial_result_set();
    bool end_result_metadata();
    bool parse_packet(COM_DATA *, enum_server_command);
    bool store_string_aux(const char *from, size_t length,
                          const CHARSET_INFO *fromcs,
                          const CHARSET_INFO *tocs);
    bool net_store_data(const uchar *from, size_t length);

    /**
      Returns the type of the connection

      @return
        enum enum_vio_type
    */
    virtual enum enum_vio_type connection_type()
    {
        Vio *v = get_vio();
        return v ? vio_type(v) : NO_VIO_TYPE;
    }
    /* Check whether VIO is healhty */
    virtual bool connection_alive();
    /* Deinitialize VIO */
    virtual int shutdown(bool server_shutdown= false);
};

/** Class used for the old (MySQL 4.0 protocol). */

class Protocol_text : public Protocol_classic
{
public:
    Protocol_text() {}
    // Protocol_text(THD *thd) : Protocol_classic(thd) {}
    virtual bool store_null();
    virtual bool store_tiny(longlong from);
    virtual bool store_short(longlong from);
    virtual bool store_long(longlong from);
    virtual bool store_longlong(longlong from, bool unsigned_flag);
    // virtual bool store_decimal(const my_decimal *, uint, uint);
    virtual bool store(const char *from, size_t length, const CHARSET_INFO *cs)
    { return store(from, length, cs, result_cs); }
    virtual bool store(float nr, uint32 decimals/*, String *buffer*/);
    virtual bool store(double from, uint32 decimals/*, String *buffer*/);
    virtual bool store(aries_acc::AriesDatetime *datetime, uint precision);
    virtual bool store_date(aries_acc::AriesDate *date);
    virtual bool store_time(aries_acc::AriesTime *tm, uint decimals);
    // virtual bool store(Proto_field *field);
    virtual void start_row();

    // virtual bool send_out_parameters(List<Item_param> *sp_params);
    virtual enum enum_protocol_type type() { return PROTOCOL_TEXT; };
protected:
    virtual bool store(const char *from, size_t length,
                       const CHARSET_INFO *fromcs,
                       const CHARSET_INFO *tocs);
};


class Protocol_binary : public Protocol_text
{
private:
    uint bit_fields;
public:
    Protocol_binary() {}
    // Protocol_binary(THD *thd_arg) :Protocol_text(thd_arg) {}
    virtual void start_row();
    virtual bool store_null();
    virtual bool store_tiny(longlong from);
    virtual bool store_short(longlong from);
    virtual bool store_long(longlong from);
    virtual bool store_longlong(longlong from, bool unsigned_flag);
    // virtual bool store_decimal(const my_decimal *, uint, uint);
    // virtual bool store(MYSQL_TIME *time, uint precision);
    virtual bool store(aries_acc::AriesDatetime *datetime, uint precision);
    virtual bool store_date(aries_acc::AriesDate *date);
    virtual bool store_time(aries_acc::AriesTime *tm, uint decimals);
    virtual bool store(float nr, uint32 decimals/*, String *buffer*/);
    virtual bool store(double from, uint32 decimals/*, String *buffer*/);
    // virtual bool store(Proto_field *field);
    virtual bool store(const char *from, size_t length, const CHARSET_INFO *cs)
    { return store(from, length, cs, result_cs); }

    // virtual bool send_out_parameters(List<Item_param> *sp_params);
    virtual bool start_result_metadata(uint num_cols, uint flags/*,
                                     const CHARSET_INFO *resultcs*/);

    virtual enum enum_protocol_type type() { return PROTOCOL_BINARY; };
protected:
    virtual bool store(const char *from, size_t length,
                       const CHARSET_INFO *fromcs,
                       const CHARSET_INFO *tocs);
};
#endif
