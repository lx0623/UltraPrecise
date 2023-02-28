//
// Created by tengjp on 19-7-16.
//

#ifndef AIRES_SQL_ERROR_H
#define AIRES_SQL_ERROR_H

#include <string>
#include "mysql_com.h" /* MYSQL_ERRMSG_SIZE */

class THD;

/**
  Representation of a SQL condition.
  A SQL condition can be a completion condition (note, warning),
  or an exception condition (error, not found).
*/
class Sql_condition {
public:
    /**
      Enumeration value describing the severity of the condition.
    */
    enum enum_severity_level {
        SL_NOTE, SL_WARNING, SL_ERROR, SEVERITY_END
    };

    Sql_condition();
    /**
   Constructor.

   @param mem_root          Memory root to use for the condition items
                            of this condition.
   @param mysql_errno       MYSQL_ERRNO
   @param returned_sqlstate RETURNED_SQLSTATE
   @param severity          Severity level - error, warning or note.
   @param message_Text      MESSAGE_TEXT
 */
    Sql_condition(uint mysql_errno,
                  const char* returned_sqlstate,
                  Sql_condition::enum_severity_level severity,
                  const char *message_text);

    /** Destructor. */
    ~Sql_condition()
    {}

    /** Set the RETURNED_SQLSTATE of this condition. */
    void set_returned_sqlstate(const char* sqlstate)
    {
        memcpy(m_returned_sqlstate, sqlstate, SQLSTATE_LENGTH);
        m_returned_sqlstate[SQLSTATE_LENGTH]= '\0';
    }

private:

    /** Message text, expressed in the character set implied by --language. */
    std::string m_message_text;

    /** MySQL extension, MYSQL_ERRNO condition item. */
    uint m_mysql_errno;

    /**
    SQL RETURNED_SQLSTATE condition item.
    This member is always NUL terminated.
  */
    char m_returned_sqlstate[SQLSTATE_LENGTH+1];

    /** Severity (error, warning, note) of this condition. */
    Sql_condition::enum_severity_level m_severity_level;
};

/**
  Stores status of the currently executed statement.
  Cleared at the beginning of the statement, and then
  can hold either OK, ERROR, or EOF status.
  Can not be assigned twice per statement.
*/
class Diagnostics_area
{
public:
    enum enum_diagnostics_status
    {
        /** The area is cleared at start of a statement. */
                DA_EMPTY= 0,
        /** Set whenever one calls my_ok(). */
                DA_OK,
        /** Set whenever one calls my_eof(). */
                DA_EOF,
        /** Set whenever one calls my_error() or my_message(). */
                DA_ERROR,
        /** Set in case of a custom response, such as one from COM_STMT_PREPARE. */
                DA_DISABLED
    };

    Diagnostics_area(bool allow_unlimited_conditions);
    ~Diagnostics_area();

    void set_overwrite_status(bool can_overwrite_status)
    { m_can_overwrite_status= can_overwrite_status; }

    bool is_sent() const { return m_is_sent; }

    void set_is_sent(bool is_sent) { m_is_sent= is_sent; }

    /**
      Set OK status -- ends commands that do not return a
      result set, e.g. INSERT/UPDATE/DELETE.

      @param affected_rows  The number of rows affected by the last statement.
                            @sa Diagnostics_area::m_affected_rows.
      @param last_insert_id The value to be returned by LAST_INSERT_ID().
                            @sa Diagnostics_area::m_last_insert_id.
      @param message_text   The OK-message text.
    */
    void set_ok_status(ulonglong affected_rows,
                       ulonglong last_insert_id,
                       const char *message_text);

    /**
      Set EOF status.

      @param thd  Thread context.
    */
    void set_eof_status(THD *thd);

    /**
      Set ERROR status in the Diagnostics Area. This function should be used to
      report fatal errors (such as out-of-memory errors) when no further
      processing is possible.

      @param mysql_errno      SQL-condition error number
    */
    void set_error_status(uint mysql_errno);

    /**
      Set ERROR status in the Diagnostics Area.

      @param mysql_errno        SQL-condition error number
      @param message_text       SQL-condition message
      @param returned_sqlstate  SQL-condition state
    */
    void set_error_status(uint mysql_errno,
                          const char *message_text,
                          const char *returned_sqlstate);

    /**
      Mark the Diagnostics Area as 'DISABLED'.

      This is used in rare cases when the COM_ command at hand sends a response
      in a custom format. One example is the query cache, another is
      COM_STMT_PREPARE.
    */
    void disable_status()
    {
        DBUG_ASSERT(m_status == DA_EMPTY);
        m_status= DA_DISABLED;
    }

    /**
      Clear this Diagnostics Area.

      Normally called at the end of a statement.
    */
    void reset_diagnostics_area();

    bool is_set() const { return m_status != DA_EMPTY; }

    bool is_error() const { return m_status == DA_ERROR; }

    bool is_eof() const { return m_status == DA_EOF; }

    bool is_ok() const { return m_status == DA_OK; }

    bool is_disabled() const { return m_status == DA_DISABLED; }

    enum_diagnostics_status status() const { return m_status; }

    const char *message_text() const
    {
        DBUG_ASSERT(m_status == DA_ERROR || m_status == DA_OK);
        return m_message_text;
    }

    uint mysql_errno() const
    {
        DBUG_ASSERT(m_status == DA_ERROR);
        return m_mysql_errno;
    }

    const char* returned_sqlstate() const
    {
        DBUG_ASSERT(m_status == DA_ERROR);
        return m_returned_sqlstate;
    }

    ulonglong affected_rows() const
    {
        DBUG_ASSERT(m_status == DA_OK);
        return m_affected_rows;
    }

    ulonglong last_insert_id() const
    {
        DBUG_ASSERT(m_status == DA_OK);
        return m_last_insert_id;
    }

private:
    /** True if status information is sent to the client. */
    bool m_is_sent;

    /** Set to make set_error_status after set_{ok,eof}_status possible. */
    bool m_can_overwrite_status;

    enum_diagnostics_status m_status;

private:
    /*
     This section contains basic attributes of Sql_condition to store
     information about error (SQL-condition of error severity) or OK-message.
     The attributes are inlined here (instead of using Sql_condition) to be able
     to store the information in case of out-of-memory error.
   */

    /**
      Message buffer. It is used only when DA is in OK or ERROR status.
      If DA status is ERROR, it's the MESSAGE_TEXT attribute of SQL-condition.
      If DA status is OK, it's the OK-message to be sent.
    */
    char m_message_text[MYSQL_ERRMSG_SIZE];

    /**
    SQL RETURNED_SQLSTATE condition item.
    This member is always NUL terminated.
  */
    char m_returned_sqlstate[SQLSTATE_LENGTH+1];

    /**
    SQL error number. One of ER_ codes from share/errmsg.txt.
    Set by set_error_status.
  */
    uint m_mysql_errno;

    /**
      The number of rows affected by the last statement. This is
      semantically close to thd->row_count_func, but has a different
      life cycle. thd->row_count_func stores the value returned by
      function ROW_COUNT() and is cleared only by statements that
      update its value, such as INSERT, UPDATE, DELETE and few others.
      This member is cleared at the beginning of the next statement.

      We could possibly merge the two, but life cycle of thd->row_count_func
      can not be changed.
    */
    ulonglong m_affected_rows;

    /**
      Similarly to the previous member, this is a replacement of
      thd->first_successful_insert_id_in_prev_stmt, which is used
      to implement LAST_INSERT_ID().
    */
    ulonglong m_last_insert_id;

    friend class THD;
};

#endif //AIRES_SQL_ERROR_H
