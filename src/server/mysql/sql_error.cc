#include <string.h>
#include "./include/mysqld.h"
#include "./include/m_ctype.h"
#include "./include/sql_error.h"
#include "./include/derror.h"

char *strmake(char *dst, const char *src, size_t length);
Sql_condition::Sql_condition() :
        m_message_text(),
        m_mysql_errno(0),
        m_severity_level(Sql_condition::SL_ERROR) {

}
/**
Constructor.

@param mem_root          Memory root to use for the condition items
                        of this condition.
@param mysql_errno       MYSQL_ERRNO
@param returned_sqlstate RETURNED_SQLSTATE
@param severity          Severity level - error, warning or note.
@param message_Text      MESSAGE_TEXT
*/
Sql_condition::Sql_condition(uint mysql_errno,
                             const char* returned_sqlstate,
                             Sql_condition::enum_severity_level severity,
                             const char *message_text) :
        m_message_text(message_text),
        m_mysql_errno(mysql_errno),
        m_severity_level(severity){
    set_returned_sqlstate(returned_sqlstate);
}

Diagnostics_area::Diagnostics_area(bool allow_unlimited_conditions)
        :// m_stacked_da(NULL),
         m_is_sent(false),
         m_can_overwrite_status(false),
         // m_allow_unlimited_conditions(allow_unlimited_conditions),
         m_status(DA_EMPTY),
         m_mysql_errno(0),
         m_affected_rows(0),
         m_last_insert_id(0)
         // m_last_statement_cond_count(0),
         // m_current_statement_cond_count(0),
         // m_current_row_for_condition(1),
         // m_saved_error_count(0),
         // m_saved_warn_count(0)
{
    /* Initialize sub structures */
    m_message_text[0]= '\0';
}

Diagnostics_area::~Diagnostics_area()
{
    // free_root(&m_condition_root,MYF(0));
}

void Diagnostics_area::reset_diagnostics_area()
{
    set_overwrite_status(false);
    // Don't take chances in production.
    m_message_text[0]= '\0';
    m_mysql_errno= 0;
    m_affected_rows= 0;
    m_last_insert_id= 0;
    set_is_sent(false);
    // Tiny reset in debug mode to see garbage right away.
    m_status= DA_EMPTY;
    DBUG_VOID_RETURN;
}


void Diagnostics_area::set_ok_status(ulonglong affected_rows,
                                     ulonglong last_insert_id,
                                     const char *message_text)
{
    DBUG_ASSERT(! is_set());
    /*
      In production, refuse to overwrite an error or a custom response
      with an OK packet.
    */
    if (is_error() || is_disabled())
        DBUG_VOID_RETURN;

    m_affected_rows= affected_rows;
    m_last_insert_id= last_insert_id;
    if (message_text)
        strmake(m_message_text, message_text, sizeof(m_message_text) - 1);
    else
        m_message_text[0]= '\0';
    m_status= DA_OK;
    DBUG_VOID_RETURN;
}


void Diagnostics_area::set_eof_status(THD *thd)
{
    /* Only allowed to report eof if has not yet reported an error */
    DBUG_ASSERT(! is_set());
    /*
      In production, refuse to overwrite an error or a custom response
      with an EOF packet.
    */
    if (is_error() || is_disabled())
        DBUG_VOID_RETURN;

    /*
      If inside a stored procedure, do not return the total
      number of warnings, since they are not available to the client
      anyway.
    */
    // m_last_statement_cond_count= (thd->sp_runtime_ctx ?
    //                               0 :
    //                               current_statement_cond_count());

    m_status= DA_EOF;
    DBUG_VOID_RETURN;
}


void Diagnostics_area::set_error_status(uint mysql_errno)
{
    set_error_status(mysql_errno,
                     ER(mysql_errno),
                     mysql_errno_to_sqlstate(mysql_errno));
}


void Diagnostics_area::set_error_status(uint mysql_errno,
                                        const char *message_text,
                                        const char *returned_sqlstate)
{
    /*
      Only allowed to report error if has not yet reported a success
      The only exception is when we flush the message to the client,
      an error can happen during the flush.
    */
    DBUG_ASSERT(! is_set() || m_can_overwrite_status);

    // message must be set properly by the caller.
    DBUG_ASSERT(message_text);

    // sqlstate must be set properly by the caller.
    DBUG_ASSERT(returned_sqlstate);

#ifdef DBUG_OFF
    /*
    In production, refuse to overwrite a custom response with an
    ERROR packet.
  */
  if (is_disabled())
    DBUG_VOID_RETURN;
#endif

    m_mysql_errno= mysql_errno;
    memcpy(m_returned_sqlstate, returned_sqlstate, SQLSTATE_LENGTH);
    m_returned_sqlstate[SQLSTATE_LENGTH]= '\0';
    strmake(m_message_text, message_text, sizeof(m_message_text)-1);

    m_status= DA_ERROR;
    DBUG_VOID_RETURN;
}

/**
   Convert string for dispatch to client(see WL#751).

   @param to          buffer to convert
   @param to_length   buffer length
   @param to_cs       chraset to convert
   @param from        string from convert
   @param from_length string length
   @param from_cs     charset from convert
   @param errors      count of errors during convertion

   @retval
   length of converted string
*/

// size_t convert_error_message(char *to, size_t to_length,
//                              const CHARSET_INFO *to_cs,
//                              const char *from, size_t from_length,
//                              const CHARSET_INFO *from_cs, uint *errors)
// {
//   int         cnvres;
//   my_wc_t     wc;
//   const uchar *from_end= (const uchar*) from+from_length;
//   char *to_start= to;
//   uchar *to_end;
//   // my_charset_conv_mb_wc mb_wc= from_cs->cset->mb_wc;
//   my_charset_conv_wc_mb wc_mb;
//   uint error_count= 0;
//   size_t length;
// 
//   DBUG_ASSERT(to_length > 0);
//   /* Make room for the null terminator. */
//   to_length--;
//   to_end= (uchar*) (to + to_length);
// 
//   if (!to_cs || from_cs == to_cs || to_cs == &my_charset_bin)
//   {
//     length= MY_MIN(to_length, from_length);
//     memmove(to, from, length);
//     to[length]= 0;
//     return length;
//   }
// 
//   wc_mb= to_cs->cset->wc_mb;
//   while (1)
//   {
//     if ((cnvres= (*mb_wc)(from_cs, &wc, (uchar*) from, from_end)) > 0)
//     {
//       if (!wc)
//         break;
//       from+= cnvres;
//     }
//     else if (cnvres == MY_CS_ILSEQ)
//     {
//       wc= (ulong) (uchar) *from;
//       from+=1;
//     }
//     else
//       break;
// 
//     if ((cnvres= (*wc_mb)(to_cs, wc, (uchar*) to, to_end)) > 0)
//       to+= cnvres;
//     else if (cnvres == MY_CS_ILUNI)
//     {
//       length= (wc <= 0xFFFF) ? 6/* '\1234' format*/ : 9 /* '\+123456' format*/;
//       if ((uchar*)(to + length) >= to_end)
//         break;
//       cnvres= snprintf(to, 9,
//                           (wc <= 0xFFFF) ? "\\%04X" : "\\+%06X", (uint) wc);
//       to+= cnvres;
//     }
//     else
//       break;
//   }
// 
//   *to= 0;
//   *errors= error_count;
//   return (uint32) (to - to_start);
// }