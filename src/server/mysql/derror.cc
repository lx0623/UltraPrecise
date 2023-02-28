//
// Created by tengjp on 19-7-15.
//
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <glog/logging.h>
#include <server/mysql/include/sql_class.h>
#include <server/mysql/include/mysqld.h>
#include <server/mysql/include/my_sys.h>
#include <server/mysql/include/mysys_err.h>

#include "server/mysql/include/derror.h"
// #include "server/mysql/include/my_global.h"
#include "server/mysql/include/mysqld_error.h"
#include "server/mysql/include/my_byteorder.h"

#include "utils/string_util.h"

static const char *ERRMSG_FILE = "errmsg.sys";
const char* ERRMSG_SYNTAX_ERROR = "You have an error in your SQL syntax.";
static const int NUM_SECTIONS=
        sizeof(errmsg_section_start) / sizeof(errmsg_section_start[0]);

extern void my_message_sql(uint error, const char *str, int MyFlags);

const char* ER_THD(const THD *thd, int mysql_errno)
{
    return thd->variables.lc_messages->lookup(mysql_errno);
}

/**
  All global error messages are sent here where the first one is stored
  for the client.
*/
/* ARGSUSED */
void my_message_sql(uint error, const char *str, int MyFlags) {
    THD *thd = current_thd;
    if (thd) {
        Sql_condition::enum_severity_level level= Sql_condition::SL_ERROR;
        if (MyFlags & ME_FATALERROR)
            thd->is_fatal_error= 1;
        (void) thd->raise_condition(error, NULL, level, str, false);
    }
}
// get error message with customized format
std::string format_err_msg(const char *format, ...)
{
    va_list args;
    char ebuff[ERRMSGSIZE];

    va_start(args, format);
    (void) vsnprintf(ebuff, sizeof(ebuff), format, args);
    va_end(args);
    return std::string(ebuff);
}
std::string format_mysql_err_msg(int nr, ...)
{
    const char *format;
    va_list args;
    char ebuff[ERRMSGSIZE];

    if (!(format = MY_LOCALE_ERRMSGS::getInstance()->lookup(nr)))
        (void) snprintf(ebuff, sizeof(ebuff), "Unknown error %d", nr);
    else
    {
        va_start(args, nr);
        (void) vsnprintf(ebuff, sizeof(ebuff), format, args);
        va_end(args);
    }
    return std::string(ebuff);
}
/**
  Fill in and print a previously registered error message.

  @note
    Goes through the (sole) function registered in error_handler_hook

  @param nr        error number
  @param MyFlags   Flags
  @param ...       variable list matching that error format string
*/

void my_error(int nr, int MyFlags, ...)
{
    const char *format;
    va_list args;
    char ebuff[ERRMSGSIZE];

    if (!(format = MY_LOCALE_ERRMSGS::getInstance()->lookup(nr)))
        (void) snprintf(ebuff, sizeof(ebuff), "Unknown error %d", nr);
    else
    {
        va_start(args,MyFlags);
        (void) vsnprintf(ebuff, sizeof(ebuff), format, args);
        va_end(args);
    }
    // (*error_handler_hook)(nr, ebuff, MyFlags);
    my_message_sql(nr, ebuff, MyFlags);
    DBUG_VOID_RETURN;
}

/**
 * 和上面的my_error类似，但是不需要用error number去找对应的format，args中错误信息已经包含format信息
  @param nr        error number
  @param MyFlags   Flags
  @param ...       full error message
*/
void my_error2(int nr, int MyFlags, ...)
{
    va_list args;
    char ebuff[ERRMSGSIZE];

    va_start(args, MyFlags);
    (void) vsnprintf(ebuff, sizeof(ebuff), "%s", args);
    va_end(args);

    my_message_sql(nr, ebuff, MyFlags);
    DBUG_VOID_RETURN;
}

/**
  Print an error message.

  @note
    Goes through the (sole) function registered in error_handler_hook

  @param error     error number
  @param format    format string
  @param MyFlags   Flags
  @param ...       variable list matching that error format string
*/

void my_printf_error(uint error, const char *format, myf MyFlags, ...)
{
    va_list args;
    char ebuff[ERRMSGSIZE];

    va_start(args,MyFlags);
    (void) vsnprintf(ebuff, sizeof(ebuff), format, args);
    va_end(args);
    my_message_sql(error, ebuff, MyFlags);
    DBUG_VOID_RETURN;
}

/**
  Print an error message.

  @note
    Goes through the (sole) function registered in error_handler_hook

  @param error     error number
  @param str       error message
  @param MyFlags   Flags
*/

void my_message(uint error, const char *str, myf MyFlags)
{
    my_message_sql(error, str, MyFlags);
}

MY_LOCALE_ERRMSGS* MY_LOCALE_ERRMSGS::instance = NULL;
bool MY_LOCALE_ERRMSGS::Init() {
    instance = new MY_LOCALE_ERRMSGS();
    return instance->read_texts();
}
const char* MY_LOCALE_ERRMSGS::lookup(int mysql_errno) {
    int offset= 0; // Position where the current section starts in the array.
    for (int i= 0; i < NUM_SECTIONS; i++)
    {
        if (mysql_errno >= errmsg_section_start[i] &&
            mysql_errno < (errmsg_section_start[i] + errmsg_section_size[i]))
            return errmsgs[mysql_errno - errmsg_section_start[i] + offset];
        offset+= errmsg_section_size[i];
    }
    if ( mysql_errno >= EE_ERROR_FIRST && mysql_errno <= EE_ERROR_LAST )
    {
        return get_global_errmsg( mysql_errno );
    }

    return nullptr;
}

/**
  Read text from packed textfile in language-directory.

  @retval false          On success
  @retval true           On failure

  @note If we can't read messagefile then it's panic- we can't continue.
*/

bool MY_LOCALE_ERRMSGS::read_texts()
{
    uint i;
    uint no_of_errmsgs;
    size_t length;
    int file;
    // char name[FN_REFLEN];
    // char lang_path[FN_REFLEN];
    uchar *start_of_errmsgs= NULL;
    uchar *pos= NULL;
    uchar head[32];
    uint error_messages= 0;

    for (int i= 0; i < NUM_SECTIONS; i++)
        error_messages+= errmsg_section_size[i];

    // convert_dirname(lang_path, language, NullS);
    // (void) my_load_path(lang_path, lang_path, lc_messages_dir);
    auto cwd = aries_utils::get_current_work_directory();
    auto errMsgFilePath = cwd + "/" + ERRMSG_FILE;
    file= open(errMsgFilePath.data(), O_RDONLY | O_SHARE | O_BINARY);
    if (file < 0)
    {
        LOG(ERROR) << "Can't find error-message file: " << errMsgFilePath;
        goto open_err;
    }

    // Read the header from the file
    if (-1 == read(file, (uchar*) head, 32)) {
        int err = errno;
        fprintf(stderr, "read error: %d, %s\n", err, strerror(err));
        goto read_err;
    }
    if (head[0] != (uchar) 254 || head[1] != (uchar) 254 ||
        head[2] != 3 || head[3] != 1 || head[4] != 1)
        goto read_err;

    // error_message_charset_info= system_charset_info;
    length= uint4korr(head+6);
    no_of_errmsgs= uint4korr(head+10);

    if (no_of_errmsgs < error_messages)
    {
        LOG(ERROR) << "Error message file " << ERRMSG_FILE << " had only " << no_of_errmsgs
                   << "error messages,\nbut it should contain at least "
                   << error_messages << " error messages.\nCheck that the above file is the right version for this program!";
        (void) close(file);
        goto open_err;
    }

    // Free old language and allocate for the new one
    free(errmsgs);
    if (!(errmsgs= (const char**)malloc(length+no_of_errmsgs*sizeof(char*))))
    {
        LOG(ERROR) << "Not enough memory for messagefile " << ERRMSG_FILE;
        (void) close(file);
        DBUG_RETURN(true);
    }

    // Get pointer to Section2.
    start_of_errmsgs= (uchar*) (errmsgs + no_of_errmsgs);

    /*
      Temporarily read message offsets into Section2.
      We cannot read these 4 byte offsets directly into Section1,
      as pointer size vary between processor architecture.
    */
    if (-1 == read(file, start_of_errmsgs, (size_t) no_of_errmsgs*4))
        goto read_err_init;

    // Copy the message offsets to Section1.
    for (i= 0, pos= start_of_errmsgs; i< no_of_errmsgs; i++)
    {
        errmsgs[i]= (char*) start_of_errmsgs+uint4korr(pos);
        pos+= 4;
    }

    // Copy all the error text messages into Section2.
    if (-1 == read(file, start_of_errmsgs, length))
        goto read_err_init;

    (void) close(file);
    DBUG_RETURN(false);

    read_err_init:
    for (uint i= 0; i < error_messages; ++i)
        errmsgs[i]= "";

read_err:
    LOG(ERROR) << "Can't read from messagefile " <<  ERRMSG_FILE;
    (void) close(file);
    open_err:
    if (!errmsgs)
    {
        /*
          Allocate and initialize errmsgs to empty string in order to avoid access
          to errmsgs during another failure in abort operation
        */
        if ((errmsgs= (const char**) malloc(error_messages * sizeof(char*))))
        {
            for (uint i= 0; i < error_messages; ++i)
                errmsgs[i]= "";
        }
    }
    DBUG_RETURN(true);
} /* read_texts */
