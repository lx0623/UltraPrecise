//
// Created by tengjp on 19-7-25.
//

#include <stdio.h>
#include <glog/logging.h>
#include <cstdarg>
#include "./include/log.h"
#include "./include/my_dbug.h"

void sql_print_error(const char *format, ...)
{
    char buff[1024];
    va_list args;
    // DBUG_ENTER("sql_print_error");

    va_start(args, format);
    // error_log_print(ERROR_LEVEL, format, args);
    vsnprintf(buff, 1024, format, args);
    LOG(ERROR) << buff;
    va_end(args);

    DBUG_VOID_RETURN;
}


void sql_print_warning(const char *format, ...)
{
    char buff[1024];
    va_list args;
    // DBUG_ENTER("sql_print_warning");

    va_start(args, format);
    // error_log_print(WARNING_LEVEL, format, args);
    vsnprintf(buff, 1024, format, args);
    LOG(WARNING) << buff;
    va_end(args);

    DBUG_VOID_RETURN;
}


void sql_print_information(const char *format, ...)
{
    char buff[1024];
    va_list args;
    // DBUG_ENTER("sql_print_information");

    va_start(args, format);
    // error_log_print(INFORMATION_LEVEL, format, args);
    vsnprintf(buff, 1024, format, args);
    LOG(INFO) << buff;
    va_end(args);

    DBUG_VOID_RETURN;
}


int error_log_print(enum loglevel level, const char *format,
                    va_list args)
{
    // return logger.error_log_print(level, format, args);
    switch (level) {
        case ERROR_LEVEL:
            sql_print_error(format, args);
            break;
        case WARNING_LEVEL:
            sql_print_warning(format, args);
            break;
        case INFORMATION_LEVEL:
            sql_print_information(format, args);
            break;
    }
    return 0;
}
