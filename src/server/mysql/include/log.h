//
// Created by tengjp on 19-7-25.
//

#ifndef AIRES_LOG_H
#define AIRES_LOG_H

#include "my_global.h"
void sql_print_error(const char *format, ...) ;
void sql_print_warning(const char *format, ...) ;
void sql_print_information(const char *format, ...);
int error_log_print(enum loglevel level, const char *format,
                    va_list args);

#endif //AIRES_LOG_H
