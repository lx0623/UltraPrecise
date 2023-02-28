//
// Created by tengjp on 19-8-14.
//

#ifndef AIRES_SQL_TIME_H
#define AIRES_SQL_TIME_H

#include "my_global.h"
#include "m_string.h"

struct Date_time_format
{
    uchar positions[8];
    char  time_separator;			/* Separator between hour and minute */
    uint flag;				/* For future */
    LEX_STRING format;
};

struct Interval
{
    ulong year, month, day, hour;
    ulonglong minute, second, second_part;
    bool neg;
};

struct Known_date_time_format
{
    const char *format_name;
    const char *date_format;
    const char *datetime_format;
    const char *time_format;
};
enum date_time_format
{
    USA_FORMAT, JIS_FORMAT, ISO_FORMAT, EUR_FORMAT, INTERNAL_FORMAT
};

extern Date_time_format global_date_format;
extern Date_time_format global_datetime_format;
extern Date_time_format global_time_format;
extern Known_date_time_format known_date_time_formats[];

#endif //AIRES_SQL_TIME_H
