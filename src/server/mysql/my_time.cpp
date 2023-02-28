/* Copyright (c) 2004, 2016, Oracle and/or its affiliates. All rights reserved.

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

/*
  Convert datetime to a string 'YYYY-MM-DD hh:mm:ss'.
  Open coded for better performance.
  This code previously resided in field.cc, in Field_timestamp::val_str().

  @param  to     OUT  The string pointer to print at.
  @param  ltime       The MYSQL_TIME value.
  @return             The length of the result string.
*/
#include "./include/my_dbug.h"
#include "./include/my_global.h"
#include "./include/binary_log_types.h"
#include "./include/my_time.h"

using namespace aries_acc;

ulonglong log_10_int[20]=
        {
                1, 10, 100, 1000, 10000UL, 100000UL, 1000000UL, 10000000UL,
                100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL,
                1000000000000ULL, 10000000000000ULL, 100000000000000ULL,
                1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL,
                1000000000000000000ULL, 10000000000000000000ULL
        };

/**
  Print the microsecond part: ".NNN"
  @param to        OUT The string pointer to print at
  @param useconds      The microseconds value.
  @param dec           Precision, between 1 and 6.
  @return              The length of the result string.
*/
static inline int
my_useconds_to_str(char *to, ulong useconds, uint dec)
{
    DBUG_ASSERT(dec <= DATETIME_MAX_DECIMALS);
    return sprintf(to, ".%0*lu", (int) dec,
                   useconds / (ulong) log_10_int[DATETIME_MAX_DECIMALS - dec]);
}

/*
  Functions to convert time/date/datetime value to a string,
  using default format.
  This functions don't check that given MYSQL_TIME structure members are
  in valid range. If they are not, return value won't reflect any
  valid date either. Additionally, make_time doesn't take into
  account time->day member: it's assumed that days have been converted
  to hours already.

  RETURN
    number of characters written to 'to'
*/

int mysql_time_to_str(const AriesTime *l_time, char *to, uint dec)
{
    uint extra_hours= 0;
    int len= sprintf(to, "%s%02u:%02u:%02u", (l_time->sign < 0 ? "-" : ""),
                     extra_hours + l_time->hour, l_time->minute, l_time->second);
    if (dec)
        len+= my_useconds_to_str(to + len, l_time->second_part, dec);
    return len;
}

static inline int
TIME_to_datetime_str(char *to, const AriesDatetime *ltime)
{
    uint32 temp, temp2;
    /* Year */
    temp= ltime->year / 100;
    *to++= (char) ('0' + temp / 10);
    *to++= (char) ('0' + temp % 10);
    temp= ltime->year % 100;
    *to++= (char) ('0' + temp / 10);
    *to++= (char) ('0' + temp % 10);
    *to++= '-';
    /* Month */
    temp= ltime->month;
    temp2= temp / 10;
    temp= temp-temp2 * 10;
    *to++= (char) ('0' + (char) (temp2));
    *to++= (char) ('0' + (char) (temp));
    *to++= '-';
    /* Day */
    temp= ltime->day;
    temp2= temp / 10;
    temp= temp - temp2 * 10;
    *to++= (char) ('0' + (char) (temp2));
    *to++= (char) ('0' + (char) (temp));
    *to++= ' ';
    /* Hour */
    temp= ltime->hour;
    temp2= temp / 10;
    temp= temp - temp2 * 10;
    *to++= (char) ('0' + (char) (temp2));
    *to++= (char) ('0' + (char) (temp));
    *to++= ':';
    /* Minute */
    temp= ltime->minute;
    temp2= temp / 10;
    temp= temp - temp2 * 10;
    *to++= (char) ('0' + (char) (temp2));
    *to++= (char) ('0' + (char) (temp));
    *to++= ':';
    /* Second */
    temp= ltime->second;
    temp2=temp / 10;
    temp= temp - temp2 * 10;
    *to++= (char) ('0' + (char) (temp2));
    *to++= (char) ('0' + (char) (temp));
    return 19;
}


/**
  Print a datetime value with an optional fractional part.

  @l_time       The MYSQL_TIME value to print.
  @to      OUT  The string pointer to print at.
  @return       The length of the result string.
*/
int mysql_datetime_to_str(const AriesDatetime *l_time, char *to, uint dec)
{
    int len= TIME_to_datetime_str(to, l_time);
    if (dec)
        len+= my_useconds_to_str(to + len, l_time->second_part, dec);
    else
        to[len]= '\0';
    return len;
}
