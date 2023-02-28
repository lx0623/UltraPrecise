// 从 mysql 5.7 移植
// Created by david shen on 2019-07-22.
//

#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <map>
#include "timefunc.h"
#include "AriesIntervalTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    /*
      The bits in week_format has the following meaning:
       WEEK_MONDAY_FIRST (0)  If not set	Sunday is first day of week
                           If set	Monday is first day of week
       WEEK_YEAR (1)	  If not set	Week is in range 0-53

           Week 0 is returned for the the last week of the previous year (for
        a date at start of january) In this case one can get 53 for the
        first week of next year.  This flag ensures that the week is
        relevant for the given year. Note that this flag is only
        releveant if WEEK_JANUARY is not set.

                  If set	 Week is in range 1-53.

        In this case one may get week 53 for a date in January (when
        the week is that last week of previous year) and week 1 for a
        date in December.

      WEEK_FIRST_WEEKDAY (2)  If not set	Weeks are numbered according
                           to ISO 8601:1988
                  If set	The week that contains the first
                        'first-day-of-week' is week 1.

        ISO 8601:1988 means that if the week containing January 1 has
        four or more days in the new year, then it is week 1;
        Otherwise it is the last week of the previous year, and the
        next week is week 1.
    */

    uint32_t calc_week(MYSQL_TIME *l_time, uint32_t week_behaviour, uint32_t *year) {
        return calc_week(l_time->year, l_time->month, l_time->day, week_behaviour, year);
    }

    /* Functions to handle periods */

    uint64_t convert_period_to_month(uint64_t period) {
        uint64_t a, b;
        if (period == 0)
            return 0L;
        if ((a = period / 100) < YY_PART_YEAR)
            a += 2000;
        else if (a < 100)
            a += 1900;
        b = period % 100;
        return a * 12 + b - 1;
    }


    uint64_t convert_month_to_period(uint64_t month) {
        uint64_t year;
        if (month == 0L)
            return 0L;
        if ((year = month / 12) < 100) {
            year += (year < YY_PART_YEAR) ? 2000 : 1900;
        }
        return year * 100 + month % 12 + 1;
    }

    /*
      Convert a system time structure to TIME
    */

    void localtime_to_TIME(MYSQL_TIME *to, struct tm *from) {
        to->neg = 0;
        to->second_part = 0;
        to->year = (int) ((from->tm_year + 1900) % 10000);
        to->month = (int) from->tm_mon + 1;
        to->day = (int) from->tm_mday;
        to->hour = (int) from->tm_hour;
        to->minute = (int) from->tm_min;
        to->second = (int) from->tm_sec;
    }

    void calc_time_from_sec(MYSQL_TIME *to, int32_t seconds, int32_t microseconds) {
        int32_t t_seconds;
        // to->neg is not cleared, it may already be set to a useful value
        to->time_type = MYSQL_TIMESTAMP_TIME;
        to->year = 0;
        to->month = 0;
        to->day = 0;
        to->hour = seconds / 3600L;
        t_seconds = seconds % 3600L;
        to->minute = t_seconds / 60L;
        to->second = t_seconds % 60L;
        to->second_part = microseconds;
    }

    /* Daynumber from year 0 to 9999-12-31 */
#define MAX_DAY_NUMBER 3652424L

    bool date_add_interval(MYSQL_TIME *ltime, interval_type int_type, INTERVAL interval) {
        int32_t period, sign;

        ltime->neg = 0;

        sign = (interval.neg ? -1 : 1);

        switch (int_type) {
            case INTERVAL_SECOND:
            case INTERVAL_SECOND_MICROSECOND:
            case INTERVAL_MICROSECOND:
            case INTERVAL_MINUTE:
            case INTERVAL_HOUR:
            case INTERVAL_MINUTE_MICROSECOND:
            case INTERVAL_MINUTE_SECOND:
            case INTERVAL_HOUR_MICROSECOND:
            case INTERVAL_HOUR_SECOND:
            case INTERVAL_HOUR_MINUTE:
            case INTERVAL_DAY_MICROSECOND:
            case INTERVAL_DAY_SECOND:
            case INTERVAL_DAY_MINUTE:
            case INTERVAL_DAY_HOUR: {
                int64_t sec, days, daynr, microseconds, extra_sec;
                ltime->time_type = MYSQL_TIMESTAMP_DATETIME; // Return full date
                microseconds = ltime->second_part + sign * interval.second_part;
                extra_sec = microseconds / 1000000L;
                microseconds = microseconds % 1000000L;

                sec = ((ltime->day - 1) * 3600 * 24L + ltime->hour * 3600 + ltime->minute * 60 +
                       ltime->second +
                       sign * (int64_t) (interval.day * 3600 * 24L +
                                          interval.hour * LL(3600) + interval.minute * LL(60) +
                                          interval.second)) + extra_sec;
                if (microseconds < 0) {
                    microseconds += LL(1000000);
                    sec--;
                }
                days = sec / (3600 * LL(24));
                sec -= days * 3600 * LL(24);
                if (sec < 0) {
                    days--;
                    sec += 3600 * LL(24);
                }
                ltime->second_part = (uint32_t) microseconds;
                ltime->second = (uint32_t) (sec % 60);
                ltime->minute = (uint32_t) (sec / 60 % 60);
                ltime->hour = (uint32_t) (sec / 3600);
                daynr = calc_daynr(ltime->year, ltime->month, 1) + days;
                /* Day number from year 0 to 9999-12-31 */
                if ((uint64_t) daynr > MAX_DAY_NUMBER)
                    goto invalid_date;
                uint16_t year;
                uint8_t month, day;
                get_date_from_daynr((int32_t) daynr, &year, &month, &day);
                ltime->year = year;
                ltime->month = month;
                ltime->day = day;
                break;
            }
            case INTERVAL_DAY:
            case INTERVAL_WEEK: {
                period = (calc_daynr(ltime->year, ltime->month, ltime->day) +
                          sign * (int32_t) interval.day);
                /* Daynumber from year 0 to 9999-12-31 */
                if ((uint64_t) period > MAX_DAY_NUMBER)
                    goto invalid_date;
                uint16_t year;
                uint8_t month, day;
                get_date_from_daynr((int32_t) period, &year, &month, &day);
                ltime->year = year;
                ltime->month = month;
                ltime->day = day;
                break;
            }
            case INTERVAL_YEAR:
                ltime->year += sign * (int32_t) interval.year;
                if ((uint64_t) ltime->year >= 10000L)
                    goto invalid_date;
                if (ltime->month == 2 && ltime->day == 29 &&
                    calc_days_in_year(ltime->year) != 366)
                    ltime->day = 28;                // Was leap-year
                break;
            case INTERVAL_YEAR_MONTH:
            case INTERVAL_QUARTER:
            case INTERVAL_MONTH:
                period = (ltime->year * 12 + sign * (int32_t) interval.year * 12 +
                          ltime->month - 1 + sign * (int32_t) interval.month);
                if ((uint64_t) period >= 120000L)
                    goto invalid_date;
                ltime->year = (uint32_t) (period / 12);
                ltime->month = (uint32_t) (period % 12L) + 1;
                /* Adjust day if the new month doesn't have enough days */
                if (ltime->day > days_in_month[ltime->month - 1]) {
                    ltime->day = days_in_month[ltime->month - 1];
                    if (ltime->month == 2 && calc_days_in_year(ltime->year) == 366)
                        ltime->day++;                // Leap-year
                }
                break;
            default:
                goto null_date;
        }

        return 0;                    // Ok

        invalid_date:
        //set warning
        //    push_warning_printf(current_thd, MYSQL_ERROR::WARN_LEVEL_WARN,
        //                        ER_DATETIME_FUNCTION_OVERFLOW,
        //                        ER(ER_DATETIME_FUNCTION_OVERFLOW),
        //                        "datetime");
        null_date:
        return 1;
    }


    /*
      Calculate difference between two datetime values as seconds + microseconds.

      SYNOPSIS
        calc_time_diff()
          l_time1         - TIME/DATE/DATETIME value
          l_time2         - TIME/DATE/DATETIME value
          l_sign          - 1 absolute values are substracted,
                            -1 absolute values are added.
          seconds_out     - Out parameter where difference between
                            l_time1 and l_time2 in seconds is stored.
          microseconds_out- Out parameter where microsecond part of difference
                            between l_time1 and l_time2 is stored.

      NOTE
        This function calculates difference between l_time1 and l_time2 absolute
        values. So one should set l_sign and correct result if he want to take
        signs into account (i.e. for MYSQL_TIME values).

      RETURN VALUES
        Returns sign of difference.
        1 means negative result
        0 means positive result

    */

    bool
    calc_time_diff(MYSQL_TIME *l_time1, MYSQL_TIME *l_time2, int l_sign, int64_t *seconds_out,
                   int32_t *microseconds_out) {
        int32_t days;
        bool neg;
        int64_t microseconds;

        /*
          We suppose that if first argument is MYSQL_TIMESTAMP_TIME
          the second argument should be TIMESTAMP_TIME also.
          We should check it before calc_time_diff call.
        */
        if (l_time1->time_type == MYSQL_TIMESTAMP_TIME)  // Time value
            days = (int32_t) l_time1->day - l_sign * (int32_t) l_time2->day;
        else {
            days = calc_daynr((uint32_t) l_time1->year,
                              (uint32_t) l_time1->month,
                              (uint32_t) l_time1->day);
            if (l_time2->time_type == MYSQL_TIMESTAMP_TIME)
                days -= l_sign * (int32_t) l_time2->day;
            else
                days -= l_sign * calc_daynr((uint32_t) l_time2->year,
                                            (uint32_t) l_time2->month,
                                            (uint32_t) l_time2->day);
        }

        microseconds = ((int64_t) days * LL(86400) +
                        (int64_t) (l_time1->hour * 3600L +
                                    l_time1->minute * 60L +
                                    l_time1->second) -
                        l_sign * (int64_t) (l_time2->hour * 3600L +
                                             l_time2->minute * 60L +
                                             l_time2->second)) * LL(1000000) +
                       (int64_t) l_time1->second_part -
                       l_sign * (int64_t) l_time2->second_part;

        neg = 0;
        if (microseconds < 0) {
            microseconds = -microseconds;
            neg = 1;
        }
        *seconds_out = microseconds / 1000000L;
        *microseconds_out = (int32_t) (microseconds % 1000000L);
        return neg;
    }


    /*
      Compares 2 MYSQL_TIME structures

      SYNOPSIS
        my_time_compare()

          a - first time
          b - second time

      RETURN VALUE
       -1   - a < b
        0   - a == b
        1   - a > b

    */

    int my_time_compare(MYSQL_TIME *a, MYSQL_TIME *b) {
        uint64_t a_t = TIME_to_ulonglong_datetime(a);
        uint64_t b_t = TIME_to_ulonglong_datetime(b);

        if (a_t < b_t)
            return -1;
        if (a_t > b_t)
            return 1;

        if (a->second_part < b->second_part)
            return -1;
        if (a->second_part > b->second_part)
            return 1;

        return 0;
    }

    /*
     *
     * */
    interval_type get_interval_type(const std::string &type) {
        static std::map<std::string, interval_type> intervalTypes = {
                {"YEAR", INTERVAL_YEAR},
                {"QUARTER", INTERVAL_QUARTER},
                {"MONTH", INTERVAL_MONTH},
                {"WEEK", INTERVAL_WEEK},
                {"DAY", INTERVAL_DAY},
                {"HOUR", INTERVAL_HOUR},
                {"MINUTE", INTERVAL_MINUTE},
                {"SECOND", INTERVAL_SECOND},
                {"MICROSECOND", INTERVAL_MICROSECOND},
                {"YEAR_MONTH", INTERVAL_YEAR_MONTH},
                {"DAY_HOUR", INTERVAL_DAY_HOUR},
                {"DAY_MINUTE", INTERVAL_DAY_MINUTE},
                {"DAY_SECOND", INTERVAL_DAY_SECOND},
                {"HOUR_MINUTE", INTERVAL_HOUR_MINUTE},
                {"HOUR_SECOND", INTERVAL_HOUR_SECOND},
                {"MINUTE_SECOND", INTERVAL_MINUTE_SECOND},
                {"DAY_MICROSECOND", INTERVAL_DAY_MICROSECOND},
                {"HOUR_MICROSECOND", INTERVAL_HOUR_MICROSECOND},
                {"MINUTE_MICROSECOND", INTERVAL_MINUTE_MICROSECOND},
                {"SECOND_MICROSECOND", INTERVAL_SECOND_MICROSECOND}
        };
        std::string tmp;
        tmp.resize(type.size());
        std::transform(type.begin(), type.end(), tmp.begin(), ::toupper);
        std::map<std::string, interval_type>::iterator found;
        found = intervalTypes.find(tmp);
        if (found != intervalTypes.end()) {
            return found->second;
        }
        return INTERVAL_LAST;
    }

    /**
  @details
  Get a array of positive numbers from a string object.
  Each number is separated by 1 non digit character
  Return error if there is too many numbers.
  If there is too few numbers, assume that the numbers are left out
  from the high end. This allows one to give:
  DAY_TO_SECOND as "D MM:HH:SS", "MM:HH:SS" "HH:SS" or as seconds.

  @param str_value       string buffer
  @param is_negative     set to true if interval is prefixed by '-'
  @param count:          count of elements in result array
  @param values:         array of results
  @param transform_msec: if value is true we suppose
                         that the last part of string value is microseconds
                         and we should transform value to six digit value.
                         For example, '1.1' -> '1.100000'
  @return true           bad thing occurred
          false          no bad thing occurred
*/

    bool
    get_interval_info(const std::string &str_value, bool *is_negative, int count, uint64_t *values, bool transform_msec) {
        for (auto c :str_value) {
            if (!isascii(c)) {
                return true;
            }
        }
        const char *str = str_value.c_str();
        const char *end = str + str_value.size();

        while (str < end && isspace(*str)) ++str;
        if (str < end && *str == '-') {
            *is_negative = true;
            str++;
        }

        while (str < end && !isdigit(*str))
            str++;

        int32_t msec_length = 0;
        for (int i = 0; i < count; i++) {
            int64_t value;
            const char *start = str;
            for (value = 0; str != end && isdigit(*str); str++)
                value = value * LL(10) + (int64_t) (*str - '0');
            msec_length = 6 - (str - start);
            values[i] = value;
            while (str != end && !isdigit(*str))
                str++;
            if (str == end && i != count - 1) {
                i++;
                /* Change values[0...i-1] -> values[0...count-1] */
                size_t len= sizeof(*values) * i;
                memmove(reinterpret_cast<uint8_t *> (values + count) - len, 
                        reinterpret_cast<uint8_t *>  (values + i) - len, 
                        len);
                bzero((uint8_t *) values, sizeof(*values) * (count - i));
                break;
            }
        }

        if (transform_msec && msec_length > 0)
            values[count - 1] *= (int32_t) log_10_int[msec_length];

        return (str != end);
    }

    /**
    Convert a string to a interval value.
    To make code easy, allow interval objects without separators.

     @return  true  some bad thing occurred
              false no bad thing occurred
    */

    bool get_interval_value(const interval_type &int_type, const std::string &str_value, INTERVAL *interval) {
        uint64_t array[5];
        int64_t UNINIT_VAR(value);

        bzero((char *) interval, sizeof(*interval));
        if ((int) int_type <= INTERVAL_MICROSECOND) {
            value = std::atoi(str_value.c_str());
            if (value < 0) {
                interval->neg = true;
                value = -value;
            }
        }

        switch (int_type) {
            case INTERVAL_YEAR:
                interval->year = (uint32_t) value;
                break;
            case INTERVAL_QUARTER:
                interval->month = (uint32_t) (value * 3);
                break;
            case INTERVAL_MONTH:
                interval->month = (uint32_t) value;
                break;
            case INTERVAL_WEEK:
                interval->day = (uint32_t) (value * 7);
                break;
            case INTERVAL_DAY:
                interval->day = (uint32_t) value;
                break;
            case INTERVAL_HOUR:
                interval->hour = (uint32_t) value;
                break;
            case INTERVAL_MINUTE:
                interval->minute = value;
                break;
            case INTERVAL_SECOND:
                interval->second = value;
                break;
            case INTERVAL_MICROSECOND:
                interval->second_part = value;
                break;
            case INTERVAL_YEAR_MONTH:            // Allow YEAR-MONTH YYYYYMM
                if (get_interval_info(str_value, &interval->neg, 2, array, false))
                    return true;
                interval->year = (uint32_t) array[0];
                interval->month = (uint32_t) array[1];
                break;
            case INTERVAL_DAY_HOUR:
                if (get_interval_info(str_value, &interval->neg, 2, array, false))
                    return true;
                interval->day = (uint32_t) array[0];
                interval->hour = (uint32_t) array[1];
                break;
            case INTERVAL_DAY_MINUTE:
                if (get_interval_info(str_value, &interval->neg, 3, array, false))
                    return true;
                interval->day = (uint32_t) array[0];
                interval->hour = (uint32_t) array[1];
                interval->minute = array[2];
                break;
            case INTERVAL_DAY_SECOND:
                if (get_interval_info(str_value, &interval->neg, 4, array, false))
                    return true;
                interval->day = (uint32_t) array[0];
                interval->hour = (uint32_t) array[1];
                interval->minute = array[2];
                interval->second = array[3];
                break;
            case INTERVAL_HOUR_MINUTE:
                if (get_interval_info(str_value, &interval->neg, 2, array, false))
                    return true;
                interval->hour = (uint32_t) array[0];
                interval->minute = array[1];
                break;
            case INTERVAL_HOUR_SECOND:
                if (get_interval_info(str_value, &interval->neg, 3, array, false))
                    return true;
                interval->hour = (uint32_t) array[0];
                interval->minute = array[1];
                interval->second = array[2];
                break;
            case INTERVAL_MINUTE_SECOND:
                if (get_interval_info(str_value, &interval->neg, 2, array, false))
                    return true;
                interval->minute = array[0];
                interval->second = array[1];
                break;
            case INTERVAL_DAY_MICROSECOND:
                if (get_interval_info(str_value, &interval->neg, 5, array, true))
                    return true;
                interval->day = (uint32_t) array[0];
                interval->hour = (uint32_t) array[1];
                interval->minute = array[2];
                interval->second = array[3];
                interval->second_part = array[4];
                break;
            case INTERVAL_HOUR_MICROSECOND:
                if (get_interval_info(str_value, &interval->neg, 4, array, true))
                    return true;
                interval->hour = (uint32_t) array[0];
                interval->minute = array[1];
                interval->second = array[2];
                interval->second_part = array[3];
                break;
            case INTERVAL_MINUTE_MICROSECOND:
                if (get_interval_info(str_value, &interval->neg, 3, array, true))
                    return true;
                interval->minute = array[0];
                interval->second = array[1];
                interval->second_part = array[2];
                break;
            case INTERVAL_SECOND_MICROSECOND:
                if (get_interval_info(str_value, &interval->neg, 2, array, true))
                    return true;
                interval->second = array[0];
                interval->second_part = array[1];
                break;
            case INTERVAL_LAST: /* purecov: begin deadcode */
                // do nothing here
                break;            /* purecov: end */
        }
        return false;
    }

    INTERVAL getIntervalValue(const std::string &value, const std::string &type) {
        interval_type t = get_interval_type(type);
        INTERVAL interval;
        ad_bool bad = get_interval_value(t, value, &interval);
        if (bad) {
            throw std::runtime_error("bad interval date format");
        }
        return interval;
    }

    INTERVAL getIntervalValue(const std::string &value, const interval_type &type) {
        INTERVAL interval;
        ad_bool bad = get_interval_value(type, value, &interval);
        if (bad) {
            throw std::runtime_error("bad interval date format");
        }
        return interval;
    }

END_ARIES_ACC_NAMESPACE
