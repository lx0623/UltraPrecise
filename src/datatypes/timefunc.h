// 从 mysql 5.7 移植
// Created by david shen on 2019-07-22.
//
#pragma once

#include <ctime>
#include "AriesDefinition.h"
#include "AriesMysqlInnerTime.h"

BEGIN_ARIES_ACC_NAMESPACE

    typedef struct {
        uint32_t year,month,day,hour;
        uint64_t minute,second,second_part;
        bool neg;
    } INTERVAL;

    struct AriesInterval {
        interval_type type;
        INTERVAL interval;
    };

    //    bool get_interval_info(std::string &str_value, bool *is_negative, int count, uint32_t *values, bool transform_msec);
    bool get_interval_value(const interval_type &int_type, const std::string &str_value, INTERVAL *interval);
    interval_type get_interval_type(const std::string &type);
    INTERVAL getIntervalValue(const std::string &value, const std::string &type);
    INTERVAL getIntervalValue(const std::string &value, const interval_type &type);

    uint64_t convert_period_to_month(uint64_t period);
    uint64_t convert_month_to_period(uint64_t month);
//    void get_date_from_daynr(int32_t daynr, uint16_t *year, uint8_t *month, uint8_t *day);
/* MYSQL_TIME operations */
    bool date_add_interval(MYSQL_TIME *ltime, interval_type int_type,
                           INTERVAL interval);
    bool calc_time_diff(MYSQL_TIME *l_time1, MYSQL_TIME *l_time2, int l_sign,
                        int64_t *seconds_out, int32_t *microseconds_out);
    int my_time_compare(MYSQL_TIME *a, MYSQL_TIME *b);
    void localtime_to_TIME(MYSQL_TIME *to, struct tm *from);
    void calc_time_from_sec(MYSQL_TIME *to, int32_t seconds, int32_t microseconds);
    uint32_t calc_week(MYSQL_TIME *l_time, uint32_t week_behaviour, uint32_t *year);

END_ARIES_ACC_NAMESPACE

