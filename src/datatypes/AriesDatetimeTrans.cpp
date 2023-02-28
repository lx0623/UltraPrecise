//
// Created by david on 19-7-26.
//

#include <cstring>
#include <stdexcept>

#include "AriesDatetimeTrans.h"


BEGIN_ARIES_ACC_NAMESPACE

    AriesDatetimeTrans::AriesDatetimeTrans() {
    }

    AriesDatetimeTrans AriesDatetimeTrans::GetInstance() {
        return ariesDatetimeTrans;
    }

    AriesYear AriesDatetimeTrans::ToAriesYear(const MYSQL_TIME &l_time) {
        if (l_time.time_type == MYSQL_TIMESTAMP_TIME) {
            throw std::runtime_error("the type is MYSQL_TIMESTAMP_TIME");
        }
        AriesYear t(l_time.year);
        return t;
    }

    AriesDate AriesDatetimeTrans::ToAriesDate(const MYSQL_TIME &l_time) {
        if (l_time.time_type == MYSQL_TIMESTAMP_TIME) {
            throw std::runtime_error("the type is MYSQL_TIMESTAMP_TIME");
        }
        AriesDate t(l_time.year, l_time.month, l_time.day);
        return t;
    }

    AriesDatetime AriesDatetimeTrans::ToAriesDatetime(const MYSQL_TIME &l_time) {
        if (l_time.time_type != MYSQL_TIMESTAMP_DATETIME && l_time.time_type != MYSQL_TIMESTAMP_DATE) {
            throw std::runtime_error("the type is not MYSQL_TIMESTAMP_DATETIME or MYSQL_TIMESTAMP_DATE");
        }
        AriesDatetime t(l_time.year, l_time.month, l_time.day, l_time.hour, l_time.minute, l_time.second,
                        l_time.second_part);
        return t;
    }

    AriesTime AriesDatetimeTrans::ToAriesTime(const MYSQL_TIME &l_time) {
        if (l_time.time_type != MYSQL_TIMESTAMP_TIME) {
            throw std::runtime_error("the type is not MYSQL_TIMESTAMP_TIME");
        }
        int32_t hour = l_time.day * 24 + l_time.hour;
        if (hour > 838) {
            AriesTime t(l_time.neg ? -1 : 1, 838, 59, 59, l_time.second_part);
            return t;
        } else {
            AriesTime t(l_time.neg ? -1 : 1, hour, l_time.minute, l_time.second, l_time.second_part);
            return t;
        }
    }

    AriesTimestamp AriesDatetimeTrans::ToAriesTimestamp(const MYSQL_TIME &l_time) {
        if (l_time.time_type != MYSQL_TIMESTAMP_DATE && l_time.time_type != MYSQL_TIMESTAMP_DATETIME ) {
            throw std::runtime_error("the type is not MYSQL_TIMESTAMP_DATE or MYSQL_TIMESTAMP_DATETIME");
        }
        struct tm timeStruct;
        memset(&timeStruct, 0, sizeof(timeStruct));
        timeStruct.tm_year = l_time.year - 1900;
        timeStruct.tm_mon = l_time.month - 1;
        timeStruct.tm_mday = l_time.day;
        timeStruct.tm_hour = l_time.hour;
        timeStruct.tm_min = l_time.minute;
        timeStruct.tm_sec = l_time.second;
        uint64_t ts = timegm(&timeStruct);
        return AriesTimestamp(ts * 1000000 + l_time.second_part);
    }

    AriesYear AriesDatetimeTrans::ToAriesYear(const std::string &s, int mode) {
        const char *str = s.c_str();
        char temp[8] = {0};
        //remove space
        while (isspace(*str)) str++;
        if (strlen(str) > 8) {
            if (mode == ARIES_DATE_NOT_STRICT_MODE) {
                return AriesYear(0);
            }
            throw std::runtime_error("bad year format");
        }
        int i = 0;
        while (isdigit(*str)) {
            temp[i++] = *str++;
        }
        while (isspace(*str)) str++;
        if (*str != 0 || i == 0 || i > 4) {
            if (mode == ARIES_DATE_NOT_STRICT_MODE) {
                return AriesYear(0);
            }
            throw std::runtime_error("bad year format");
        }
        int y = 0;
        for (int j = 0; j < i; ++j) {
            y = y * 10 + temp[j] - '0';
        }
        AriesYear t(y);
        return t;
    }

    AriesDate AriesDatetimeTrans::ToAriesDate(const std::string &s, int mode) {
        int was_cut;
        MYSQL_TIME d;
        int flag = TIME_FUZZY_DATE;
        if (mode == ARIES_DATE_STRICT_MODE) {
            flag |= TIME_NO_ZERO_IN_DATE | TIME_NO_ZERO_DATE;
        }
        enum_mysql_timestamp_type type = str_to_datetime(s.c_str(), s.size(), &d, flag, &was_cut);
        if (type < 0) {
            if (mode == ARIES_DATE_NOT_STRICT_MODE) {
                return AriesDate();
            }
            throw std::runtime_error("bad date format");
        }
        return ToAriesDate(d);
    }

    AriesDatetime AriesDatetimeTrans::ToAriesDatetime(const std::string &s, int mode) {
        int was_cut;
        MYSQL_TIME d;
        int flag = TIME_FUZZY_DATE;
        if (mode == ARIES_DATE_STRICT_MODE) {
            flag |= TIME_NO_ZERO_IN_DATE | TIME_NO_ZERO_DATE | TIME_DATETIME_ONLY;
        }
        enum_mysql_timestamp_type type = str_to_datetime(s.c_str(), s.size(), &d, flag, &was_cut);
        if (type != MYSQL_TIMESTAMP_DATETIME && type != MYSQL_TIMESTAMP_DATE) {
            if (mode == ARIES_DATE_NOT_STRICT_MODE) {
                return AriesDatetime();
            }
            throw std::runtime_error("bad datetime format");
        }
        return ToAriesDatetime(d);
    }

    AriesTime AriesDatetimeTrans::ToAriesTime(const std::string &s, int mode) {
        MYSQL_TIME d;
        int warning = 0;
        ad_bool bad = str_to_time(s.c_str(), s.size(), &d, &warning);
        if (bad || (mode == ARIES_DATE_STRICT_MODE && warning > 0)) {
            throw std::runtime_error("bad time format");
        }
        return ToAriesTime(d);
    }

    AriesTimestamp AriesDatetimeTrans::ToAriesTimestamp(const std::string &s, int mode) {
        int warning;
        MYSQL_TIME d;
        int flag = TIME_FUZZY_DATE;
        if (mode == ARIES_DATE_STRICT_MODE) {
            flag |= TIME_NO_ZERO_IN_DATE | TIME_NO_ZERO_DATE | TIME_DATETIME_ONLY;
        }
        enum_mysql_timestamp_type type = str_to_datetime( s.c_str(), s.size(), &d, flag, &warning );
        if ( type != MYSQL_TIMESTAMP_DATETIME && type != MYSQL_TIMESTAMP_DATE )
        {
            throw std::runtime_error( "bad datetime or date format" );
        }
        return ToAriesTimestamp( d );
    }

    std::string AriesDatetimeTrans::ToString( const AriesYear &year ) {
        char tmp[8];
        sprintf(tmp, "%04u", year.getYear());
        return std::string(tmp);
    }

    std::string AriesDatetimeTrans::ToString( const AriesDate &date ) {
        char tmp[16];
        sprintf(tmp, "%04u-%02u-%02u", date.getYear(), date.getMonth(), date.getDay());
        return std::string(tmp);
    }

    std::string AriesDatetimeTrans::ToString( const MYSQL_TIME & time ) {
        char tmp[32];
        sprintf(tmp, "%04u-%02u-%02u %02u:%02u:%02u", time.year, time.month, time.day,
                time.hour, time.minute, time.second);
        return std::string(tmp);
    }

    std::string AriesDatetimeTrans::ToString( const AriesTime & time ) {
        char tmp[32] = {0};
        if ( time.second_part )
        {
            sprintf(tmp, "%s%u:%2u:%2u.%u", time.sign > 0 ? "" : "-", time.hour, time.minute, time.second, time.second_part);
        } else {
            sprintf(tmp, "%s%u:%2u:%2u", time.sign > 0 ? "" : "-", time.hour, time.minute, time.second);
        }
        
        return std::string(tmp);
    }

    std::string AriesDatetimeTrans::ToString( const AriesDatetime &datetime ) {
        char tmp[32];
        if ( datetime.getMicroSec() )
        {
            sprintf(tmp, "%04u-%02u-%02u %02u:%02u:%02u.%u", datetime.getYear(), datetime.getMonth(), datetime.getDay(),
                    datetime.getHour(), datetime.getMinute(), datetime.getSecond(), datetime.getMicroSec() );
        } else {
            sprintf(tmp, "%04u-%02u-%02u %02u:%02u:%02u", datetime.getYear(), datetime.getMonth(), datetime.getDay(),
                    datetime.getHour(), datetime.getMinute(), datetime.getSecond() );
        }
        return std::string(tmp);
    }

    std::string AriesDatetimeTrans::ToString( const AriesTimestamp &timestamp ) {
        char tmp[32];
        AriesDatetime datetime( timestamp.getTimeStamp(), 0 );
        if ( datetime.getMicroSec() )
        {
            sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu.%06u", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                    datetime.getMinute(), datetime.getSecond(), datetime.getMicroSec() );
        }
        else
        {
            sprintf( tmp, "%04hu-%02hhu-%02hhu %02hhu:%02hhu:%02hhu", datetime.getYear(), datetime.getMonth(), datetime.getDay(), datetime.getHour(),
                    datetime.getMinute(), datetime.getSecond() );
        }
        return std::string(tmp);
    }

    bool AriesDatetimeTrans::ToBool( const AriesDate &date )
    {
        return !( 0 == date.year && 0 == date.day && 0 == date.month );
    }
    bool AriesDatetimeTrans::ToBool( const AriesTime &time )
    {
        return time.toMicroseconds() != 0;
    }
    bool AriesDatetimeTrans::ToBool( const AriesYear &year )
    {
        return year.year != 0;
    }
    bool AriesDatetimeTrans::ToBool( const AriesDatetime &datetime )
    {
        return !( 0 == datetime.year && 0 == datetime.day && 0 == datetime.month &&
                  0 == datetime.hour && 0 == datetime.minute && 0 == datetime.second &&
                  0 == datetime.second_part );
    }
    bool AriesDatetimeTrans::ToBool( const AriesTimestamp &ts )
    {
        return ts.timestamp != 0;
    }

    //for interval time
    void AriesDatetimeTrans::checkIntervalValue(const INTERVAL &interval) {
        if (interval.year > MAX_YEAR_NUMBER) {
            throw std::runtime_error("bad year number");
        }
        uint64_t tmp = interval.year * 12 + interval.month;
        if (tmp > MAX_MONTH_NUMBER) {
            throw std::runtime_error("bad month number");
        }
        if (interval.day > MAX_DAY_NUMBER) {
            throw std::runtime_error("bad day number");
        }
        tmp = interval.day * 24 + interval.hour;
        if (tmp > MAX_HOUR_NUMBER) {
            throw std::runtime_error("bad hour number");
        }
        tmp = tmp * 60 + interval.minute;
        if (tmp > MAX_MINUTE_NUMBER) {
            throw std::runtime_error("bad minute number");
        }
        tmp = tmp * 60 + interval.second;
        if (tmp > MAX_SECOND_NUMBER) {
            throw std::runtime_error("bad second number");
        }
        if (interval.second_part > MAX_SECOND_PART_NUMBER) {
            throw std::runtime_error("bad micro second number");
        }
    }

    YearInterval AriesDatetimeTrans::ToYearInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        YearInterval y(interval.year, interval.neg ? -1 : 1);
        return y;
    }

    MonthInterval AriesDatetimeTrans::ToMonthInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        MonthInterval monthInterval(interval.month, interval.neg ? -1 : 1);
        return monthInterval;
    }

    YearMonthInterval AriesDatetimeTrans::ToYearMonthInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        YearMonthInterval yearMonthInterval(interval.year, interval.month, interval.neg ? -1 : 1);
        return yearMonthInterval;
    }

    DayInterval AriesDatetimeTrans::ToDayInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        DayInterval dayInterval(interval.day, interval.neg ? -1 : 1);
        return dayInterval;
    }

    SecondInterval AriesDatetimeTrans::ToSecondInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        uint64_t sec = (uint64_t) interval.day * 3600 * 24 + interval.hour * 3600 + (uint64_t) interval.minute * 60 + interval.second;
        SecondInterval secondInterval(sec, interval.neg ? -1 : 1);
        return secondInterval;
    }

    SecondPartInterval AriesDatetimeTrans::ToSecondPartInterval(const INTERVAL &interval) {
        checkIntervalValue(interval);
        uint64_t sec = (uint64_t) interval.day * 3600 * 24 + interval.hour * 3600 + (uint64_t) interval.minute * 60 + interval.second;
        uint32_t tmp = interval.second_part > MAX_SECOND_PART_NUMBER ? MAX_SECOND_PART_NUMBER : interval.second_part;
        SecondPartInterval secondPartInterval(sec, tmp, interval.neg ? -1 : 1);
        return secondPartInterval;
    }

    YearInterval AriesDatetimeTrans::ToYearInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (t != INTERVAL_YEAR) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToYearInterval(getIntervalValue(s, t));
    }

    MonthInterval AriesDatetimeTrans::ToMonthInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (t != INTERVAL_MONTH && t != INTERVAL_QUARTER) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToMonthInterval(getIntervalValue(s, t));
    }

    YearMonthInterval AriesDatetimeTrans::ToYearMonthInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (t != INTERVAL_YEAR_MONTH) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToYearMonthInterval(getIntervalValue(s, t));
    }

    DayInterval AriesDatetimeTrans::ToDayInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (t != INTERVAL_WEEK && t != INTERVAL_DAY) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToDayInterval(getIntervalValue(s, t));
    }

    SecondInterval AriesDatetimeTrans::ToSecondInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (!(t == INTERVAL_HOUR || t == INTERVAL_MINUTE || t == INTERVAL_SECOND || t == INTERVAL_DAY_HOUR || t == INTERVAL_DAY_MINUTE ||
            t == INTERVAL_DAY_SECOND || t == INTERVAL_HOUR_MINUTE || t == INTERVAL_HOUR_SECOND || t == INTERVAL_MINUTE_SECOND)) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToSecondInterval(getIntervalValue(s, t));
    }
    SecondPartInterval AriesDatetimeTrans::ToSecondPartInterval(std::string &s, std::string &type) {
        interval_type t = get_interval_type(type);
        if (!(t == INTERVAL_MICROSECOND || t == INTERVAL_DAY_MICROSECOND || t == INTERVAL_HOUR_MICROSECOND || t == INTERVAL_MINUTE_MICROSECOND ||
              t == INTERVAL_SECOND_MICROSECOND)) {
            throw std::runtime_error("transfer to bad interval type");
        }
        return ToSecondPartInterval(getIntervalValue(s, t));
    }

    AriesDatetime AriesDatetimeTrans::Now() {
        auto time_now = time((time_t*)nullptr);
        struct tm tm_now;
        localtime_r(&time_now, &tm_now);

        auto year = tm_now.tm_year + 1900;
        auto month = tm_now.tm_mon + 1;
        auto day = tm_now.tm_mday;
        auto hour = tm_now.tm_hour;
        auto minute = tm_now.tm_min;
        auto second = tm_now.tm_sec;

        return AriesDatetime(static_cast<uint16_t>(year), static_cast<uint8_t>(month), static_cast<uint8_t>(day), static_cast<uint8_t>(hour), static_cast<uint8_t>(minute), static_cast<uint8_t>(second), 0);
    }

    AriesDatetime AriesDatetimeTrans::DatetimeOfToday() {
        auto datetime = Now();
        datetime.hour = 0;
        datetime.minute = 0;
        datetime.second = 0;

        return datetime;
    }

END_ARIES_ACC_NAMESPACE