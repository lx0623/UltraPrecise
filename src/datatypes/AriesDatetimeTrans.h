//
// Created by david on 19-7-26.
//

#pragma once

#include <string>
#include "AriesDefinition.h"
#include "timefunc.h"
#include "AriesMysqlInnerTime.h"
#include "AriesDate.hxx"
#include "AriesYear.hxx"
#include "AriesDatetime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesIntervalTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

#define MAX_YEAR_NUMBER 10000
#define MAX_MONTH_NUMBER (MAX_YEAR_NUMBER * 12)
#define MAX_DAY_NUMBER 3652424L
#define MAX_HOUR_NUMBER (MAX_DAY_NUMBER * 24)
#define MAX_MINUTE_NUMBER (MAX_HOUR_NUMBER * 60)
#define MAX_SECOND_NUMBER (MAX_MINUTE_NUMBER * 60)
#define MAX_SECOND_PART_NUMBER 999999

#define ARIES_DATE_NOT_STRICT_MODE 0
#define ARIES_DATE_STRICT_MODE 1

class AriesDatetimeTrans {
public:
    static AriesDatetimeTrans GetInstance();
    AriesYear ToAriesYear(const MYSQL_TIME &l_time);
    AriesDate ToAriesDate(const MYSQL_TIME &l_time);
    AriesDatetime ToAriesDatetime(const MYSQL_TIME &l_time);
    AriesTime ToAriesTime(const MYSQL_TIME &l_time);
    AriesTimestamp ToAriesTimestamp(const MYSQL_TIME &l_time);

    AriesYear ToAriesYear(const std::string &s, int mode = ARIES_DATE_NOT_STRICT_MODE);
    AriesDate ToAriesDate(const std::string &s, int mode = ARIES_DATE_NOT_STRICT_MODE);
    AriesDatetime ToAriesDatetime(const std::string &s, int mode = ARIES_DATE_NOT_STRICT_MODE);
    AriesTime ToAriesTime(const std::string &s, int mode = ARIES_DATE_NOT_STRICT_MODE);
    AriesTimestamp ToAriesTimestamp(const std::string &s, int mode = ARIES_DATE_NOT_STRICT_MODE);

    std::string ToString(const MYSQL_TIME & time);
    std::string ToString(const AriesTime & time);

    std::string ToString(const AriesYear &year);
    std::string ToString(const AriesDate &date);
    std::string ToString(const AriesDatetime &datetime);
    std::string ToString( const AriesTimestamp &timestamp );

    //for interval time
    YearInterval ToYearInterval(const INTERVAL &interval);
    MonthInterval ToMonthInterval(const INTERVAL &interval);
    YearMonthInterval ToYearMonthInterval(const INTERVAL &interval);
    DayInterval ToDayInterval(const INTERVAL &interval);
    SecondInterval ToSecondInterval(const INTERVAL &interval);
    SecondPartInterval ToSecondPartInterval(const INTERVAL &interval);

    YearInterval ToYearInterval(std::string &s, std::string &type);
    MonthInterval ToMonthInterval(std::string &s, std::string &type);
    YearMonthInterval ToYearMonthInterval(std::string &s, std::string &type);
    DayInterval ToDayInterval(std::string &s, std::string &type);
    SecondInterval ToSecondInterval(std::string &s, std::string &type);
    SecondPartInterval ToSecondPartInterval(std::string &s, std::string &type);

    bool ToBool( const AriesDate &date );
    bool ToBool( const AriesTime &time );
    bool ToBool( const AriesYear &year );
    bool ToBool( const AriesDatetime &datetime );
    bool ToBool( const AriesTimestamp &ts );

    static AriesDatetime Now();
    static AriesDatetime DatetimeOfToday();

private:
    AriesDatetimeTrans();
    static AriesDatetimeTrans ariesDatetimeTrans;
    void checkIntervalValue(const INTERVAL &interval);
};

END_ARIES_ACC_NAMESPACE
