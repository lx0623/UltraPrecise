//
// Created by david.shen on 2019-08-13.
//

#ifndef DATETIMELIB_ARIESTIMECALC_HXX
#define DATETIMELIB_ARIESTIMECALC_HXX

#include "AriesDefinition.h"
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesTimestamp.hxx"
#include "AriesYear.hxx"
#include "AriesInnerHelper.hxx"
#include "decimal.hxx"
#include "AriesDateFormat.hxx"
#include "aries_types.hxx"
#include "aries_char.hxx"


BEGIN_ARIES_ACC_NAMESPACE

/* Daynumber from year 0 to 9999-12-31 */
#define MAX_DAY_NUMBER 3652424L


    class AriesTimeCalc {
    public:
        //for AriesDate add
        ARIES_HOST_DEVICE_NO_INLINE AriesDate add(const AriesDate &ariesDate, const YearInterval &yearInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate add(const AriesDate &ariesDate, const MonthInterval &monthInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate add(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate add(const AriesDate &ariesDate, const DayInterval &dayInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDate &ariesDate, const SecondInterval &secondInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval);

        //for AriesDatetime add
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDatetime, const DayInterval &dayInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDate, const SecondInterval &secondInterval);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime add(const AriesDatetime &ariesDate, const SecondPartInterval &secondPartInterval);

        //for datediff / timediff
        /*
         * Calculate difference between two date values as days
         * */
        ARIES_HOST_DEVICE_NO_INLINE int datediff(const AriesDate &date1, const AriesDate &date2);
        ARIES_HOST_DEVICE_NO_INLINE int datediff(const AriesDatetime &datetime1, const AriesDatetime &datetime2);
        ARIES_HOST_DEVICE_NO_INLINE int datediff(const AriesDate &date, const AriesDatetime &datetime);
        ARIES_HOST_DEVICE_NO_INLINE int datediff(const AriesDatetime &datetime, const AriesDate &date);
        /*
         * Calculate difference between two times or dates type values as AriesTime
         * */
        ARIES_HOST_DEVICE_NO_INLINE AriesTime timediff(const AriesDatetime &datetime1, const AriesDatetime &datetime2);
        ARIES_HOST_DEVICE_NO_INLINE AriesTime timediff(const AriesDate &date1, const AriesDate &date2);
        ARIES_HOST_DEVICE_NO_INLINE AriesTime timediff(const AriesTime &time1, const AriesTime &time2);

        //for subtime
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime subtime(const AriesDatetime &ariesDatetime, const AriesTime &time);
        ARIES_HOST_DEVICE_NO_INLINE AriesTime subtime(const AriesTime &time1, const AriesTime &time2);

        //for extract
        ARIES_HOST_DEVICE_NO_INLINE int32_t extract(const interval_type &type, const AriesDate &date, uint32_t week_behaviour);
        ARIES_HOST_DEVICE_NO_INLINE int32_t extract(const interval_type &type, const AriesDatetime &datetime, uint32_t week_behaviour);
        ARIES_HOST_DEVICE_NO_INLINE int32_t extract(const interval_type &type, const AriesTime &time);

        //for timestampdiff
        ARIES_HOST_DEVICE_NO_INLINE int64_t timestampdiff(const AriesDate &date1, const AriesDate &date2, const interval_type &type);
        ARIES_HOST_DEVICE_NO_INLINE int64_t timestampdiff(const AriesDate &date, const AriesDatetime &datetime, const interval_type &type);
        ARIES_HOST_DEVICE_NO_INLINE int64_t timestampdiff(const AriesDatetime &datetime, const AriesDate &date, const interval_type &type);
        ARIES_HOST_DEVICE_NO_INLINE int64_t timestampdiff(const AriesDatetime &datetime1, const AriesDatetime &datetime2, const interval_type &type);

    private:
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime genAriesDatetime(int daynr, int64_t sec, int32_t ms);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate genAriesDateByMonth(const AriesDate &date, int month);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime genAriesDatetimeByMonth(const AriesDatetime &date, int month);
    };

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesDate &ariesDate);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesDatetime &ariesDatetime);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesTimestamp &timestamp);

    // for DATE_ADD / DATE_SUB
    //for AriesDate
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDate &ariesDate, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDate &ariesDate, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval);

    //for AriesDatetime
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval);

    //for SUBDATE
    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDate &ariesDate, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const DayInterval & dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval);

    //for DATEDIFF / TIMEDIFF
    /*
     * Calculate difference between two date values as days
     * */
    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDate &date1, const AriesDate &date2);
    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2);
    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDatetime &datetime, const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDate &date, const AriesDatetime &datetime);
    /*
     * Calculate difference between two date values as AriesTime
     * */
    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2);
    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesDate &date1, const AriesDate &date2);
    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesTime &time1, const AriesTime &time2);

    //for SUBTIME
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBTIME(const AriesDatetime &ariesDatetime, const AriesTime &time);
    ARIES_HOST_DEVICE_NO_INLINE AriesTime SUBTIME(const AriesTime &time1, const AriesTime &time2);

    //for EXTRACT
    /*
     * if use interval type NOT supported, will return -1 which means NULL
     * */
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesDate &date, uint32_t week_behaviour = WEEK_FIRST_WEEKDAY);
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesDatetime &datetime, uint32_t week_behaviour = WEEK_FIRST_WEEKDAY);
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesTime &time);
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesDate &date, const interval_type &type, uint32_t week_behaviour = WEEK_FIRST_WEEKDAY);
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesDatetime &datetime, const interval_type &type, uint32_t week_behaviour = WEEK_FIRST_WEEKDAY);
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesTime &time, const interval_type &type, uint32_t week_behaviour = WEEK_FIRST_WEEKDAY);

    //for TIMESTAMP
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDate & date);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDatetime & datetime);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDate & date, const AriesTime &time);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDatetime & datetime, const AriesTime &time);
    /*
     * for TIMESTAMPADD or TIMESTAMPDIFF, the interval type must be one of the following type:
     *   MICROSECOND (microseconds), SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.
     * */
    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate & date, const YearInterval &yearInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate & date, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate & date, const DayInterval &dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDate & date, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDate & date, const SecondPartInterval &secondPartInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime & datetime, const MonthInterval &monthInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime & datetime, const DayInterval &dayInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime & datetime, const SecondInterval &secondInterval);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime & datetime, const SecondPartInterval &secondPartInterval);

    /*
     * for microsecond type, the diff is based on 1000ms
     * */
    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDate &date1, const AriesDate &date2, const interval_type &type);
    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDate &date, const AriesDatetime &datetime, const interval_type &type);
    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDatetime &datetime, const AriesDate &date, const interval_type &type);
    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2, const interval_type &type);

    ARIES_HOST_DEVICE_NO_INLINE uint16_t YEAR(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint16_t YEAR(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t QUARTER(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t QUARTER(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesDatetime &datetime);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesTimestamp &timestamp);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAY(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAY(const AriesDatetime &datetime);

    /*
     * for mode:
    Mode First day of week	Range	Week 1 is the first week …
    0	Sunday	0-53	with a Sunday in this year
    1	Monday	0-53	with 4 or more days this year
    2	Sunday	1-53	with a Sunday in this year
    3	Monday	1-53	with 4 or more days this year
    4	Sunday	0-53	with 4 or more days this year
    5	Monday	0-53	with a Monday in this year
    6	Sunday	1-53	with 4 or more days this year
    7	Monday	1-53	with a Monday in this year

     definition: WEEK_MONDAY_FIRST WEEK_YEAR WEEK_FIRST_WEEKDAY
     IMPORT: if user omitted mode, caller should use value of the default_week_format system variable.
     * */
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEK(const AriesDate &date, uint8_t mode = WEEK_FIRST_WEEKDAY);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEK(const AriesDatetime &datetime, uint8_t mode = WEEK_FIRST_WEEKDAY);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t HOUR(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t HOUR(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MINUTE(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t MINUTE(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t SECOND(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t SECOND(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint32_t MICROSECOND(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t MICROSECOND(const AriesDatetime &datetime);

//    ARIES_HOST_DEVICE_NO_INLINE char * DAYNAME(const AriesDate &date);
//    ARIES_HOST_DEVICE_NO_INLINE char * DAYNAME(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFMONTH(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFMONTH(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFWEEK(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFWEEK(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint16_t DAYOFYEAR(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint16_t DAYOFYEAR(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE AriesDate LAST_DAY(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE AriesDate LAST_DAY(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKDAY(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKDAY(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKOFYEAR(const AriesDate &date);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKOFYEAR(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE AriesDate FROM_DAYS(uint32_t days);

    ARIES_HOST_DEVICE_NO_INLINE uint32_t YEARWEEK(const AriesDate &date, uint8_t mode = 0);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t YEARWEEK(const AriesDatetime &datetime, uint8_t mode = 0);

    //for unix timestamp
    ARIES_HOST_DEVICE_NO_INLINE Decimal UNIX_TIMESTAMP(const AriesDate &date, int offset);
    ARIES_HOST_DEVICE_NO_INLINE Decimal UNIX_TIMESTAMP(const AriesDatetime &datetime, int offset);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime FROM_UNIXTIME(const AriesTimestamp &timestamp, int offset);
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime FROM_UNIXTIME(const Decimal &timestamp, int offset);

    //for abs function
    ARIES_HOST_DEVICE_NO_INLINE AriesTime abs(AriesTime time);

    //for DATA_FORMAT
    /*
     * calculate target buffer length by format string
     * */
    ARIES_HOST_DEVICE_NO_INLINE uint32_t get_format_length(const char *format, const LOCALE_LANGUAGE &language = en_US);
    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesDate &date, const LOCALE_LANGUAGE &language = en_US);
    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesDatetime &datetime, const LOCALE_LANGUAGE &language = en_US);
    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesTimestamp &timestamp, const LOCALE_LANGUAGE &language = en_US);

    /*
     * only handle format such as "1001-01-01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesDate &date, const LOCALE_LANGUAGE &language = en_US);
    /*
     * only handle format such as "1001-01-01 01:01:01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesDatetime &datetime, const LOCALE_LANGUAGE &language = en_US);
    /*
     * only handle format such as "1001-01-01 01:01:01.000"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesTimestamp &timestamp, const LOCALE_LANGUAGE &language = en_US);

    template< typename type_t, int LEN, bool has_null = false >
    struct op_dateformat_t
    {
        ARIES_HOST_DEVICE aries_char< LEN > operator()( const type_t &date, const aries_acc::AriesDatetime &current, const char* format, const aries_acc::LOCALE_LANGUAGE &locale ) const
        {
            aries_char< LEN > result;
            char* pOut = result;
            if (!DATE_FORMAT( pOut, format, date, locale ))
            {
                //error happened
                pOut[0] = 'E';
                pOut[1] = '\0';
            }
            return result;
        }
    };

    template< typename type_t, int LEN >
    struct op_dateformat_t< type_t, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( const nullable_type<type_t> &date, const aries_acc::AriesDatetime &current, const char* format, const aries_acc::LOCALE_LANGUAGE &locale ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* pOut = result;
            if (date.flag)
            {
                pOut[0] = 1;
                if (!DATE_FORMAT( pOut + 1, format, date.value, locale ))
                {
                    //error happened
                    pOut[1] = 'E';
                    pOut[2] = '\0';
                }
            }
            else
            {
                pOut[0] = 0;
            }

            return result;
        }
    };

    template< int LEN >
    struct op_dateformat_t<aries_acc::AriesTime, LEN, false >
    {
        ARIES_HOST_DEVICE aries_char< LEN > operator()( const aries_acc::AriesTime &date, const aries_acc::AriesDatetime &current, const char* format, const aries_acc::LOCALE_LANGUAGE &locale ) const
        {
            aries_char< LEN > result;
            char* pOut = result;
            if (!DATE_FORMAT( pOut, format, current + date, locale ))
            {
                //error happened
                pOut[0] = 'E';
                pOut[1] = '\0';
            }
            return result;
        }
    };

    template< int LEN >
    struct op_dateformat_t< aries_acc::AriesTime, LEN, true >
    {
        ARIES_HOST_DEVICE nullable_type< aries_char< LEN > > operator()( const nullable_type<aries_acc::AriesTime> &date, const aries_acc::AriesDatetime &current, const char* format, const aries_acc::LOCALE_LANGUAGE &locale ) const
        {
            nullable_type< aries_char< LEN > > result;
            char* pOut = result;
            if (date.flag)
            {
                pOut[0] = 1;
                if (!DATE_FORMAT( pOut + 1, format, current + date.value, locale ))
                {
                    //error happened
                    pOut[1] = 'E';
                    pOut[2] = '\0';
                }
            }
            else
            {
                pOut[0] = 0;
            }

            return result;
        }
    };
END_ARIES_ACC_NAMESPACE

#endif //DATETIMELIB_ARIESTIMECALC_HXX
