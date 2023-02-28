//
// Created by david on 19-7-25.
//

#pragma once

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

/*
  Available interval types used in any statement.

  'interval_type' must be sorted so that simple intervals comes first,
  ie year, quarter, month, week, day, hour, etc. The order based on
  interval size is also important and the intervals should be kept in a
  large to smaller order. (get_interval_value() depends on this)

  Note: If you change the order of elements in this enum you should fix
  order of elements in 'interval_type_to_name' and 'interval_names'
  arrays

  See also interval_type_to_name, get_interval_value, interval_names
*/

    enum interval_type : int32_t {
        INTERVAL_YEAR, INTERVAL_QUARTER, INTERVAL_MONTH, INTERVAL_WEEK, INTERVAL_DAY,
        INTERVAL_HOUR, INTERVAL_MINUTE, INTERVAL_SECOND, INTERVAL_MICROSECOND,
        INTERVAL_YEAR_MONTH, INTERVAL_DAY_HOUR, INTERVAL_DAY_MINUTE,
        INTERVAL_DAY_SECOND, INTERVAL_HOUR_MINUTE, INTERVAL_HOUR_SECOND,
        INTERVAL_MINUTE_SECOND, INTERVAL_DAY_MICROSECOND, INTERVAL_HOUR_MICROSECOND,
        INTERVAL_MINUTE_MICROSECOND, INTERVAL_SECOND_MICROSECOND, INTERVAL_LAST
    };

    /*
     * support interval_type:
     *      INTERVAL_YEAR
     */
    struct ARIES_PACKED YearInterval {
        uint16_t year;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE YearInterval(uint16_t y, int8_t sign);
        ARIES_HOST_DEVICE_NO_INLINE YearInterval();
    };

    /*
     * support interval_type:
     *      INTERVAL_MONTH
     *      INTERVAL_QUARTER
     */
    struct ARIES_PACKED MonthInterval {
        uint32_t month;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE MonthInterval();
        ARIES_HOST_DEVICE_NO_INLINE MonthInterval(uint32_t m, int8_t sign);
    };

    /*
     * support interval_type:
     *      INTERVAL_YEAR_MONTH
     */
    struct ARIES_PACKED YearMonthInterval {
        uint16_t year;
        uint32_t month;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE YearMonthInterval();
        ARIES_HOST_DEVICE_NO_INLINE YearMonthInterval(uint16_t y, uint32_t m, int8_t sign);
    };

    /*
     * support interval_type:
     *      INTERVAL_WEEK
     *      INTERVAL_DAY
     */
    struct ARIES_PACKED DayInterval {
        uint32_t day;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE DayInterval();
        ARIES_HOST_DEVICE_NO_INLINE DayInterval(uint32_t d, int8_t sign);
    };

    /*
     * support interval_type:
     *      INTERVAL_HOUR
     *      INTERVAL_MINUTE
     *      INTERVAL_SECOND
     *      INTERVAL_DAY_HOUR
     *      INTERVAL_DAY_MINUTE
     *      INTERVAL_DAY_SECOND
     *      INTERVAL_HOUR_MINUTE
     *      INTERVAL_HOUR_SECOND
     *      INTERVAL_MINUTE_SECOND
     */
    struct ARIES_PACKED SecondInterval {
        uint64_t second;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE SecondInterval();
        ARIES_HOST_DEVICE_NO_INLINE SecondInterval(uint64_t s, int8_t sign);
    };

    /*
     * support interval_type:
     *      INTERVAL_DAY_MICROSECOND
     *      INTERVAL_HOUR_MICROSECOND
     *      INTERVAL_MINUTE_MICROSECOND
     *      INTERVAL_SECOND_MICROSECOND
     */
    struct ARIES_PACKED SecondPartInterval {
        uint64_t second;
        uint32_t second_part;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE SecondPartInterval();
        ARIES_HOST_DEVICE_NO_INLINE SecondPartInterval(uint64_t s, uint32_t sp, int8_t sign);
    };

END_ARIES_ACC_NAMESPACE
