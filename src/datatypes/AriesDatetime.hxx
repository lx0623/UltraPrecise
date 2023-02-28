//
// Created by david shen on 2019-07-19.
//

#pragma once
#include "AriesDefinition.h"
#include "AriesDate.hxx"
#include "AriesInnerHelper.hxx"
#include "AriesTimestamp.hxx"
#include "AriesTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    /*
     * it is for datetime, no timezone
     * example:
     * '1000-01-01 00:00:00.000000' to '9999-12-31 23:59:59.999999'
     * */
    struct ARIES_PACKED AriesDatetime {
        uint16_t year;
        uint8_t month;
        uint8_t day;
        uint8_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t second_part;

    public:
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime();
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(const AriesDatetime & datetime);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(const AriesDate & date);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(uint16_t y, uint8_t m, uint8_t d, uint8_t h, uint8_t mt, uint8_t s, uint32_t ms);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(int64_t timestamp, int offset);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(const AriesTimestamp &timestamp, int offset = 0);
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime(const Decimal &timestamp, int offset);
        ARIES_HOST_DEVICE_NO_INLINE bool isValid() const;

        ARIES_HOST_DEVICE_NO_INLINE uint16_t getYear() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getMonth() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getDay() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getHour() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getMinute() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getSecond() const;
        ARIES_HOST_DEVICE_NO_INLINE uint32_t getMicroSec() const;
        ARIES_HOST_DEVICE_NO_INLINE uint32_t getWeek(uint32_t week_behaviour) const;
        ARIES_HOST_DEVICE_NO_INLINE int getQuarter() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynr() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynrOfMonth1stDay() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynrOfYear1stDay() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getLastDay() const;
        ARIES_HOST_DEVICE_NO_INLINE int diffMonths(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffYears(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffQuarters(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffDays(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffWeeks(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffHours(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffMinutes(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffSeconds(const AriesDatetime &datetime) const;
        /*
         * return value is based on 1000ms
         * */
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffMicroSeconds(const AriesDatetime &datetime) const;

        /*
         * the result is Decimal(17, 6) based on second, fraction part is ms
         * */
        ARIES_HOST_DEVICE_NO_INLINE Decimal getUnixTimestamp(int offset) const;
        ARIES_HOST_DEVICE_NO_INLINE int getSecFromTimePart() const;

        /*
         * return value is based on us
         * */
        ARIES_HOST_DEVICE_NO_INLINE int64_t toTimestamp() const;

        ARIES_HOST_DEVICE_NO_INLINE int compare(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int comparetimepart(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int comparedatepart(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int compareminutepart(const AriesDatetime &datetime) const;
        ARIES_HOST_DEVICE_NO_INLINE int comparesecondpart(const AriesDatetime &datetime) const;

        ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesDatetime& dt) const;
        ARIES_HOST_DEVICE_NO_INLINE AriesDatetime& operator+=(const AriesTime &time);

        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesDate &date, const AriesDatetime& dt);
        friend ARIES_HOST_DEVICE_NO_INLINE AriesDatetime operator+ (const AriesDatetime& dt, const AriesTime& time);
    };


END_ARIES_ACC_NAMESPACE
