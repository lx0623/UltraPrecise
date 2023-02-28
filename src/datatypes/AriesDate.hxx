//
// Created by david shen on 2019-07-19.
//

#ifndef DATETIMELIB_ARIESDATE_HXX
#define DATETIMELIB_ARIESDATE_HXX

#include "AriesDefinition.h"
#include "AriesInnerHelper.hxx"
#include "decimal.hxx"
#include "AriesTimestamp.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    struct ARIES_PACKED AriesDatetime;

    struct ARIES_PACKED AriesDate {
        uint16_t year;
        uint8_t month;
        uint8_t day;

    public:
        ARIES_HOST_DEVICE_NO_INLINE AriesDate();
        ARIES_HOST_DEVICE_NO_INLINE AriesDate(uint16_t y, uint8_t m, uint8_t d);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate(const AriesTimestamp &timestamp, int offset = 0);
        ARIES_HOST_DEVICE_NO_INLINE AriesDate(const AriesDatetime &datetime);

        ARIES_HOST_DEVICE_NO_INLINE bool isValid() const;

        ARIES_HOST_DEVICE_NO_INLINE uint16_t getYear() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getMonth() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getDay() const;
        ARIES_HOST_DEVICE_NO_INLINE uint32_t getWeek(uint32_t week_behaviour) const;
        ARIES_HOST_DEVICE_NO_INLINE int getQuarter() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynr() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynrOfMonth1stDay() const;
        ARIES_HOST_DEVICE_NO_INLINE int getDaynrOfYear1stDay() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getLastDay() const;
        ARIES_HOST_DEVICE_NO_INLINE int diffMonths(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffYears(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffQuarters(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffDays(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffWeeks(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int diffHours(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffMinutes(const AriesDate &date) const;
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffSeconds(const AriesDate &date) const;
        /*
         * return value is based on 1000ms
         * */
        ARIES_HOST_DEVICE_NO_INLINE int64_t diffMicroSeconds(const AriesDate &date) const;
        /*
         * return value is based on us
         * */
        ARIES_HOST_DEVICE_NO_INLINE int64_t toTimestamp() const;

        /*
         * the result is decimal(11, 0) based on second
         * */
        ARIES_HOST_DEVICE_NO_INLINE Decimal getUnixTimestamp(int offset) const;

        ARIES_HOST_DEVICE_NO_INLINE int compare(const AriesDate& date) const;

        ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesDate& date) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesDate& date) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesDate& date) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesDate& date) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesDate& date) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesDate& date) const;
    };

END_ARIES_ACC_NAMESPACE

#endif //DATETIMELIB_ARIESDATE_HXX
