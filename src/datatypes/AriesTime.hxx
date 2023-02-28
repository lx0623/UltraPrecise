//
// Created by david shen on 2019-07-19.
//

#ifndef DATETIMELIB_ARIESTIME_HXX
#define DATETIMELIB_ARIESTIME_HXX

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

/* Limits for the TIME data type */
#define TIME_MAX_HOUR 838
#define TIME_MAX_MINUTE 59
#define TIME_MAX_SECOND 59

    struct ARIES_PACKED AriesTime {
        uint16_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t second_part;
        int8_t sign;

        ARIES_HOST_DEVICE_NO_INLINE AriesTime();
        ARIES_HOST_DEVICE_NO_INLINE AriesTime(int8_t sign, uint16_t h, uint8_t m, uint8_t s, uint32_t ms);
        ARIES_HOST_DEVICE_NO_INLINE AriesTime(int64_t ms);

        ARIES_HOST_DEVICE_NO_INLINE AriesTime operator-() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getHour() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getMinute() const;
        ARIES_HOST_DEVICE_NO_INLINE uint8_t getSecond() const;
        ARIES_HOST_DEVICE_NO_INLINE uint32_t getMicroSec() const;
        ARIES_HOST_DEVICE_NO_INLINE int8_t getSign() const;

        ARIES_HOST_DEVICE_NO_INLINE int64_t toMicroseconds() const;

        ARIES_HOST_DEVICE_NO_INLINE int compare(const AriesTime &date) const;

        ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesTime &time) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesTime &time) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesTime &time) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesTime &time) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesTime &time) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesTime &time) const;

    private:
        ARIES_HOST_DEVICE_NO_INLINE void checkValue();
    };

END_ARIES_ACC_NAMESPACE


#endif //DATETIMELIB_ARIESTIME_HXX
