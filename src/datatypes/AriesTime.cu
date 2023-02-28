//
// Created by david shen on 2019-07-19.
//

#include <cstring>
#include "AriesTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE AriesTime::AriesTime() {
        memset(this, 0x00, sizeof(AriesTime));
        sign = 1;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime::AriesTime(int8_t sign, uint16_t h, uint8_t m, uint8_t s, uint32_t ms) : sign(sign), hour(h), minute(m),
                                                                                    second(s), second_part(ms) {
        checkValue();
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime::AriesTime(int64_t ms) {
        if (ms < 0) {
            sign = -1;
            ms = -ms;
        } else {
            sign = 1;
        }
        second_part = (uint32_t) (ms % 1000000L);
        //microseconds is used as seconds
        ms = ms / 1000000L;
        hour = (uint16_t) (ms / 3600L);
        ms = ms % 3600L;
        minute = (uint8_t) (ms / 60L);
        second = (uint8_t) (ms % 60L);
        checkValue();
    }

    ARIES_HOST_DEVICE_NO_INLINE void AriesTime::checkValue() {
        if (hour > TIME_MAX_HOUR) {
            hour = TIME_MAX_HOUR;
            minute = TIME_MAX_MINUTE;
            second = TIME_MAX_SECOND;
            second_part = 0;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime AriesTime::operator-() const {
        AriesTime time(*this);
        time.sign = -time.sign;
        return time;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesTime::getHour() const {
        return hour;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesTime::getMinute() const {
        return minute;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesTime::getSecond() const {
        return second;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t AriesTime::getMicroSec() const {
        return second_part;
    }

    ARIES_HOST_DEVICE_NO_INLINE int8_t AriesTime::getSign() const {
        return sign;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesTime::toMicroseconds() const {
        int64_t sec = hour * 3600 + minute * 60 + second;
        sec = sec * 1000000 + second_part;
        if (sign) {
            sec = -sec;
        }
        return sec;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesTime::compare(const AriesTime &time) const {
        int tmp = (int) sign - (int) time.sign;
        if (!tmp) {
            tmp = (int) hour - (int) time.hour;
            if (!tmp) {
                tmp = (int) minute - (int) time.minute;
                if (!tmp) {
                    tmp = (int) second - (int) time.second;
                    if (!tmp) {
                        tmp = (int) second_part - (int) time.second_part;
                    }
                }
            }
            if (sign == -1) {
                tmp = -tmp;
            }
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator>(const AriesTime &time) const {
        return compare(time) > 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator>=(const AriesTime &time) const {
        return compare(time) >= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator<(const AriesTime &time) const {
        return compare(time) < 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator<=(const AriesTime &time) const {
        return compare(time) <= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator==(const AriesTime &time) const {
        return (hour == time.hour
                && minute == time.minute
                && second == time.second
                && second_part == time.second_part);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTime::operator!=(const AriesTime &time) const {
        return (second_part != time.second_part
                || second != time.second
                || minute != time.minute
                || hour != time.hour);
    }

END_ARIES_ACC_NAMESPACE