//
// Created by david shen on 2019-07-19.
//

#include "AriesDatetime.hxx"


BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime() {
        memset(this, 0x00, sizeof(AriesDatetime));
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(const AriesDatetime & datetime)
        : year( datetime.year ),
          month( datetime.month ),
          day( datetime.day ),
          hour( datetime.hour ),
          minute( datetime.minute ),
          second( datetime.second ),
          second_part( datetime.second_part )
    {
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(const AriesDate &date) : AriesDatetime(date.year, date.month, date.day, 0, 0, 0, 0) {
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(uint16_t y, uint8_t m, uint8_t d, uint8_t h, uint8_t mt, uint8_t s, uint32_t ms) {
        year = y;
        month = m;
        day = d;
        hour = h;
        minute = mt;
        second = s;
        second_part = ms;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(int64_t timestamp, int offset) {
        if (timestamp <= 0) {
            memset(this, 0x00, sizeof(AriesDatetime));
            return;
        }
        second_part = timestamp % 1000000;
        timestamp /= 1000000; // to seconds
        uint64_t ts = (uint64_t) timestamp + offset;
        uint32_t days = ts / 86400;
        days += DAYS_AT_TIMESTAMP_START;
        uint16_t y;
        uint8_t m, d;
        get_date_from_daynr(days, &y, &m, &d);
        year = y;
        month = m;
        day = d;
        ts = ts % 86400;
        minute = ts / 60 % 60;
        hour = ts / 3600;
        second = ts % 60;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(const AriesTimestamp &timestamp, int offset) {
        if (timestamp.getTimeStamp() <= 0) {
            memset(this, 0x00, sizeof(AriesDatetime));
            return;
        }
        uint64_t uts = timestamp.getTimeStamp();
        second_part = uts % 1000000;
        uts /= 1000000; // to seconds
        uint64_t ts = (uint64_t) uts + offset;
        uint32_t days = ts / 86400;
        days += DAYS_AT_TIMESTAMP_START;
        uint16_t y;
        uint8_t m, d;
        get_date_from_daynr(days, &y, &m, &d);
        year = y;
        month = m;
        day = d;
        ts = ts % 86400;
        minute = ts / 60 % 60;
        hour = ts / 3600;
        second = ts % 60;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime::AriesDatetime(const Decimal &timestamp, int offset) {
        if (timestamp <= 0) {
            memset(this, 0x00, sizeof(AriesDatetime));
            return;
        }
        uint64_t ts = (uint64_t) timestamp.getIntPart(0) - (uint64_t)offset;
        uint32_t days = ts / 86400;
        days += DAYS_AT_TIMESTAMP_START;
        uint16_t y;
        uint8_t m, d;
        get_date_from_daynr(days, &y, &m, &d);
        year = y;
        month = m;
        day = d;

        ts = ts % 86400;
        minute = ts / 60 % 60;
        hour = ts / 3600;
        second = ts % 60;
        second_part = timestamp.getFracPart(0) / 1000;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::isValid() const {
        return year && month && day;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t AriesDatetime::getYear() const {
        return year;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getMonth() const {
        return month;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getDay() const {
        return day;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getHour() const {
        return hour;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getMinute() const {
        return minute;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getSecond() const {
        return second;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t AriesDatetime::getMicroSec() const {
        return second_part;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t AriesDatetime::getWeek(uint32_t week_behaviour) const {
        uint32_t out;
        return calc_week((uint32_t) year, month, day, week_behaviour, &out);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::getQuarter() const {
        return ((int) month + 2) / 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::getDaynr() const {
        return calc_daynr((uint32_t) year, (uint32_t) month, (uint32_t) day);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::getDaynrOfMonth1stDay() const {
        return calc_daynr((uint32_t) year, (uint32_t) month, 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::getDaynrOfYear1stDay() const {
        return calc_daynr((uint32_t) year, 1, 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDatetime::getLastDay() const {
        return get_last_day(year, month);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffMonths(const AriesDatetime &datetime) const {
        int months = 0;
        if (*this < datetime) {
            months = -diff_months(datetime.year, datetime.month, datetime.day, year, month, day, datetime.comparetimepart(*this));
        } else {
            months = diff_months(year, month, day, datetime.year, datetime.month, datetime.day, comparetimepart(datetime));
        }
        return months;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffYears(const AriesDatetime &datetime) const {
        return diffMonths(datetime) / 12;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffQuarters(const AriesDatetime &datetime) const {
        return diffMonths(datetime) / 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffDays(const AriesDatetime &datetime) const {
        int days = getDaynr() - datetime.getDaynr();
        if (days < 0) {
            if (comparetimepart(datetime) > 0) {
                days += 1;
            }
        } else if (days > 0) {
            if (comparetimepart(datetime) < 0) {
                days -= 1;
            }
        }
        return days;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffWeeks(const AriesDatetime &datetime) const {
        return diffDays(datetime) / 7;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::diffHours(const AriesDatetime &datetime) const {
        int hours = (getDaynr() - datetime.getDaynr()) * 24 + hour - datetime.hour;
        if (hours < 0) {
            if (compareminutepart(datetime) > 0) {
                hours += 1;
            }
        } else if (hours > 0) {
            if (compareminutepart(datetime) < 0) {
                hours -= 1;
            }
        }
        return hours;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDatetime::diffMinutes(const AriesDatetime &datetime) const {
        int64_t minutes = (int64_t) (getDaynr() - datetime.getDaynr()) * 1440 + (hour - datetime.hour) * 60 + minute - datetime.minute;
        if (minutes < 0) {
            if (comparesecondpart(datetime) > 0) {
                minutes += 1;
            }
        } else if (minutes > 0) {
            if (comparesecondpart(datetime) < 0) {
                minutes -= 1;
            }
        }
        return minutes;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDatetime::diffSeconds(const AriesDatetime &datetime) const {
        int64_t sec = (int64_t) (getDaynr() - datetime.getDaynr()) * 86400 +
                      (hour - datetime.hour) * 3600 +
                      (minute - datetime.minute) * 60 +
                      second - datetime.second;
        if (sec < 0) {
            if (second_part > datetime.second_part) {
                sec += 1;
            }
        } else if (sec > 0) {
            if (second_part < datetime.second_part) {
                sec -= 1;
            }
        }
        return sec;
    }

    /*
     * return value is based on microsecond
     * */
    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDatetime::diffMicroSeconds(const AriesDatetime &datetime) const {
        return (((int64_t) (getDaynr() - datetime.getDaynr()) * 86400 +
                 (hour - datetime.hour) * 3600 +
                 (minute - datetime.minute) * 60 +
                 second - datetime.second) * 1000000 +
                second_part - datetime.second_part);
    }

    /*
     * the result is Decimal(17, 6) based on second, fraction part is microsecond
     * */
    ARIES_HOST_DEVICE_NO_INLINE Decimal AriesDatetime::getUnixTimestamp(int offset) const {
        int days = getDaynr();
        int secPart = getSecFromTimePart() + offset;
        days += secPart / 86400;
        secPart %= 86400;
        if (secPart < 0) {
            days -= 1;
            secPart += 86400;
        }
        if (days < DAYS_AT_TIMESTAMP_START || days > DAYS_AT_TIMESTAMP_END ||
            (days == DAYS_AT_TIMESTAMP_END && secPart > SECONDS_AT_TIMESTAMP_END_DAY)) {
            Decimal r(0);
            return r;
        }
        days -= DAYS_AT_TIMESTAMP_START;
        Decimal r(17, 6);
        r.setIntPart(days * 86400 + secPart, 0);
        r.setFracPart(second_part * 1000, 0);
        return r;
    }

    /*
     * return value is based on us
     * */
    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDatetime::toTimestamp() const {
        int64_t days = getDaynr() - DAYS_AT_TIMESTAMP_START;
        return (days * 86400 + getSecFromTimePart()) * 1000000 + second_part;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::getSecFromTimePart() const {
        return (int) hour * 3600 + (int) minute * 60 + second;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::compare(const AriesDatetime &datetime) const {
        int tmp = comparedatepart(datetime);
        if (tmp == 0) {
            tmp = comparetimepart(datetime);
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::comparedatepart(const AriesDatetime &datetime) const {
        int tmp = (int) year - (int) datetime.year;
        if (tmp == 0) {
            tmp = (int) month - (int) datetime.month;
            if (tmp == 0) {
                tmp = (int) day - (int) datetime.day;
            }
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::comparetimepart(const AriesDatetime &datetime) const {
        int tmp = (int) hour - (int) datetime.hour;
        if (tmp == 0) {
            tmp = (int) minute - (int) datetime.minute;
            if (tmp == 0) {
                tmp = (int) second - (int) datetime.second;
                if (tmp == 0) {
                    tmp = (int) second_part - (int) datetime.second_part;
                }
            }
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::compareminutepart(const AriesDatetime &datetime) const {
        int tmp = (int) minute - (int) datetime.minute;
        if (tmp == 0) {
            tmp = (int) second - (int) datetime.second;
            if (tmp == 0) {
                tmp = (int) second_part - (int) datetime.second_part;
            }
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDatetime::comparesecondpart(const AriesDatetime &datetime) const {
        int tmp = (int) second - (int) datetime.second;
        if (tmp == 0) {
            tmp = (int) second_part - (int) datetime.second_part;
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator>(const AriesDatetime &dt) const {
        return compare(dt) > 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator>=(const AriesDatetime &dt) const {
        return compare(dt) >= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator<(const AriesDatetime &dt) const {
        return compare(dt) < 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator<=(const AriesDatetime &dt) const {
        return compare(dt) <= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator==(const AriesDatetime &dt) const {
        return compare(dt) == 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDatetime::operator!=(const AriesDatetime &dt) const {
        return compare(dt) != 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime& AriesDatetime::operator+= (const AriesTime& dt) {
        int64_t ts = (getDaynr() - DAYS_AT_TIMESTAMP_START) * 86400;
        ts += getSecFromTimePart();
        ts = ts * 1000000 + ( int64_t )second_part; // to microsecond
        ts += dt.toMicroseconds();
        if (ts < 0) {
            year = 0;
            month = 0;
            day = 0;
            hour = 0;
            minute = 0;
            second = 0;
            second_part = 0;
        } else {
            AriesDatetime datetime(ts);
            year = datetime.year;
            month = datetime.month;
            day = datetime.day;
            hour = datetime.hour;
            minute = datetime.minute;
            second = datetime.second;
            second_part = datetime.second_part;
        }
        return *this;
    }
    ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime > dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime >= dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime < dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime <= dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime == dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesDate &date, const AriesDatetime &dt) {
        AriesDatetime datetime(date);
        return datetime != dt;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime operator+ (const AriesDatetime& dt, const AriesTime& time) {
        AriesDatetime tmp(dt);
        tmp += time;
        return tmp;
    }

END_ARIES_ACC_NAMESPACE