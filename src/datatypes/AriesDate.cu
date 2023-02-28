//
// Created by david shen on 2019-07-19.
//
#include "AriesDate.hxx"
#include "AriesDatetime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE AriesDate::AriesDate() : AriesDate(0, 0, 0) {
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate::AriesDate(uint16_t y, uint8_t m, uint8_t d) : year(y), month(m), day(d) {
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate::AriesDate(const AriesTimestamp &timestamp, int offset) {
        if (timestamp.getTimeStamp() <= 0) {
            memset(this, 0x00, sizeof(AriesDate));
        }
        uint64_t uts = timestamp.getTimeStamp();
        uts /= 1000000; // to seconds
        uint32_t days = (uts + offset) / 86400;
        days += DAYS_AT_TIMESTAMP_START;
        uint16_t y;
        uint8_t m, d;
        get_date_from_daynr(days, &y, &m, &d);
        year = y;
        month = m;
        day = d;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate::AriesDate(const AriesDatetime &datetime) {
        year = datetime.year;
        month = datetime.month;
        day = datetime.day;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::isValid() const {
        return year && month && day;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t AriesDate::getYear() const {
        return year;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDate::getMonth() const {
        return month;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDate::getDay() const {
        return day;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t AriesDate::getWeek(uint32_t week_behaviour) const {
        uint32_t out;
        return calc_week((uint32_t) year, month, day, week_behaviour, &out);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::getQuarter() const {
        return ((int) month + 2) / 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::getDaynr() const {
        return calc_daynr((uint32_t) year, (uint32_t) month, (uint32_t) day);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::getDaynrOfMonth1stDay() const {
        return calc_daynr((uint32_t) year, (uint32_t) month, 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::getDaynrOfYear1stDay() const {
        return calc_daynr((uint32_t) year, 1, 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t AriesDate::getLastDay() const {
        return get_last_day(year, month);
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::compare(const AriesDate &date) const {
        int tmp = (int) year - (int) date.year;
        if (tmp == 0) {
            tmp = (int) month - (int) date.month;
            if (tmp == 0) {
                tmp = (int) day - (int) date.day;
            }
        }
        return tmp;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffMonths(const AriesDate &date) const {
        int months = 0;
        if (*this < date) {
            months = -diff_months(date.year, date.month, date.day, year, month, day, 0);
        } else {
            months = diff_months(year, month, day, date.year, date.month, date.day, 0);
        }
        return months;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffYears(const AriesDate &date) const {
        return diffMonths(date) / 12;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffQuarters(const AriesDate &date) const {
        return diffMonths(date) / 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffDays(const AriesDate &date) const {
        return getDaynr() - date.getDaynr();
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffWeeks(const AriesDate &date) const {
        return diffDays(date) / 7;
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesDate::diffHours(const AriesDate &date) const {
        return diffDays(date) * 24;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDate::diffMinutes(const AriesDate &date) const {
        return (int64_t) diffDays(date) * 1440;
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDate::diffSeconds(const AriesDate &date) const {
        return (int64_t) diffDays(date) * 86400;
    }

    /*
     * return value is based on 1000ms
     * */
    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDate::diffMicroSeconds(const AriesDate &date) const {
        return (int64_t) diffDays(date) * 86400000;
    }

    /* return value based on us*/
    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesDate::toTimestamp() const {
        int days = getDaynr();
        if (days < DAYS_AT_TIMESTAMP_START || days > DAYS_AT_TIMESTAMP_END) {
            return 0;
        }
        days -= DAYS_AT_TIMESTAMP_START;
        return (int64_t) days * 86400000000;
    }

    /*
     * the result is decimal(11, 0) based on second
     * */
    ARIES_HOST_DEVICE_NO_INLINE Decimal AriesDate::getUnixTimestamp(int offset) const {
        Decimal r = Decimal(11,0);
        int days = getDaynr();
        if (days < DAYS_AT_TIMESTAMP_START || days > DAYS_AT_TIMESTAMP_END) {
            return r;
        }
        days -= DAYS_AT_TIMESTAMP_START;
        return r.cast(Decimal((int64_t) days * 86400 + offset));
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator>(const AriesDate &date) const {
        return compare(date) > 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator>=(const AriesDate &date) const {
        return compare(date) >= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator<(const AriesDate &date) const {
        return compare(date) < 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator<=(const AriesDate &date) const {
        return compare(date) <= 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator==(const AriesDate &date) const {
        return compare(date) == 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesDate::operator!=(const AriesDate &date) const {
        return compare(date) != 0;
    }

END_ARIES_ACC_NAMESPACE