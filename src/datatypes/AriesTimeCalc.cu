//
// Created by david.shen on 2019-08-13.
//

#include "AriesTimeCalc.hxx"
#include "AriesInnerHelper.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::genAriesDatetime(int daynr, int64_t sec, int32_t ms) {
        sec += ms / 1000000L;
        ms = ms % 1000000L;
        if (ms < 0) {
            ms += 1000000LL;
            sec--;
        }

        daynr += sec / (3600 * 24LL);
        sec %= 3600 * 24LL;
        if (sec < 0) {
            daynr--;
            sec += 3600 * 24LL;
        }
        /* Day number from year 0 to 9999-12-31 */
        if ((uint64_t) daynr > MAX_DAY_NUMBER) {  //invalid date return 0000-00-00
            AriesDatetime tmp;
            return tmp;
        } else {
            AriesDatetime d;
            d.second_part = (uint32_t) ms;
            d.second = (uint32_t) (sec % 60);
            d.minute = (uint32_t) (sec / 60 % 60);
            d.hour = (uint32_t) (sec / 3600);
            uint16_t year;
            uint8_t month, day;
            get_date_from_daynr(daynr, &year, &month, &day);
            d.year = year;
            d.month = month;
            d.day = day;
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate AriesTimeCalc::genAriesDateByMonth(const AriesDate &date, int month) {
        if ((uint32_t) month >= 120000L) { //invalid date, return 0000-00-00
            AriesDate tmp;
            return tmp;
        } else {
            AriesDate d = date;
            d.year = (uint32_t) (month / 12);
            d.month = (uint32_t) (month % 12L) + 1;
            /* Adjust day if the new month doesn't have enough days */
            if (d.day > getDaysInMonth(d.month - 1)) {
                d.day = getDaysInMonth(d.month - 1);
                if (d.month == 2 && calc_days_in_year(d.year) == 366)
                    d.day++;                // Leap-year
            }
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::genAriesDatetimeByMonth(const AriesDatetime &date, int month) {
        if ((uint32_t) month >= 120000L) { //invalid date, return 0000-00-00
            AriesDatetime tmp;
            return tmp;
        } else {
            AriesDatetime d = date;
            d.year = (uint32_t) (month / 12);
            d.month = (uint32_t) (month % 12L) + 1;
            /* Adjust day if the new month doesn't have enough days */
            if (d.day > getDaysInMonth(d.month - 1)) {
                d.day = getDaysInMonth(d.month - 1);
                if (d.month == 2 && calc_days_in_year(d.year) == 366)
                    d.day++;                // Leap-year
            }
            return d;
        }
    }

    //for AriesDate
    ARIES_HOST_DEVICE_NO_INLINE AriesDate AriesTimeCalc::add(const AriesDate &ariesDate, const YearInterval &yearInterval) {
        int32_t year = (int32_t) ariesDate.year + (int) yearInterval.sign * (int) yearInterval.year;
        if ((uint32_t) year >= 10000L) { //valid date, return 0000-00-00
            AriesDate tmp;
            return tmp;
        } else {
            AriesDate d = ariesDate;
            d.year = (uint16_t) year;
            if (d.month == 2 && d.day == 29 && calc_days_in_year(d.year) != 366)
                d.day = 28;                // Was leap-year
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate AriesTimeCalc::add(const AriesDate &ariesDate, const MonthInterval &monthInterval) {
        return genAriesDateByMonth(ariesDate, ariesDate.year * 12 + ariesDate.month - 1 + (int) monthInterval.sign * (int) monthInterval.month);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate AriesTimeCalc::add(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval) {
        return genAriesDateByMonth(ariesDate, ariesDate.year * 12 + (int) yearMonthInterval.sign * (int) yearMonthInterval.year * 12 +
                                              ariesDate.month - 1 + (int) yearMonthInterval.sign * (int) yearMonthInterval.month);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate AriesTimeCalc::add(const AriesDate &ariesDate, const DayInterval &dayInterval) {
        int period = ariesDate.getDaynr() + (int) dayInterval.sign * (int) dayInterval.day;
        /* Daynumber from year 0 to 9999-12-31 */
        if ((uint32_t) period > MAX_DAY_NUMBER) { //valid date return 0000-000-00
            AriesDate tmp;
            return tmp;
        } else {
            AriesDate d;
            uint16_t year;
            uint8_t month, day;
            get_date_from_daynr(period, &year, &month, &day);
            d.year = year;
            d.month = month;
            d.day = day;
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDate &ariesDate, const SecondInterval &secondInterval) {
        return genAriesDatetime(ariesDate.getDaynrOfMonth1stDay(),
                                (ariesDate.day - 1) * 3600 * 24L + (int) secondInterval.sign * (int64_t) (secondInterval.second),
                                0);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval) {
        return genAriesDatetime(ariesDate.getDaynrOfMonth1stDay(),
                                (ariesDate.day - 1) * 3600 * 24L + (int) secondPartInterval.sign * (int64_t) (secondPartInterval.second),
                                (int) secondPartInterval.sign * secondPartInterval.second_part);
    }

    //for AriesDate diff
    /*
     * Calculate difference between two date values as days
     * */
    ARIES_HOST_DEVICE_NO_INLINE int AriesTimeCalc::datediff(const AriesDate &date1, const AriesDate &date2) {
        return date1.getDaynr() - date2.getDaynr();
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesTimeCalc::datediff(const AriesDatetime &datetime1, const AriesDatetime &datetime2) {
        return datetime1.getDaynr() - datetime2.getDaynr();
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesTimeCalc::datediff(const AriesDate &date, const AriesDatetime &datetime) {
        return date.getDaynr() - datetime.getDaynr();
    }

    ARIES_HOST_DEVICE_NO_INLINE int AriesTimeCalc::datediff(const AriesDatetime &datetime, const AriesDate &date) {
        return datetime.getDaynr() - date.getDaynr();
    }

    /*
     * Calculate difference between two date values as AriesTime
     * */
    ARIES_HOST_DEVICE_NO_INLINE AriesTime AriesTimeCalc::timediff(const AriesDatetime &datetime1, const AriesDatetime &datetime2) {
        int32_t days = datediff(datetime1, datetime2);
        int64_t ms = ((int64_t) days * 86400LL +
                      (int64_t) (datetime1.hour * 3600L + datetime1.minute * 60L + datetime1.second) -
                      (int64_t) (datetime2.hour * 3600L + datetime2.minute * 60L + datetime2.second)) * 1000000LL +
                     (int64_t) datetime1.second_part - (int64_t) datetime2.second_part;
        AriesTime time(ms);
        return time;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime AriesTimeCalc::timediff(const AriesDate &date1, const AriesDate &date2) {
        int32_t days = datediff(date1, date2);
        int sign = 1;
        if (days < 0) {
            sign = -1;
            days = -days;
        }
        AriesTime time(sign, days * 24, 0, 0, 0);
        return time;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime AriesTimeCalc::timediff(const AriesTime &time1, const AriesTime &time2) {
        int64_t ms;
        if (time1.sign == time2.sign) {
            ms = (int64_t) (((int) time1.hour - (int) time2.hour) * 3600L + ((int) time1.minute - (int) time2.minute) * 60L + (int) time1.second -
                            (int) time2.second) * 1000000LL + time1.second_part - time2.second_part;
        } else {
            ms = (int64_t) (((int) time1.hour + (int) time2.hour) * 3600L + ((int) time1.minute + (int) time2.minute) * 60L + (int) time1.second +
                            (int) time2.second) * 1000000LL + time1.second_part + time2.second_part;
        }
        if (time1.sign == -1) {
            ms = -ms;
        }
        AriesTime time(ms);
        return time;
    }

    //for subtime
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::subtime(const AriesDatetime &ariesDatetime, const AriesTime &time) {
        SecondPartInterval interval((int32_t) time.hour * 3600L + (int32_t) time.minute * 60L + time.second, time.second_part, -time.sign);
        return add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime AriesTimeCalc::subtime(const AriesTime &time1, const AriesTime &time2) {
        return timediff(time1, time2);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t AriesTimeCalc::extract(const interval_type &type, const AriesDate &date, uint32_t week_behaviour) {
        switch (type) {
            case INTERVAL_YEAR:
                return date.year;
            case INTERVAL_QUARTER:
                return date.getQuarter();
            case INTERVAL_MONTH:
                return date.month;
            case INTERVAL_WEEK:
                return date.getWeek(week_behaviour);
            case INTERVAL_DAY:
                return date.day;
            case INTERVAL_YEAR_MONTH:
                return date.year * 100 + date.month;
            case INTERVAL_LAST:
                return -1;
            default:
                return 0;
        }
    }

    // micro second will be scaled down to 123 ms from 123999 ms
    ARIES_HOST_DEVICE_NO_INLINE int32_t AriesTimeCalc::extract(const interval_type &type, const AriesDatetime &datetime, uint32_t week_behaviour) {
        switch (type) {
            case INTERVAL_YEAR:
                return datetime.year;
            case INTERVAL_QUARTER:
                return datetime.getQuarter();
            case INTERVAL_MONTH:
                return datetime.month;
            case INTERVAL_WEEK:
                return datetime.getWeek(week_behaviour);
            case INTERVAL_DAY:
                return datetime.day;
            case INTERVAL_HOUR:
                return datetime.hour;
            case INTERVAL_MINUTE:
                return datetime.minute;
            case INTERVAL_SECOND:
                return datetime.second;
            case INTERVAL_MICROSECOND:
                return datetime.second_part / 1000;
            case INTERVAL_YEAR_MONTH:
                return (int32_t) datetime.year * 100 + datetime.month;
            case INTERVAL_DAY_HOUR:
                return (int32_t) datetime.day * 100 + datetime.hour;
            case INTERVAL_DAY_MINUTE:
                return (int32_t) datetime.day * 10000 + (int32_t) datetime.hour * 100 + datetime.minute;
            case INTERVAL_DAY_SECOND:
                return (int32_t) datetime.day * 1000000 + (int32_t) datetime.hour * 10000 + (int32_t) datetime.minute * 100 + datetime.second;
            case INTERVAL_HOUR_MINUTE:
                return (int32_t) datetime.hour * 100 + datetime.minute;
            case INTERVAL_HOUR_SECOND:
                return (int32_t) datetime.hour * 10000 + (int32_t) datetime.minute * 100 + datetime.second;
            case INTERVAL_MINUTE_SECOND:
                return (int32_t) datetime.minute * 100 + datetime.second;
            case INTERVAL_DAY_MICROSECOND:
                //NOT support, because int32_t can't contain day_microsecond
//                return ((int64_t) datetime.day * 1000000 + (int64_t) datetime.hour * 10000 + (int64_t) datetime.minute * 100 + datetime.second) * 1000000 + datetime.second_part;
                return -1;
            case INTERVAL_HOUR_MICROSECOND:
                // micro second will be scaled down to 123 ms from 123999 ms
                return ((int32_t) datetime.hour * 10000 + (int32_t) datetime.minute * 100 + datetime.second) * 1000 + datetime.second_part / 1000;
            case INTERVAL_MINUTE_MICROSECOND:
                return ((int32_t) datetime.minute * 100 + datetime.second) * 1000 + datetime.second_part / 1000;
            case INTERVAL_SECOND_MICROSECOND:
                return (int32_t) datetime.second * 1000 + datetime.second_part / 1000;
            case INTERVAL_LAST:
                return -1;
            default:
                return 0;
        }
    }

    // micro second will be scaled down to 123 ms from 123999 ms
    ARIES_HOST_DEVICE_NO_INLINE int32_t AriesTimeCalc::extract(const interval_type &type, const AriesTime &time) {
        int32_t result = 0;
        switch (type) {
            case INTERVAL_DAY_HOUR:
            case INTERVAL_HOUR:
                result = time.hour;
                break;
            case INTERVAL_MINUTE:
                result = time.minute;
                break;
            case INTERVAL_SECOND:
                result = time.second;
                break;
            case INTERVAL_MICROSECOND:
                result = time.second_part / 1000;
                break;
            case INTERVAL_DAY_MINUTE:
            case INTERVAL_HOUR_MINUTE:
                result = (int32_t) time.hour * 100 + time.minute;
                break;
            case INTERVAL_DAY_SECOND:
            case INTERVAL_HOUR_SECOND:
                result = (int32_t) time.hour * 10000 + (int32_t) time.minute * 100 + time.second;
                break;
            case INTERVAL_MINUTE_SECOND:
                result = (int32_t) time.minute * 100 + time.second;
                break;
            case INTERVAL_DAY_MICROSECOND:
            case INTERVAL_HOUR_MICROSECOND:
                result = ((int32_t) time.hour * 10000 + (int32_t) time.minute * 100 + time.second) * 1000 + time.second_part / 1000;
                break;
            case INTERVAL_MINUTE_MICROSECOND:
                result = ((int32_t) time.minute * 100 + time.second) * 1000 + time.second_part / 1000;
                break;
            case INTERVAL_SECOND_MICROSECOND:
                result = (int32_t) time.second * 1000 + time.second_part / 1000;
                break;
            case INTERVAL_LAST:
            default:
                return -1;
        }
        return time.sign == -1 ? -result : result;
    }

    //for AriesDatetime
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval) {
        int32_t year = (int32_t) ariesDatetime.year + (int) yearInterval.sign * (int) yearInterval.year;
        if (year >= 10000L) { //valid date, return 0000-00-00
            AriesDatetime tmp;
            return tmp;
        } else {
            AriesDatetime d = ariesDatetime;
            d.year = (uint16_t) year;
            if (d.month == 2 && d.day == 29 && calc_days_in_year(d.year) != 366)
                d.day = 28;                // Was leap-year
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval) {
        return genAriesDatetimeByMonth(ariesDatetime,
                                       ariesDatetime.year * 12 + ariesDatetime.month - 1 + (int) monthInterval.sign * (int) monthInterval.month);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval) {
        return genAriesDatetimeByMonth(ariesDatetime, ariesDatetime.year * 12 + (int) yearMonthInterval.sign * (int) yearMonthInterval.year * 12 +
                                                      ariesDatetime.month - 1 + (int) yearMonthInterval.sign * (int) yearMonthInterval.month);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const DayInterval &dayInterval) {
        int period = ariesDatetime.getDaynr() + (int) dayInterval.sign * (int) dayInterval.day;
        /* Daynumber from year 0 to 9999-12-31 */
        if ((uint32_t) period > MAX_DAY_NUMBER) { //valid date return 0000-000-00
            AriesDatetime tmp;
            return tmp;
        } else {
            AriesDatetime d = ariesDatetime;
            uint16_t year;
            uint8_t month, day;
            get_date_from_daynr(period, &year, &month, &day);
            d.year = year;
            d.month = month;
            d.day = day;
            return d;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval) {
        return genAriesDatetime(ariesDatetime.getDaynrOfMonth1stDay(),
                                (ariesDatetime.day - 1) * 3600 * 24L + ariesDatetime.hour * 3600 + ariesDatetime.minute * 60 + ariesDatetime.second +
                                (int) secondInterval.sign * (int64_t) secondInterval.second,
                                ariesDatetime.second_part);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime AriesTimeCalc::add(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval) {
        return genAriesDatetime(ariesDatetime.getDaynrOfMonth1stDay(),
                                (ariesDatetime.day - 1) * 3600 * 24L + ariesDatetime.hour * 3600 + ariesDatetime.minute * 60 + ariesDatetime.second +
                                (int) secondPartInterval.sign * (int64_t) secondPartInterval.second,
                                (int) ariesDatetime.second_part + (int) secondPartInterval.sign * secondPartInterval.second_part);
    }

    //for timestampdiff
    //date2 - date1
    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesTimeCalc::timestampdiff(const AriesDate &date1, const AriesDate &date2, const interval_type &type) {
        switch (type) {
            case INTERVAL_YEAR:
                return date2.diffYears(date1);
            case INTERVAL_QUARTER:
                return date2.diffQuarters(date1);
            case INTERVAL_MONTH:
                return date2.diffMonths(date1);
            case INTERVAL_WEEK:
                return date2.diffWeeks(date1);
            case INTERVAL_DAY:
                return date2.diffDays(date1);
            case INTERVAL_HOUR:
                return date2.diffHours(date1);
            case INTERVAL_MINUTE:
                return date2.diffMinutes(date1);
            case INTERVAL_SECOND:
                return date2.diffSeconds(date1);
            case INTERVAL_MICROSECOND:
                return date2.diffMicroSeconds(date1);
            default:
                return 0;
        }
    }

    // datetime - date
    ARIES_HOST_DEVICE_NO_INLINE int64_t
    AriesTimeCalc::timestampdiff(const AriesDate &date, const AriesDatetime &datetime, const interval_type &type) {
        AriesDatetime dt(date.year, date.month, date.day, 0, 0, 0, 0);
        switch (type) {
            case INTERVAL_YEAR:
                return datetime.diffYears(dt);
            case INTERVAL_QUARTER:
                return datetime.diffQuarters(dt);
            case INTERVAL_MONTH:
                return datetime.diffMonths(dt);
            case INTERVAL_WEEK:
                return datetime.diffWeeks(dt);
            case INTERVAL_DAY:
                return datetime.diffDays(dt);
            case INTERVAL_HOUR:
                return datetime.diffHours(dt);
            case INTERVAL_MINUTE:
                return datetime.diffMinutes(dt);
            case INTERVAL_SECOND:
                return datetime.diffSeconds(dt);
            case INTERVAL_MICROSECOND:
                return datetime.diffMicroSeconds(dt);
            default:
                return 0;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t
    AriesTimeCalc::timestampdiff(const AriesDatetime &datetime, const AriesDate &date, const interval_type &type) {
        return -timestampdiff(date, datetime, type);
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t
    AriesTimeCalc::timestampdiff(const AriesDatetime &datetime1, const AriesDatetime &datetime2, const interval_type &type) {
        switch (type) {
            case INTERVAL_YEAR:
                return datetime2.diffYears(datetime1);
            case INTERVAL_QUARTER:
                return datetime2.diffQuarters(datetime1);
            case INTERVAL_MONTH:
                return datetime2.diffMonths(datetime1);
            case INTERVAL_WEEK:
                return datetime2.diffWeeks(datetime1);
            case INTERVAL_DAY:
                return datetime2.diffDays(datetime1);
            case INTERVAL_HOUR:
                return datetime2.diffHours(datetime1);
            case INTERVAL_MINUTE:
                return datetime2.diffMinutes(datetime1);
            case INTERVAL_SECOND:
                return datetime2.diffSeconds(datetime2);
            case INTERVAL_MICROSECOND:
                return datetime2.diffMicroSeconds(datetime2);
            default:
                return 0;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesDate &ariesDate) {
        AriesDate date = ariesDate;
        return date;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesDatetime &ariesDatetime) {
        AriesDate date(ariesDatetime.year, ariesDatetime.month, ariesDatetime.day);
        return date;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE(const AriesTimestamp &timestamp) {
        return AriesDate(timestamp);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const DayInterval &dayInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const DayInterval &dayInterval) {
        DayInterval interval = dayInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const YearInterval &yearInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const YearInterval &yearInterval) {
        YearInterval interval = yearInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const MonthInterval &monthInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const MonthInterval &monthInterval) {
        MonthInterval interval = monthInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_ADD(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, yearMonthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate DATE_SUB(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval) {
        AriesTimeCalc ariesTimeCalc;
        YearMonthInterval interval = yearMonthInterval;
        interval.sign = -interval.sign;
        return ariesTimeCalc.add(ariesDate, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDate &ariesDate, const SecondInterval &secondInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDate &ariesDate, const SecondInterval &secondInterval) {
        AriesTimeCalc ariesTimeCalc;
        SecondInterval interval = secondInterval;
        interval.sign = -interval.sign;
        return ariesTimeCalc.add(ariesDate, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDate, secondPartInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval) {
        AriesTimeCalc ariesTimeCalc;
        SecondPartInterval interval = secondPartInterval;
        interval.sign = -interval.sign;
        return ariesTimeCalc.add(ariesDate, secondPartInterval);
    }

    //for AriesDatetime
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const DayInterval &dayInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const DayInterval &dayInterval) {
        DayInterval interval = dayInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval) {
        YearInterval interval = yearInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval) {
        MonthInterval interval = monthInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, yearMonthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval) {
        YearMonthInterval interval = yearMonthInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval) {
        SecondInterval interval = secondInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_ADD(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, secondPartInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime DATE_SUB(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval) {
        SecondPartInterval interval = secondPartInterval;
        interval.sign = -interval.sign;
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.add(ariesDatetime, interval);
    }

    //for SUBDATE
    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const DayInterval &dayInterval) {
        return DATE_SUB(ariesDate, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const YearInterval &yearInterval) {
        return DATE_SUB(ariesDate, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const MonthInterval &monthInterval) {
        return DATE_SUB(ariesDate, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate SUBDATE(const AriesDate &ariesDate, const YearMonthInterval &yearMonthInterval) {
        return DATE_SUB(ariesDate, yearMonthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDate &ariesDate, const SecondInterval &secondInterval) {
        return DATE_SUB(ariesDate, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDate &ariesDate, const SecondPartInterval &secondPartInterval) {
        return DATE_SUB(ariesDate, secondPartInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const DayInterval &dayInterval) {
        return DATE_SUB(ariesDatetime, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const YearInterval &yearInterval) {
        return DATE_SUB(ariesDatetime, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const MonthInterval &monthInterval) {
        return DATE_SUB(ariesDatetime, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const YearMonthInterval &yearMonthInterval) {
        return DATE_SUB(ariesDatetime, yearMonthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const SecondInterval &secondInterval) {
        return DATE_SUB(ariesDatetime, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBDATE(const AriesDatetime &ariesDatetime, const SecondPartInterval &secondPartInterval) {
        return DATE_SUB(ariesDatetime, secondPartInterval);
    }

    //for DATEDIFF / TIMEDIFF
    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDate &date1, const AriesDate &date2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.datediff(date1, date2);
    }

    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.datediff(datetime1, datetime2);
    }

    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDatetime &datetime, const AriesDate &date) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.datediff(datetime, date);
    }

    ARIES_HOST_DEVICE_NO_INLINE int DATEDIFF(const AriesDate &date, const AriesDatetime &datetime) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.datediff(date, datetime);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timediff(datetime1, datetime2);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesDate &date1, const AriesDate &date2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timediff(date1, date2);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime TIMEDIFF(const AriesTime &time1, const AriesTime &time2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timediff(time1, time2);
    }

    //for SUBTIME
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime SUBTIME(const AriesDatetime &ariesDatetime, const AriesTime &time) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.subtime(ariesDatetime, time);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime SUBTIME(const AriesTime &time1, const AriesTime &time2) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.subtime(time1, time2);
    }

    //for EXTRACT
    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesDate &date, uint32_t week_behaviour) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.extract(type, date, week_behaviour);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesDatetime &datetime, uint32_t week_behaviour) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.extract(type, datetime, week_behaviour);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const interval_type &type, const AriesTime &time) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.extract(type, time);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesDate &date, const interval_type &type, uint32_t week_behaviour) {
        return EXTRACT(type, date, week_behaviour);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesDatetime &datetime, const interval_type &type, uint32_t week_behaviour) {
        return EXTRACT(type, datetime,  week_behaviour);
    }

    ARIES_HOST_DEVICE_NO_INLINE int32_t EXTRACT(const AriesTime &time, const interval_type &type, uint32_t week_behaviour) {
        return EXTRACT(type, time);
    }

    //for TIMESTAMP
    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDate &date) {
        AriesDatetime datetime(date);
        return datetime;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDatetime &datetime) {
        AriesDatetime datetime1 = datetime;
        return datetime1;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDate &date, const AriesTime &time) {
        AriesTime t = time;
        t.sign = -t.sign;
        return SUBTIME(date, t);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMP(const AriesDatetime &datetime, const AriesTime &time) {
        AriesTime t = time;
        t.sign = -t.sign;
        return SUBTIME(datetime, t);
    }
    /*
     * for TIMESTAMPADD or TIMESTAMPDIFF, the interval type must be one of the following type:
     *   MICROSECOND (microseconds), SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, or YEAR.
     * */
    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate &date, const YearInterval &yearInterval) {
        return DATE_ADD(date, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate &date, const MonthInterval &monthInterval) {
        return DATE_ADD(date, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate TIMESTAMPADD(const AriesDate &date, const DayInterval &dayInterval) {
        return DATE_ADD(date, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDate &date, const SecondInterval &secondInterval) {
        return DATE_ADD(date, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDate &date, const SecondPartInterval &secondPartInterval) {
        return DATE_ADD(date, secondPartInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime &datetime, const YearInterval &yearInterval) {
        return DATE_ADD(datetime, yearInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime &datetime, const MonthInterval &monthInterval) {
        return DATE_ADD(datetime, monthInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime &datetime, const DayInterval &dayInterval) {
        return DATE_ADD(datetime, dayInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime &datetime, const SecondInterval &secondInterval) {
        return DATE_ADD(datetime, secondInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime TIMESTAMPADD(const AriesDatetime &datetime, const SecondPartInterval &secondPartInterval) {
        return DATE_ADD(datetime, secondPartInterval);
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDate &date1, const AriesDate &date2, const interval_type &type) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timestampdiff(date1, date2, type);
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDate &date, const AriesDatetime &datetime, const interval_type &type) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timestampdiff(date, datetime, type);
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDatetime &datetime, const AriesDate &date, const interval_type &type) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timestampdiff(datetime, date, type);
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t TIMESTAMPDIFF(const AriesDatetime &datetime1, const AriesDatetime &datetime2, const interval_type &type) {
        AriesTimeCalc ariesTimeCalc;
        return ariesTimeCalc.timestampdiff(datetime1, datetime2, type);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t YEAR(const AriesDate &date) {
        return date.getYear();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t YEAR(const AriesDatetime &datetime) {
        return datetime.getYear();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t QUARTER(const AriesDate &date) {
        return date.getQuarter();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t QUARTER(const AriesDatetime &datetime) {
        return datetime.getQuarter();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesDate &date) {
        return date.getMonth();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesDatetime &datetime) {
        return datetime.getMonth();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MONTH(const AriesTimestamp &timestamp) {
        return AriesDate(timestamp).getMonth();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAY(const AriesDate &date) {
        return date.getDay();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAY(const AriesDatetime &datetime) {
        return datetime.getDay();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEK(const AriesDate &date, uint8_t mode) {
        return date.getWeek(mode);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEK(const AriesDatetime &datetime, uint8_t mode) {
        return datetime.getWeek(mode);
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t HOUR(const AriesDate &date) {
        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t HOUR(const AriesDatetime &datetime) {
        return datetime.getHour();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MINUTE(const AriesDate &date) {
        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t MINUTE(const AriesDatetime &datetime) {
        return datetime.getMinute();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t SECOND(const AriesDate &date) {
        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t SECOND(const AriesDatetime &datetime) {
        return datetime.getSecond();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t MICROSECOND(const AriesDate &date) {
        return 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t MICROSECOND(const AriesDatetime &datetime) {
        return datetime.getMicroSec();
    }

//    ARIES_HOST_DEVICE_NO_INLINE char * DAYNAME(const AriesDate &date);
//    ARIES_HOST_DEVICE_NO_INLINE char * DAYNAME(const AriesDatetime &datetime);

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFMONTH(const AriesDate &date) {
        return date.getDay();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFMONTH(const AriesDatetime &datetime) {
        return datetime.getDay();
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFWEEK(const AriesDate &date) {
        bool sunday_first_day_of_week = true;
        return (uint8_t) (calc_weekday(date.getDaynr(), sunday_first_day_of_week) +  + ad_test(sunday_first_day_of_week));
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t DAYOFWEEK(const AriesDatetime &datetime) {
        bool sunday_first_day_of_week = true;
        return (uint8_t) (calc_weekday(datetime.getDaynr(), sunday_first_day_of_week) +  + ad_test(sunday_first_day_of_week));
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t DAYOFYEAR(const AriesDate &date) {
        return (uint16_t) (date.getDaynr() - date.getDaynrOfYear1stDay() + 1);
    }
    ARIES_HOST_DEVICE_NO_INLINE uint16_t DAYOFYEAR(const AriesDatetime &datetime) {
        return (uint16_t) (datetime.getDaynr() - datetime.getDaynrOfYear1stDay() + 1);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate LAST_DAY(const AriesDate &date) {
        AriesDate d = date;
        d.day = date.getLastDay();
        return d;
    }
    ARIES_HOST_DEVICE_NO_INLINE AriesDate LAST_DAY(const AriesDatetime &datetime) {
        AriesDate d(datetime.year, datetime.month, datetime.day);
        d.day = datetime.getLastDay();
        return d;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKDAY(const AriesDate &date) {
        bool sunday_first_day_of_week = false;
        return (uint8_t) (calc_weekday(date.getDaynr(), sunday_first_day_of_week) +  + ad_test(sunday_first_day_of_week));
    }
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKDAY(const AriesDatetime &datetime) {
        bool sunday_first_day_of_week = false;
        return (uint8_t) (calc_weekday(datetime.getDaynr(), sunday_first_day_of_week) +  + ad_test(sunday_first_day_of_week));
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKOFYEAR(const AriesDate &date) {
        uint32_t out;
        return calc_week((uint32_t) date.year, date.month, date.day, week_mode(3), &out);
    }
    ARIES_HOST_DEVICE_NO_INLINE uint8_t WEEKOFYEAR(const AriesDatetime &datetime) {
        uint32_t out;
        return calc_week((uint32_t) datetime.year, datetime.month, datetime.day, week_mode(3), &out);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDate FROM_DAYS(uint32_t days) {
        AriesDate date;
        uint16_t year;
        uint8_t month, day;
        get_date_from_daynr(days, &year, &month, &day);
        date.year = year;
        date.month = month;
        date.day = day;
        return date;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t YEARWEEK(const AriesDate &date, uint8_t mode) {
        uint32_t year;
        uint32_t week= calc_week(date.year, date.month, date.day, (week_mode(mode) | WEEK_YEAR), &year);
        return week + year * 100;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t YEARWEEK(const AriesDatetime &datetime, uint8_t mode) {
        uint32_t year;
        uint32_t week= calc_week(datetime.year, datetime.month, datetime.day, (week_mode(mode) | WEEK_YEAR), &year);
        return week + year * 100;
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal UNIX_TIMESTAMP(const AriesDate &date, int offset) {
        return date.getUnixTimestamp(offset);
    }

    ARIES_HOST_DEVICE_NO_INLINE Decimal UNIX_TIMESTAMP(const AriesDatetime &datetime, int offset) {
        return datetime.getUnixTimestamp(offset);
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime FROM_UNIXTIME(const AriesTimestamp &timestamp, int offset) {
        AriesDatetime datetime(timestamp.getTimeStamp(), offset);
        return datetime;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesDatetime FROM_UNIXTIME(const Decimal &timestamp, int offset) {
        SecondInterval interval(offset < 0 ? -offset : offset, offset < 0 ? -1 : 1);
        AriesDatetime datetime(timestamp, offset);
        return datetime;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTime abs(AriesTime time) {
        if (time.sign < 0) {
            time.sign = -time.sign;
        }
        return time;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t get_format_length(const char *format, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).get_format_length(format);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesDate &date, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).make_date_time(to, format, date);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesDatetime &datetime, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).make_date_time(to, format, datetime);
    }

    ARIES_HOST_DEVICE_NO_INLINE bool DATE_FORMAT(char *to, const char *format, const AriesTimestamp &timestamp, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).make_date_time(to, format, timestamp);
    }

    /*
     * only handle format such as "1001-01-01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesDate &date, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).str_to_date(str, len, date);
    }
    /*
     * only handle format such as "1001-01-01 01:01:01"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesDatetime &datetime, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).str_to_datetime(str, len, datetime);
    }
    /*
     * only handle format such as "1001-01-01 01:01:01.000"
     * */
    ARIES_HOST_DEVICE_NO_INLINE bool STRING_TO_DATE(const char *str, int len, AriesTimestamp &timestamp, const LOCALE_LANGUAGE &language) {
        return AriesDateFormat(language).str_to_timestamp(str, len, timestamp);
    }

END_ARIES_ACC_NAMESPACE