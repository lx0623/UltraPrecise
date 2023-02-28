//
// Created by david.shen on 2019-08-21.
//

#include "AriesInnerHelper.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    //uchar days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0};
    ARIES_HOST_DEVICE_NO_INLINE uint8_t getDaysInMonth(int m) {
        switch (m) {
            case 0:
            case 2:
            case 4:
            case 6:
            case 7:
            case 9:
            case 11:
                return 31;
            case 1:
                return 28;
            case 3:
            case 5:
            case 8:
            case 10:
                return 30;
            default:
                return 0;
        }
    }

    /* Change a daynr to year, month and day */
    /* Daynr 0 is returned as date 00.00.00 */

    ARIES_HOST_DEVICE_NO_INLINE void get_date_from_daynr(int daynr, uint16_t *ret_year, uint8_t *ret_month, uint8_t *ret_day) {
        uint32_t day_of_year, days_in_year;

        if (daynr <= 365L || daynr >= 3652500) {                        /* Fix if wrong daynr */
            *ret_year = *ret_month = *ret_day = 0;
        } else {
            *ret_year = (uint32_t) (daynr * 100 / 36525L);
            //used temporary
            day_of_year = (((*ret_year - 1) / 100 + 1) * 3) / 4;
            day_of_year = (uint32_t) (daynr - (int) *ret_year * 365L) - (*ret_year - 1) / 4 + day_of_year;
            while (day_of_year > (days_in_year = calc_days_in_year(*ret_year))) {
                day_of_year -= days_in_year;
                (*ret_year)++;
            }
            uint32_t leap_day = 0;
            if (days_in_year == 366) {
                if (day_of_year > 31 + 28) {
                    day_of_year--;
                    if (day_of_year == 31 + 28)
                        leap_day = 1;        /* Handle leapyears leapday */
                }
            }
            *ret_month = 1;
            for (int i = 0;;) {
                if (day_of_year <= (uint32_t) getDaysInMonth(i)) break;
                day_of_year -= getDaysInMonth(i++);
                (*ret_month)++;
            }
            *ret_day = day_of_year + leap_day;
        }
    }


    ARIES_HOST_DEVICE_NO_INLINE uint32_t calc_days_in_year(uint32_t year) {
        return ((year & 3) == 0 && (year % 100 || (year % 400 == 0 && year)) ? 366 : 365);
    }

    /*
      Calculate nr of day since year 0 in new date-system (from 1615)

      SYNOPSIS
        calc_daynr()
        year		 Year (exact 4 digit year, no year conversions)
        month		 Month
        day			 Day

      NOTES: 0000-00-00 is a valid date, and will return 0

      RETURN
        Days since 0000-00-00
    */

    ARIES_HOST_DEVICE_NO_INLINE int calc_daynr(uint32_t year, uint32_t month, uint32_t day) {
        int delsum, temp;
        int y = year;                                  /* may be < 0 temporarily */

        if (y == 0 && month == 0)
            return 0;                /* Skip errors */
        /* Cast to int to be able to handle month == 0 */
        delsum = (365 * y + 31 * ((int) month - 1) + (int) day);
        if (month <= 2)
            y--;
        else
            delsum -= ((int) month * 4 + 23) / 10;
        temp = (int) ((y / 100 + 1) * 3) / 4;
        return (delsum + (int) y / 4 - temp);
    } /* calc_daynr */


    /* Calc weekday from daynr */
    /* Returns 0 for monday, 1 for tuesday .... */

    ARIES_HOST_DEVICE_NO_INLINE int calc_weekday(int32_t daynr, bool sunday_first_day_of_week) {
        return ((int) ((daynr + 5L + (sunday_first_day_of_week ? 1L : 0L)) % 7));
    }

    /*
      The bits in week_format has the following meaning:
       WEEK_MONDAY_FIRST (0)  If not set	Sunday is first day of week
                           If set	Monday is first day of week
       WEEK_YEAR (1)	  If not set	Week is in range 0-53

           Week 0 is returned for the the last week of the previous year (for
        a date at start of january) In this case one can get 53 for the
        first week of next year.  This flag ensures that the week is
        relevant for the given year. Note that this flag is only
        releveant if WEEK_JANUARY is not set.

                  If set	 Week is in range 1-53.

        In this case one may get week 53 for a date in January (when
        the week is that last week of previous year) and week 1 for a
        date in December.

      WEEK_FIRST_WEEKDAY (2)  If not set	Weeks are numbered according
                           to ISO 8601:1988
                  If set	The week that contains the first
                        'first-day-of-week' is week 1.

        ISO 8601:1988 means that if the week containing January 1 has
        four or more days in the new year, then it is week 1;
        Otherwise it is the last week of the previous year, and the
        next week is week 1.
    */

    uint32_t calc_week(uint32_t year, uint8_t month, uint8_t day, uint32_t week_behaviour, uint32_t *out_year) {
        uint32_t days;
        uint32_t daynr = calc_daynr(year, month, day);
        uint32_t first_daynr = calc_daynr(year, 1, 1);
        bool monday_first = ad_test(week_behaviour & WEEK_MONDAY_FIRST);
        bool week_year = ad_test(week_behaviour & WEEK_YEAR);
        bool first_weekday = ad_test(week_behaviour & WEEK_FIRST_WEEKDAY);

        uint32_t weekday = calc_weekday(first_daynr, !monday_first);
        *out_year = year;

        if (month == 1 && day <= 7 - weekday) {
            if (!week_year &&
                ((first_weekday && weekday != 0) ||
                 (!first_weekday && weekday >= 4)))
                return 0;
            week_year = 1;
            (*out_year)--;
            first_daynr -= (days = calc_days_in_year(*out_year));
            weekday = (weekday + 53 * 7 - days) % 7;
        }

        if ((first_weekday && weekday != 0) ||
            (!first_weekday && weekday >= 4))
            days = daynr - (first_daynr + (7 - weekday));
        else
            days = daynr - (first_daynr - weekday);

        if (week_year && days >= 52 * 7) {
            weekday = (weekday + calc_days_in_year(*out_year)) % 7;
            if ((!first_weekday && weekday < 4) ||
                (first_weekday && weekday == 0)) {
                (*out_year)++;
                return 1;
            }
        }
        return days / 7 + 1;
    }

    /*
     * assume date_end >= date_beg, and timediff is time of date_end - time of date_begin
     * */
    ARIES_HOST_DEVICE_NO_INLINE int diff_months(int year_end, int month_end, int day_end, int year_beg, int month_beg, int day_beg, int timediff) {
//        /* calc years */
//        int years= year_end - year_beg;
//        if (month_end < month_beg || (month_end == month_beg && day_end < day_beg)) {
//            years-= 1;
//        }
//
//        /* calc months */
//        int months= 12*years;
//        if (month_end < month_beg || (month_end == month_beg && day_end < day_beg)) {
//            months+= 12 - (month_beg - month_end);
//        } else {
//            months+= (month_end - month_beg);
//        }
//        if (day_end < day_beg) {
//            months-= 1;
//        }

//        return months;

        //optimized
        int months = (year_end - year_beg) * 12 + (month_end - month_beg);
        if (day_end < day_beg || (day_end == day_beg && timediff < 0)) {
            months -= 1;
        }

        return months;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t week_mode(uint32_t mode) {
        uint32_t week_format = (mode & 7);
        if (!(week_format & WEEK_MONDAY_FIRST)) {
            week_format ^= WEEK_FIRST_WEEKDAY;
        }
        return week_format;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint8_t get_last_day(uint16_t year, uint8_t month) {
        uint8_t monthIndex = month - 1;
        uint8_t lastDay = getDaysInMonth(monthIndex);
        if (monthIndex == 1 && calc_days_in_year(year) == 366) {
            lastDay += 1;
        }
        return lastDay;
    }


END_ARIES_ACC_NAMESPACE