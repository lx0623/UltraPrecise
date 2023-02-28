//
// Created by david.shen on 2019-08-21.
//

#ifndef DATETIMELIB_ARIESINNERHELPER_HXX
#define DATETIMELIB_ARIESINNERHELPER_HXX

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

/* Flags for calc_week() function.  */
/* for mode:
Mode First day of week	Range	Week 1 is the first week …
0	Sunday	0-53	with a Sunday in this year
1	Monday	0-53	with 4 or more days this year
2	Sunday	1-53	with a Sunday in this year
3	Monday	1-53	with 4 or more days this year
4	Sunday	0-53	with 4 or more days this year
5	Monday	0-53	with a Monday in this year
6	Sunday	1-53	with 4 or more days this year
7	Monday	1-53	with a Monday in this year
 * */
#define WEEK_MONDAY_FIRST    1 //0x1 : NOT set: sunday is first day                                SET: monday is first day
#define WEEK_YEAR            2 //0x2 : NOT set: 0 ~ 53 week                                        SET: 1 ~ 54 week
#define WEEK_FIRST_WEEKDAY   4 //0x4 : NOT set: Weeks are numbered according to ISO 8601:1988      SET: 'first-day-of-week' is week 1.

#define DAYS_AT_TIMESTAMP_START 719528 /* daynr at 1970.01.01 */
#define DAYS_AT_TIMESTAMP_END 744383 /* daynr at 2038.01.19 */
#define SECONDS_AT_TIMESTAMP_END_DAY 11647 /* seconds at 03:14:07*/

#define ad_test(a)		((a) ? 1 : 0)

    //helper method
    ARIES_HOST_DEVICE_NO_INLINE uint8_t getDaysInMonth(int m);
    ARIES_HOST_DEVICE_NO_INLINE void get_date_from_daynr(int daynr, uint16_t *ret_year, uint8_t *ret_month, uint8_t *ret_day);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t calc_days_in_year(uint32_t year);
    ARIES_HOST_DEVICE_NO_INLINE int calc_daynr(uint32_t year, uint32_t month, uint32_t day);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t week_mode(uint32_t mode);
    ARIES_HOST_DEVICE_NO_INLINE int calc_weekday(int32_t daynr, bool sunday_first_day_of_week);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t calc_week(uint32_t year, uint8_t month, uint8_t day, uint32_t week_behaviour, uint32_t *out_year);
    /*
     * assume date_end >= date_beg, and timediff is time of date_end - time of date_begin
     * */
    ARIES_HOST_DEVICE_NO_INLINE int diff_months(int year_end, int month_end, int day_end, int year_beg, int month_beg, int day_beg, int timediff);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t week_mode(uint32_t mode);
    ARIES_HOST_DEVICE_NO_INLINE uint8_t get_last_day(uint16_t year, uint8_t month);

END_ARIES_ACC_NAMESPACE

#endif //DATETIMELIB_ARIESINNERHELPER_HXX
