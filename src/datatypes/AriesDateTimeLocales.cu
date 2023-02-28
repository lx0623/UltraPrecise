//
// Created by david.shen on 2019/12/18.
//

#include <cassert>
#include "AriesDateTimeLocales.h"
#include "AriesDataTypeUtil.hxx"

BEGIN_ARIES_ACC_NAMESPACE
    //for en_US
    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_month_name(char *des, int monthIndex, int &len) {
        switch (monthIndex) {
            case 1:
                aries_strcpy(des, "January");
                len = 7;
                break;
            case 2:
                aries_strcpy(des, "February");
                len = 8;
                break;
            case 3:
                aries_strcpy(des, "March");
                len = 5;
                break;
            case 4:
                aries_strcpy(des, "April");
                len = 5;
                break;
            case 5:
                aries_strcpy(des, "May");
                len = 3;
                break;
            case 6:
                aries_strcpy(des, "June");
                len = 4;
                break;
            case 7:
                aries_strcpy(des, "July");
                len = 4;
                break;
            case 8:
                aries_strcpy(des, "August");
                len = 6;
                break;
            case 9:
                aries_strcpy(des, "September");
                len = 9;
                break;
            case 10:
                aries_strcpy(des, "October");
                len = 7;
                break;
            case 11:
                aries_strcpy(des, "November");
                len = 8;
                break;
            case 12:
                aries_strcpy(des, "December");
                len = 8;
                break;
            default:
                aries_strcpy(des, "Monthfake");
                len = 9;
                assert(0);
                break;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_month_name_max_len() {
        return 9;
    }

    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_month_ab_name(char *des, int monthIndex, int &len) {
        switch (monthIndex) {
            case 1:
                aries_strcpy(des, "Jan");
                break;
            case 2:
                aries_strcpy(des, "Feb");
                break;
            case 3:
                aries_strcpy(des, "Mar");
                break;
            case 4:
                aries_strcpy(des, "Apr");
                break;
            case 5:
                aries_strcpy(des, "May");
                break;
            case 6:
                aries_strcpy(des, "Jun");
                break;
            case 7:
                aries_strcpy(des, "Jul");
                break;
            case 8:
                aries_strcpy(des, "Aug");
                break;
            case 9:
                aries_strcpy(des, "Sep");
                break;
            case 10:
                aries_strcpy(des, "Oct");
                break;
            case 11:
                aries_strcpy(des, "Nov");
                break;
            case 12:
                aries_strcpy(des, "Dec");
                break;
            default:
                aries_strcpy(des, "Fak");
                assert(0);
                break;
        }
        len = 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_month_ab_name_max_len() {
        return 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_weekday_name(char *des, int weekIndex, int &len) {
        switch (weekIndex) {
            case 1:
                aries_strcpy(des, "Monday");
                len = 6;
                break;
            case 2:
                aries_strcpy(des, "Tuesday");
                len = 7;
                break;
            case 3:
                aries_strcpy(des, "Wednesday");
                len = 9;
                break;
            case 4:
                aries_strcpy(des, "Thursday");
                len = 8;
                break;
            case 5:
                aries_strcpy(des, "Friday");
                len = 6;
                break;
            case 6:
                aries_strcpy(des, "Saturday");
                len = 8;
                break;
            case 7:
                aries_strcpy(des, "Sunday");
                len = 6;
                break;
            default:
                aries_strcpy(des, "Fakeday");
                len = 7;
                break;
        }
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_weekday_name_max_len() {
        return 9;
    }

    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_weekday_ab_name(char *des, int weekIndex, int &len) {
        switch (weekIndex) {
            case 1:
                aries_strcpy(des, "Mon");
                break;
            case 2:
                aries_strcpy(des, "Tue");
                break;
            case 3:
                aries_strcpy(des, "Wed");
                break;
            case 4:
                aries_strcpy(des, "Thu");
                break;
            case 5:
                aries_strcpy(des, "Fri");
                break;
            case 6:
                aries_strcpy(des, "Sat");
                break;
            case 7:
                aries_strcpy(des, "Sun");
                break;
            default:
                aries_strcpy(des, "Fkd");
                break;
        }
        len = 3;
    }

    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_weekday_ab_name_max_len() {
        return 3;
    }


END_ARIES_ACC_NAMESPACE