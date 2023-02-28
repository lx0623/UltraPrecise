//
// Created by david.shen on 2019/12/18.
//

#ifndef DATATYPELIB_ARIESDATETIMELOCALES_H
#define DATATYPELIB_ARIESDATETIMELOCALES_H

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE
    //for en_US
    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_month_name(char *des, int monthIndex, int &len);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_month_name_max_len();
    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_month_ab_name(char *des, int monthIndex, int &len);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_month_ab_name_max_len();
    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_weekday_name(char *des, int weekIndex, int &len);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_weekday_name_max_len();
    ARIES_HOST_DEVICE_NO_INLINE void en_US_get_weekday_ab_name(char *des, int weekIndex, int &len);
    ARIES_HOST_DEVICE_NO_INLINE uint32_t en_US_get_weekday_ab_name_max_len();
END_ARIES_ACC_NAMESPACE

#endif //DATATYPELIB_ARIESDATETIMELOCALES_H
