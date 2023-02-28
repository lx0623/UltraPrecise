//
// Created by david shen on 2019-07-19.
//

#pragma once

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

    /* it is a timestamp for Timezone
     * transferred from Timestamp '1970-01-01 00:00:01.000000' UTC to '2038-01-19 03:14:07.999999' UTC
     * */
    struct ARIES_PACKED AriesTimestampShow {
        uint16_t year;
        uint8_t month;
        uint8_t day;
        uint8_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t second_part;

        AriesTimestampShow();
        AriesTimestampShow(uint16_t y, uint8_t m, uint8_t d, uint8_t h, uint8_t mt, uint8_t s, uint32_t ms);

    };

END_ARIES_ACC_NAMESPACE

