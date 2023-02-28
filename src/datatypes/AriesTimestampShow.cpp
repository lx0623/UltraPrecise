//
// Created by david shen on 2019-07-19.
//

#include <string.h>
#include "AriesTimestampShow.h"


BEGIN_ARIES_ACC_NAMESPACE

    AriesTimestampShow::AriesTimestampShow() {
        memset(this, 0x00, sizeof(AriesTimestampShow));
    }

    AriesTimestampShow::AriesTimestampShow(uint16_t y, uint8_t m, uint8_t d, uint8_t h, uint8_t mt, uint8_t s,
                                           uint32_t sp) {
        year = y;
        month = m;
        day = d;
        hour = h;
        minute = mt;
        second = s;
        second_part = sp;
    }

END_ARIES_ACC_NAMESPACE
