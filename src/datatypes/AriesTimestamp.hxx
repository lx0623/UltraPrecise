//
// Created by david on 19-7-25.
//

#ifndef DATETIMELIB_ARIESTIMESTAMP_HXX
#define DATETIMELIB_ARIESTIMESTAMP_HXX

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE
    struct AriesDate;
    /* it is a timestamp for UTC
     * '1970-01-01 00:00:01.000000' UTC to '2038-01-19 03:14:07.999999' UTC
     * */
    struct ARIES_PACKED AriesTimestamp {
        int64_t timestamp;

    public:
        ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp();
        ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp(int64_t t);
        ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp(const AriesDate &date);

        ARIES_HOST_DEVICE_NO_INLINE int64_t getTimeStamp() const;

        ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesTimestamp& ts) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesTimestamp& ts) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesTimestamp& ts) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesTimestamp& ts) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesTimestamp& ts) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesTimestamp& ts) const;
    };

END_ARIES_ACC_NAMESPACE

#endif //DATETIMELIB_ARIESTIMESTAMP_HXX
