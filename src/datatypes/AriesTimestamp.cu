//
// Created by david on 19-7-25.
//

#include "AriesTimestamp.hxx"
#include "AriesDate.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp::AriesTimestamp() {
        // 0 indicates invalid timestamp
        timestamp = 0;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp::AriesTimestamp(int64_t t) {
        timestamp = t;
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesTimestamp::AriesTimestamp(const AriesDate &date) {
        timestamp = date.toTimestamp();
    }

    ARIES_HOST_DEVICE_NO_INLINE int64_t AriesTimestamp::getTimeStamp() const {
        return timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator>(const AriesTimestamp &ts) const {
        return timestamp > ts.timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator>=(const AriesTimestamp& ts) const {
        return timestamp >= ts.timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator<(const AriesTimestamp& ts) const {
        return timestamp < ts.timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator<=(const AriesTimestamp& ts) const {
        return timestamp <= ts.timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator==(const AriesTimestamp& ts) const {
        return timestamp == ts.timestamp;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesTimestamp::operator!=(const AriesTimestamp& ts) const {
        return timestamp != ts.timestamp;
    }


END_ARIES_ACC_NAMESPACE