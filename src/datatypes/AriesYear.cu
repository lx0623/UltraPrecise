//
// Created by david shen on 2019-07-19.
//

#include <cstring>
#include "AriesYear.hxx"

BEGIN_ARIES_ACC_NAMESPACE
    ARIES_HOST_DEVICE_NO_INLINE AriesYear::AriesYear() {
        memset(this, 0x00, sizeof(AriesYear));
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesYear::AriesYear(uint16_t y) : year(y) {
    }

    ARIES_HOST_DEVICE_NO_INLINE AriesYear::AriesYear(int32_t y) : year(y) {
    }

    ARIES_HOST_DEVICE_NO_INLINE uint16_t AriesYear::getYear() const {
        return year;
    }

    // for AriesYear
    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator>(const AriesYear &y) const {
        return year > y.year;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator>=(const AriesYear &y) const {
        return year >= y.year;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator<(const AriesYear &y) const {
        return year < y.year;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator<=(const AriesYear &y) const {
        return year <= y.year;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator==(const AriesYear &y) const {
        return year == y.year;
    }

    ARIES_HOST_DEVICE_NO_INLINE bool AriesYear::operator!=(const AriesYear &y) const {
        return year != y.year;
    }

END_ARIES_ACC_NAMESPACE