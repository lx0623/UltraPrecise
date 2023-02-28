//
// Created by david shen on 2019-07-19.
//

#ifndef DATETIMELIB_ARIESYEAR_HXX
#define DATETIMELIB_ARIESYEAR_HXX

#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

    struct ARIES_PACKED AriesYear {
        uint16_t year;

    public:
        ARIES_HOST_DEVICE_NO_INLINE AriesYear();
        ARIES_HOST_DEVICE_NO_INLINE AriesYear(uint16_t y);
        ARIES_HOST_DEVICE_NO_INLINE AriesYear(int32_t y);

        ARIES_HOST_DEVICE_NO_INLINE uint16_t getYear() const;

        // for AriesYear
        ARIES_HOST_DEVICE_NO_INLINE bool operator>(const AriesYear& y) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const AriesYear& y) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<(const AriesYear& y) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const AriesYear& y) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator==(const AriesYear& y) const;
        ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const AriesYear& y) const;
    };

END_ARIES_ACC_NAMESPACE

#endif //DATETIMELIB_ARIESYEAR_HXX
