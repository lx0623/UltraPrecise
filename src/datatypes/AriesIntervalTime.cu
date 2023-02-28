//
// Created by david on 19-7-25.
//

#include "AriesIntervalTime.hxx"

BEGIN_ARIES_ACC_NAMESPACE

    ARIES_HOST_DEVICE_NO_INLINE YearInterval::YearInterval(): year(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE YearInterval::YearInterval(uint16_t y, int8_t sign) : year(y), sign(sign) {
    }

    ARIES_HOST_DEVICE_NO_INLINE MonthInterval::MonthInterval() : month(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE MonthInterval::MonthInterval(uint32_t m, int8_t sign) : month(m), sign(sign) {
    }

    ARIES_HOST_DEVICE_NO_INLINE YearMonthInterval::YearMonthInterval() : year(0), month(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE YearMonthInterval::YearMonthInterval(uint16_t y, uint32_t m, int8_t sign) : year(y), month(m), sign(sign) {
    }

    ARIES_HOST_DEVICE_NO_INLINE DayInterval::DayInterval() : day(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE DayInterval::DayInterval(uint32_t d, int8_t sign) : day(d), sign(sign) {
    }

    ARIES_HOST_DEVICE_NO_INLINE SecondInterval::SecondInterval() : second(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE SecondInterval::SecondInterval(uint64_t s, int8_t sign) : second(s), sign(sign) {
    }

    ARIES_HOST_DEVICE_NO_INLINE SecondPartInterval::SecondPartInterval() : second(0), second_part(0), sign(1) {
    }

    ARIES_HOST_DEVICE_NO_INLINE SecondPartInterval::SecondPartInterval(uint64_t s, uint32_t sp, int8_t sign) : second(s), second_part(sp), sign(sign) {
    }

END_ARIES_ACC_NAMESPACE