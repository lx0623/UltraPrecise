//
// Created by david shen on 2020-04-17.
//

#include "AriesTruncateFunctions.hxx"

BEGIN_ARIES_ACC_NAMESPACE

ARIES_HOST_DEVICE_NO_INLINE int64_t getPower10( int32_t pow )
{
    int64_t res = 1;
    switch (pow) {
        case 0:
            res = 1;
            break;
        case 1:
            res = 10;
            break;
        case 2:
            res = 100;
            break;
        case 3:
            res = 1000;
            break;
        case 4:
            res = 10000;
            break;
        case 5:
            res = 100000;
            break;
        case 6:
            res = 1000000;
            break;
        case 7:
            res = 10000000;
            break;
        case 8:
            res = 100000000;
            break;
        case 9:
            res = 1000000000;
            break;
        case 10:
            res = 10000000000;
            break;
        case 11:
            res = 100000000000;
            break;
        case 12:
            res = 1000000000000;
            break;
        case 13:
            res = 10000000000000;
            break;
        case 14:
            res = 100000000000000;
            break;
        case 15:
            res = 1000000000000000;
            break;
        case 16:
            res = 10000000000000000;
            break;
        case 17:
            res = 100000000000000000;
            break;
        case 18:
            res = 1000000000000000000;
            break;
        default:
            break;
    }
    return res;
}

ARIES_HOST_DEVICE_NO_INLINE int8_t truncate( int8_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -3)
        {
            int8_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE uint8_t truncate( uint8_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -3)
        {
            int8_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE int16_t truncate( int16_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -5)
        {
            int16_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE uint16_t truncate( uint16_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -5)
        {
            int16_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE int32_t truncate( int32_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -10)
        {
            int32_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE uint32_t truncate( uint32_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -10)
        {
            int32_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE int64_t truncate( int64_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision > -19)
        {
            int64_t powNum = getPower10(-precision);
            num = num / powNum * powNum;
        } else {
            num = 0;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE uint64_t truncate( uint64_t num, int32_t precision )
{
    if (precision < 0)
    {
        if (precision <= -20)
        {
            num = 0;
        }
        else if (precision == 19)
        {
            uint64_t powNum = 10000000000000000000;
            num = num / powNum * powNum;
        } else {
            int64_t powNum = getPower10(precision);
            num = num / powNum * powNum;
        }
    }
    return num;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal truncate( Decimal num, int32_t precision )
{
    return num.truncate(precision);
}

END_ARIES_ACC_NAMESPACE
