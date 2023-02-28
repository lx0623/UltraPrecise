//
// Created by david shen on 2020-04-17.
//

#pragma once

#include "AriesDefinition.h"
#include "decimal.hxx"

BEGIN_ARIES_ACC_NAMESPACE

ARIES_HOST_DEVICE_NO_INLINE int64_t getPower10( int32_t pow );
ARIES_HOST_DEVICE_NO_INLINE int8_t truncate( int8_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE uint8_t truncate( uint8_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE int16_t truncate( int16_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE uint16_t truncate( uint16_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE int32_t truncate( int32_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE uint32_t truncate( uint32_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE int64_t truncate( int64_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE uint64_t truncate( uint64_t num, int32_t precision );
ARIES_HOST_DEVICE_NO_INLINE Decimal truncate( Decimal num, int32_t precision );

END_ARIES_ACC_NAMESPACE
