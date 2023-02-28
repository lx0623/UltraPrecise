#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "AriesDefinition.h"

#define HASH_SEED1 0
#define HASH_SEED2 3

ARIES_HOST_DEVICE_NO_INLINE
uint32_t murmur_hash( const void* key,
                      int len,
                      const uint32_t seed,
                      bool aligned = true );