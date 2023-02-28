#include "murmur_hash.h"

ARIES_HOST_DEVICE_NO_INLINE
uint32_t murmur_hash( const void* key,
                      int len,
                      const uint32_t seed,
                      bool aligned
                    )
{
    const unsigned int m = 0xc6a4a793;
    const int r = 16;
    uint32_t h = seed ^ (len * m);

    const unsigned char* data = (const unsigned char*)key;
    while (len >= 4) {
        unsigned int k = 0;
        if ( aligned )
            k = *(unsigned int*)data;
        else
            k = ( data[ 3 ] << 24 ) + ( data[ 2 ] << 16 ) + ( data[ 1 ] << 8 ) + data[ 0 ];

        h += k;
        h *= m;
        h ^= h >> 16;

        data += 4;
        len -= 4;
    }

    switch (len) {
        case 3:
        h += data[2] << 16;
        case 2:
        h += data[1] << 8;
        case 1:
        h += data[0];
        h *= m;
        h ^= h >> r;
    };

    h *= m;
    h ^= h >> 10;
    h *= m;
    h ^= h >> 17;
    return h;
}