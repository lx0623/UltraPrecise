#ifndef CPU_TIMER_H
#define CPU_TIMER_H

#include <stddef.h>
#include <sys/time.h>
#include <cuda_runtime.h>

namespace aries{

struct CPU_Timer
{
    long start;
    long stop;
    void begin()
    {
        struct timeval tv;
        gettimeofday( &tv, NULL );
        start = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    }
    long end(bool sync=false)
    {
        if(sync)
            cudaDeviceSynchronize();
        struct timeval tv;
        gettimeofday( &tv, NULL );
        stop = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        long elapsed = stop - start;
        //printf( "cpu time: %ld\n", elapsed );
        return elapsed;
    }
};


}

#endif //CPU_TIMER_H