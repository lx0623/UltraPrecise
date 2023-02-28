#include <thread>
#include "thread.h"
size_t getConcurrency( size_t totalJobCnt,
                       vector<size_t>& threadsJobCnt,
                       vector<size_t>& threadsJobStartIdx )
{
    if ( 0 == totalJobCnt )
        return 0;
    size_t threadCnt = thread::hardware_concurrency();
    threadCnt = totalJobCnt < threadCnt ? totalJobCnt : threadCnt;
    size_t perThreadJobCount = totalJobCnt / threadCnt; // count of columns for each thread to handle
    size_t extraJobCount = totalJobCnt % threadCnt;
    threadsJobCnt.assign( threadCnt, perThreadJobCount );
    threadsJobStartIdx.assign( threadCnt, 0 );
    for ( size_t tmpJobIdx = 0; tmpJobIdx < extraJobCount; ++tmpJobIdx )
    {
        threadsJobCnt[ tmpJobIdx ] += 1;
    }
    if ( threadCnt > 1 )
    {
        for ( size_t threadIdx = 1; threadIdx < threadCnt; ++threadIdx )
        {
            threadsJobStartIdx[ threadIdx ] = threadsJobStartIdx[ threadIdx - 1 ] +
                                              threadsJobCnt[ threadIdx -1 ];
        }
    }
    return threadCnt;
}