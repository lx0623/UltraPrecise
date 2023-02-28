//
// Created by tengjp on 19-7-25.
//

#ifndef AIRES_MY_PTHREAD_H
#define AIRES_MY_PTHREAD_H
#include <sched.h>
ulonglong my_getsystime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (ulonglong)tp.tv_sec*10000000+(ulonglong)tp.tv_nsec/100;
}
/*
  The defines set_timespec and set_timespec_nsec should be used
  for calculating an absolute time at which
  pthread_cond_timedwait should timeout
*/
#define set_timespec(ABSTIME,SEC) set_timespec_nsec((ABSTIME),(SEC)*1000000000ULL)

#ifndef set_timespec_nsec
#define set_timespec_nsec(ABSTIME,NSEC)                                 \
  set_timespec_time_nsec((ABSTIME),my_getsystime(),(NSEC))
#endif /* !set_timespec_nsec */

/* adapt for two different flavors of struct timespec */
#ifdef HAVE_TIMESPEC_TS_SEC
#define MY_tv_sec  ts_sec
#define MY_tv_nsec ts_nsec
#else
#define MY_tv_sec  tv_sec
#define MY_tv_nsec tv_nsec
#endif /* HAVE_TIMESPEC_TS_SEC */

#ifndef set_timespec_time_nsec
#define set_timespec_time_nsec(ABSTIME,TIME,NSEC) do {                  \
  ulonglong nsec= (NSEC);                                               \
  ulonglong now= (TIME) + (nsec/100);                                   \
  (ABSTIME).MY_tv_sec=  (now / 10000000ULL);                          \
  (ABSTIME).MY_tv_nsec= (now % 10000000ULL * 100 + (nsec % 100));     \
} while(0)
#endif /* !set_timespec_time_nsec */

#endif //AIRES_MY_PTHREAD_H
