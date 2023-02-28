//
// Created by tengjp on 19-7-24.
//

#ifndef AIRES_MYSQL_THREAD_H
#define AIRES_MYSQL_THREAD_H

#include <pthread.h>
#include "utils/mutex_lock.h"
extern pthread_mutexattr_t fast_mutexattr;
#define MY_MUTEX_INIT_FAST &fast_mutexattr
typedef pthread_mutex_t mysql_mutex_t;
typedef pthread_cond_t mysql_cond_t;
typedef pthread_rwlock_t mysql_rwlock_t;
#define mysql_mutex_init(M, A) \
    pthread_mutex_init(M, A)
#define mysql_mutex_destroy(M) \
    pthread_mutex_destroy(M)
#define mysql_mutex_lock(M) \
    pthread_mutex_lock(M)
#define mysql_mutex_trylock(M) \
    pthread_mutex_trylock(M)
#define mysql_mutex_unlock(M) \
    pthread_mutex_unlock(M)
#define mysql_rwlock_init(RW) pthread_rwlock_init(RW, NULL)
#define mysql_rwlock_destroy(RW) pthread_rwlock_destroy(RW)
#define mysql_rwlock_rdlock(RW) \
    pthread_rwlock_rdlock(RW)
#define mysql_rwlock_wrlock(RW) \
    pthread_rwlock_wrlock(RW)
#define mysql_rwlock_tryrdlock(RW) \
    pthread_rwlock_tryrdlock(RW)
#define mysql_rwlock_trywrlock(RW) \
    pthread_rwlock_trywrlock(RW)
#define mysql_rwlock_unlock(RW) pthread_rwlock_unlock(RW)

#define mysql_mutex_assert_owner(M) { }

#define mysql_cond_init(K, C, A) pthread_cond_init(C, A)
#define mysql_cond_destroy(C) pthread_cond_destroy(C)
#define mysql_cond_wait(C, M) \
    pthread_cond_wait(C, M)
#define mysql_cond_timedwait(C, M, W) \
    pthread_cond_timedwait(C, M, W)
#define mysql_cond_signal(C) pthread_cond_signal(C)
#define mysql_cond_broadcast(C) pthread_cond_broadcast(C)
#endif //AIRES_MYSQL_THREAD_H
