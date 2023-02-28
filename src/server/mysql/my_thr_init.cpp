//
// Created by tengjp on 19-12-11.
//

#include <glog/logging.h>
#include <server/mysql/include/my_thread_local.h>
#include <server/mysql/include/mysql_def.h>
#include <server/mysql/include/mysql_thread.h>

static my_bool THR_KEY_mysys_initialized= FALSE;
static my_bool my_thread_global_init_done= FALSE;

static thread_local_key_t THR_KEY_myerrno;

/**
  initialize thread environment

  @retval  FALSE  ok
  @retval  TRUE   error (Couldn't create THR_KEY_mysys)
*/

my_bool my_thread_global_init() {
    int pth_ret;

    if (my_thread_global_init_done)
        return FALSE;
    my_thread_global_init_done = TRUE;

    /*
    Set mutex type to "fast" a.k.a "adaptive"

    In this case the thread may steal the mutex from some other thread
    that is waiting for the same mutex.  This will save us some
    context switches but may cause a thread to 'starve forever' while
    waiting for the mutex (not likely if the code within the mutex is
    short).
  */
    pthread_mutexattr_init(&fast_mutexattr);
    pthread_mutexattr_settype(&fast_mutexattr,
                              PTHREAD_MUTEX_ADAPTIVE_NP);

    if ((pth_ret= my_create_thread_local_key(&THR_KEY_myerrno, NULL)) != 0)
    { /* purecov: begin inspected */
        LOG(ERROR) << "Can't initialize threads: error %d" << pth_ret;
        /* purecov: end */
        return TRUE;
    }
    THR_KEY_mysys_initialized= TRUE;
    return FALSE;
}

void my_thread_global_end()
{
    DBUG_ASSERT(THR_KEY_mysys_initialized);
    my_delete_thread_local_key(THR_KEY_myerrno);
    THR_KEY_mysys_initialized= FALSE;
    pthread_mutexattr_destroy(&fast_mutexattr);
    my_thread_global_init_done= FALSE;
}

int my_errno()
{
    if (THR_KEY_mysys_initialized)
        return  (int)(intptr)my_get_thread_local(THR_KEY_myerrno);
    return 0;
}

void set_my_errno(int my_errno)
{
    if (THR_KEY_mysys_initialized)
        (void) my_set_thread_local(THR_KEY_myerrno, (void*)(intptr)my_errno);
}