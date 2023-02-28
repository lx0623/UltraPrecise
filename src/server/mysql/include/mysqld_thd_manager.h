/* Copyright (c) 2013, 2015, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA */

#ifndef MYSQLD_THD_MANAGER_INCLUDED
#define MYSQLD_THD_MANAGER_INCLUDED

#include <unordered_map>
#include <vector>
#include "gcc_atomic.h"
#include "mysql_thread.h"
#include "my_dbug.h"
#include "my_thread_local.h"

class THD;
/**
  This class maintains THD object of all registered threads.
  It provides interface to perform functions such as find, count,
  perform some action for each THD object in the list.

  It also provide mutators for inserting, and removing an element:
  add_thd() inserts a THD into the set, and increments the counter.
  remove_thd() removes a THD from the set, and decrements the counter.
  Method remove_thd() also broadcasts COND_thd_list.
*/

class Global_THD_manager {
public:
    /**
      Value for thread_id reserved for THDs which does not have an
      assigned value yet. get_new_thread_id() will never return this
      value.
    */
    static const uint32_t reserved_thread_id;

    /**
      Retrieves singleton instance
    */
    static Global_THD_manager *get_instance() {
        DBUG_ASSERT(thd_manager != NULL);
        return thd_manager;
    }

    /**
      Initializes the thd manager.
      Must be called before get_instance() can be used.

      @return true if initialization failed, false otherwise.
    */
    static bool create_instance();

    /**
      Destroys the singleton instance.
    */
    static void destroy_instance();

    /**
    Adds THD to global THD list.

    @param thd THD object
  */
    void add_thd(THD *thd);

    /**
      Removes THD from global THD list.

      @param thd THD object
    */
    void remove_thd(THD *thd);

    /**
      Retrieves thread running statistic variable.
      @return int Returns the total number of threads currently running
      @note       This is a dirty read.
    */
    int get_num_thread_running() const { return num_thread_running; }

    /**
      Increments thread running statistic variable.
    */
    void inc_thread_running()
    {
        my_atomic_add32(&num_thread_running, 1);
    }

    /**
      Decrements thread running statistic variable.
    */
    void dec_thread_running()
    {
        my_atomic_add32(&num_thread_running, -1);
    }

    /**
      Retrieves thread created statistic variable.
      @return ulonglong Returns the total number of threads created
                        after server start
      @note             This is a dirty read.
    */
    ulonglong get_num_thread_created() const
    {
        return static_cast<ulonglong>(thread_created);
    }

    /**
      Increments thread created statistic variable.
    */
    void inc_thread_created()
    {
        // my_atomic_add64(&thread_created, 1);
        __atomic_fetch_add(&thread_created, 1, __ATOMIC_SEQ_CST);
    }

    /**
      Returns an unused thread id.
    */
    my_thread_id get_new_thread_id();

    /**
      Releases a thread id so that it can be reused.
      Note that this is done automatically by remove_thd().
    */
    void release_thread_id(my_thread_id thread_id);

    /**
      Retrieves thread id counter value.
      @return my_thread_id Returns the thread id counter value
      @note                This is a dirty read.
    */
    my_thread_id get_thread_id() const { return thread_id_counter; }

    /**
      Sets thread id counter value. Only used in testing for now.
      @param new_id  The next ID to hand out (if it's unused).
    */
    void set_thread_id_counter(my_thread_id new_id);

    /**
      Retrieves total number of items in global THD list.
      @return uint Returns the count of items in global THD list
      @note        This is a dirty read.
    */
    uint get_thd_count() const { return global_thd_count; }

    /**
    Returns a pointer to the first THD for which operator() returns true.
    @param func Object of class which overrides operator()
    @return THD
      @retval THD* Matching THD
      @retval NULL When THD is not found in the list
  */
    THD* find_thd(my_thread_id id);

    // Declared static as it is referenced in handle_fatal_signal()
    static int global_thd_count;

private:
    Global_THD_manager();
    ~Global_THD_manager();

    // Singleton instance.
    static Global_THD_manager *thd_manager;

    std::unordered_map<uint32_t, THD*> thd_map;
    std::unordered_map<uint32_t, uint32_t> thread_ids;

    mysql_cond_t COND_thd_list;

    // Mutex that guards thd_list
    mysql_mutex_t LOCK_thd_list;
    // Mutex used to guard removal of elements from thd list.
    mysql_mutex_t LOCK_thd_remove;
    // Mutex protecting thread_ids
    mysql_mutex_t LOCK_thread_ids;

    // Count of active threads which are running queries in the system.
    volatile int32_t num_thread_running;

    // Cumulative number of threads created by mysqld daemon.
    volatile int64_t thread_created;

    // Counter to assign thread id.
    my_thread_id thread_id_counter;

};

struct CONN_ARG {
    bool unix_sock;
    char client_addr[INET6_ADDRSTRLEN] = {0}; // enough for IPV6 address
    int client_port;
    int client_fd;
};

#endif //AIRES_MYSQLD_THD_MANAGER_H
