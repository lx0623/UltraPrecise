/* Copyright (c) 2000, 2016, Oracle and/or its affiliates. All rights reserved.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "./include/mysqld_thd_manager.h"
#include "./include/sql_class.h"               // THD

int Global_THD_manager::global_thd_count= 0;
Global_THD_manager *Global_THD_manager::thd_manager = NULL;
const my_thread_id Global_THD_manager::reserved_thread_id= 0;

Global_THD_manager::Global_THD_manager()
        : num_thread_running(0),
          thread_created(0),
          thread_id_counter(reserved_thread_id + 1)
{
    mysql_mutex_init(&LOCK_thd_list, MY_MUTEX_INIT_FAST);
    mysql_mutex_init(&LOCK_thd_remove, MY_MUTEX_INIT_FAST);
    mysql_mutex_init(&LOCK_thread_ids, MY_MUTEX_INIT_FAST);
    mysql_cond_init(0, &COND_thd_list, nullptr);

    // The reserved thread ID should never be used by normal threads,
    // so mark it as in-use. This ID is used by temporary THDs never
    // added to the list of THDs.
    thread_ids[reserved_thread_id] = reserved_thread_id;
}


Global_THD_manager::~Global_THD_manager()
{
    thread_ids.erase(reserved_thread_id);
    DBUG_ASSERT(thd_map.empty());
    DBUG_ASSERT(thread_ids.empty());
    mysql_mutex_destroy(&LOCK_thd_list);
    mysql_mutex_destroy(&LOCK_thd_remove);
    mysql_mutex_destroy(&LOCK_thread_ids);
    mysql_cond_destroy(&COND_thd_list);
}

/*
  Singleton Instance creation
  This method do not require mutex guard as it is called only from main thread.
*/
bool Global_THD_manager::create_instance()
{
    if (thd_manager == NULL)
        thd_manager= new (std::nothrow) Global_THD_manager();
    return (thd_manager == NULL);
}


void Global_THD_manager::destroy_instance()
{
    delete thd_manager;
    thd_manager= NULL;
}


void Global_THD_manager::add_thd(THD *thd)
{
    // DBUG_PRINT("info", ("Global_THD_manager::add_thd %p", thd));
    // Should have an assigned ID before adding to the list.
    DBUG_ASSERT(thd->thread_id() != reserved_thread_id);
    mysql_mutex_lock(&LOCK_thd_list);
    thd_map[thd->thread_id()]= thd;
    ++global_thd_count;
    // Adding the same THD twice is an error.
    mysql_mutex_unlock(&LOCK_thd_list);
}


void Global_THD_manager::remove_thd(THD *thd)
{
    // DBUG_PRINT("info", ("Global_THD_manager::remove_thd %p", thd));
    mysql_mutex_lock(&LOCK_thd_remove);
    mysql_mutex_lock(&LOCK_thd_list);

    // if (!unit_test)
    //     DBUG_ASSERT(thd->release_resources_done());

    /*
      Used by binlog_reset_master.  It would be cleaner to use
      DEBUG_SYNC here, but that's not possible because the THD's debug
      sync feature has been shut down at this point.
    */
    // DBUG_EXECUTE_IF("sleep_after_lock_thread_count_before_delete_thd", sleep(5););

    const size_t num_erased= thd_map.erase(thd->thread_id());
    if (num_erased == 1)
        --global_thd_count;
    // Removing a THD that was never added is an error.
    DBUG_ASSERT(1 == num_erased);
    mysql_mutex_unlock(&LOCK_thd_remove);
    mysql_cond_broadcast(&COND_thd_list);
    mysql_mutex_unlock(&LOCK_thd_list);
}


my_thread_id Global_THD_manager::get_new_thread_id()
{
    my_thread_id new_id;
    Mutex_lock lock(&LOCK_thread_ids);
    do {
        new_id = thread_id_counter++;
    } while (thread_ids.end() != thread_ids.find(new_id));
    // } while (!thread_ids.insert_unique(new_id).second);
    thread_ids[ new_id ] = new_id;
    return new_id;
}


void Global_THD_manager::release_thread_id(my_thread_id thread_id)
{
    if (thread_id == reserved_thread_id)
        return; // Some temporary THDs are never given a proper ID.
    Mutex_lock lock(&LOCK_thread_ids);
    const size_t num_erased MY_ATTRIBUTE((unused))=
            thread_ids.erase(thread_id);
    ( void )num_erased;
    // Assert if the ID was not found in the list.
    DBUG_ASSERT(1 == num_erased);
}


void Global_THD_manager::set_thread_id_counter(my_thread_id new_id)
{
    // DBUG_ASSERT(unit_test == true);
    Mutex_lock lock(&LOCK_thread_ids);
    thread_id_counter= new_id;
}
THD* Global_THD_manager::find_thd(my_thread_id id)
{
    THD* thd = nullptr;
    auto it = thd_map.find( id );
    if ( thd_map.end() != it)
    {
        thd = it->second;
    }
    return thd;
}
