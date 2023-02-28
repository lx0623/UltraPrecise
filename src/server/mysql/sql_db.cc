#include <glog/logging.h>
#include <server/mysql/include/mysqld.h>
#include "server/mysql/include/derror.h"
#include <utils/string_util.h>
#include "./include/sql_class.h"
#include "../../schema/SchemaManager.h"
#include "frontend/SQLExecutor.h"

using namespace aries::schema;
/**
  @brief Change the current database and its attributes unconditionally.

  @param thd          thread handle
  @param new_db_name  database name
  @param force_switch if force_switch is FALSE, then the operation will fail if

                        - new_db_name is NULL or empty;

                        - OR new database name is invalid
                          (check_db_name() failed);

                        - OR user has no privilege on the new database;

                        - OR new database does not exist;

                      if force_switch is TRUE, then

                        - if new_db_name is NULL or empty, the current
                          database will be NULL, @@collation_database will
                          be set to @@collation_server, the operation will
                          succeed.

                        - if new database name is invalid
                          (check_db_name() failed), the current database
                          will be NULL, @@collation_database will be set to
                          @@collation_server, but the operation will fail;

                        - user privileges will not be checked
                          (THD::db_access however is updated);

                          TODO: is this really the intention?
                                (see sp-security.test).

                        - if new database does not exist,the current database
                          will be NULL, @@collation_database will be set to
                          @@collation_server, a warning will be thrown, the
                          operation will succeed.

  @details The function checks that the database name corresponds to a
  valid and existent database, checks access rights and changes the current
  database with database attributes (@@collation_database session variable,
  THD::db_access).

  This function is not the only way to switch the database that is
  currently employed. When the replication slave thread switches the
  database before executing a query, it calls thd->set_db directly.
  However, if the query, in turn, uses a stored routine, the stored routine
  will use this function, even if it's run on the slave.

  This function allocates the name of the database on the system heap: this
  is necessary to be able to uniformly change the database from any module
  of the server. Up to 5.0 different modules were using different memory to
  store the name of the database, and this led to memory corruption:
  a stack pointer set by Stored Procedures was used by replication after
  the stack address was long gone.

  @return Operation status
    @retval FALSE Success
    @retval TRUE  Error
*/
#define TRUE 1
#define FALSE 0

bool mysql_change_db(THD *thd, const string &new_db_name, bool force_switch)
{
    if (new_db_name.empty()) {
        if (force_switch)
        {
            /*
              This can happen only if we're switching the current database back
              after loading stored program. The thing is that loading of stored
              program can happen when there is no current database.

            */

            LOG(INFO) << "Set database to empty\n";
            thd->set_db("");
            goto done;
        }
        else
        {
            my_error(ER_NO_DB_ERROR, MYF(0));

            LOG(ERROR) << "Empty database name\n";
            DBUG_RETURN(TRUE);
        }
    }
    else
    {
        auto schema = aries::schema::SchemaManager::GetInstance()->GetSchema();
        assert (schema != nullptr);

        string lowerDbName = new_db_name;
        aries_utils::to_lower(lowerDbName);
        auto db = schema->GetDatabaseByName(lowerDbName);
        if (db == nullptr) {
            LOG(ERROR) << "cannot find database named: " << new_db_name;
            my_error(ER_BAD_DB_ERROR, MYF(0), new_db_name.c_str());
            DBUG_RETURN(TRUE);
        }

        // if (!myrocks::AriesRdb::GetInstance(lowerDbName)) {
        //     my_error(ER_BAD_DB_ERROR, MYF(0), new_db_name.c_str());
        //     DBUG_RETURN(TRUE);
        // }
        thd->set_db(lowerDbName);
    }
done:
    /*
   Check if current database tracker is enabled. If so, set the 'changed' flag.
  */
    if (thd->session_tracker.get_tracker(CURRENT_SCHEMA_TRACKER)->is_enabled())
    {
        LEX_CSTRING dummy = {C_STRING_WITH_LEN("")};
        thd->session_tracker.get_tracker(CURRENT_SCHEMA_TRACKER)->mark_as_changed(thd, &dummy);
    }
    // if (thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->is_enabled())
    //     thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->mark_as_changed(thd, NULL);
    DBUG_RETURN(FALSE);

}
