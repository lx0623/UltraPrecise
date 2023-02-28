#include <server/mysql/include/mysql_thread.h>
#include <server/mysql/include/sql_prepare.h>
#include <server/mysql/include/mysqld_thd_manager.h>
#include "./include/mysqld.h"
#include "./include/protocol_classic.h"
#include "./include/sql_class.h"
#include "./include/sql_const.h"
#include "utils/string_util.h"

char empty_c_string[1]= {0};    /* used for not defined db */

THD *createPseudoThd() {
    THD *thd = new (std::nothrow) THD;
    if (thd == NULL)
    {
        return NULL;
    }
    if (thd->store_globals())
    {
        delete thd;
        return NULL;
    }

    return thd;
}
/**
  Check the killed state of a user thread
  @param thd  user thread
  @retval 0 the user thread is active
  @retval 1 the user thread has been killed
*/
extern "C" int thd_killed(const THD* thd)
{
    if (thd == NULL)
        return current_thd != NULL ? current_thd->killed : 0;
    return thd->killed;
}

extern bool serverInitDone;
bool IsServerInitDone()
{
    return serverInitDone;
}
bool IsThdKilled( THD* thd ) {
    if ( !serverInitDone )
    {
        return false;
    }
    else
    {
        return thd->is_killed();
    }
}
bool IsCurrentThdKilled() {
    if ( !serverInitDone || !current_thd )
    {
        return false;
    }
    else
    {
        return current_thd->is_killed();
    }
}

void SendKillMessage()
{
    if ( serverInitDone )
    {
        current_thd->send_kill_message();
    }
}

/**
  Return time in microseconds.

  @remark This function is to be used to measure performance in
  micro seconds.

  @retval Number of microseconds since the Epoch, 1970-01-01 00:00:00 +0000 (UTC)
*/

ulonglong my_micro_time()
{
#ifdef _WIN32
    ulonglong newtime;
  my_get_system_time_as_file_time((FILETIME*)&newtime);
  newtime-= OFFSET_TO_EPOCH;
  return (newtime/10);
#else
    ulonglong newtime;
    struct timeval t;
    /*
      The following loop is here because gettimeofday may fail on some systems
    */
    while (gettimeofday(&t, NULL) != 0)
    {}
    newtime= (ulonglong)t.tv_sec * 1000000 + t.tv_usec;
    return newtime;
#endif
}
/**
  Set the state on connection to killed

  @param thd               THD object
*/
void thd_set_killed(THD *thd)
{
    thd->killed= THD::KILL_CONNECTION;
}

bool check_column_name(const char *name)
{
    // name length in symbols
    size_t name_length= 0;
    bool last_char_is_space= 1;

    while (*name)
    {
        last_char_is_space= my_isspace(system_charset_info, *name);
        if (use_mb(system_charset_info))
        {
            int len=my_ismbchar(system_charset_info, name,
                                name+system_charset_info->mbmaxlen);
            if (len)
            {
                name += len;
                name_length++;
                continue;
            }
        }
        if (*name == NAMES_SEP_CHAR)
            return 1;
        name++;
        name_length++;
    }
    /* Error if empty or too long column name */
    return last_char_is_space || (name_length > NAME_CHAR_LEN);
}

/*
  Frees memory used by system variables

  Unlike plugin_vars_free_values() it frees all variables of all plugins,
  it's used on shutdown.
*/
static void cleanup_variables(THD *thd, struct system_variables *vars)
{
    vars->dynamic_variables_ptr= NULL;
    vars->dynamic_variables_size= 0;
    vars->dynamic_variables_version= 0;
}

void plugin_thdvar_init(THD *thd)
{
    cleanup_variables(thd, &thd->variables);

    thd->variables= global_system_variables;

    thd->variables.dynamic_variables_version= 0;
    thd->variables.dynamic_variables_size= 0;
    thd->variables.dynamic_variables_ptr= 0;

    /* Initialize all Sys_var_charptr variables here. */

    // @@session.session_track_system_variables
    // thd->session_sysvar_res_mgr.init(&thd->variables.track_sysvars_ptr, thd->charset());

    DBUG_VOID_RETURN;
}

void plugin_thdvar_cleanup(THD *thd)
{
    DBUG_ENTER("plugin_thdvar_cleanup");

    cleanup_variables(thd, &thd->variables);

    DBUG_VOID_RETURN;
}


extern "C"
void **thd_ha_data(const THD *thd /* , const struct handlerton *hton*/ )
{
    return (void **) &thd->ha_data[/* hton->slot */ 0].ha_ptr;
}
THD::THD()
: peer_port(0),
  is_fatal_error(false),
  m_connection_id(0),
  m_server_idle(false),
  query_id(0),
  active_vio(NULL),
  time_zone_used(false),
  current_mutex(NULL),
  current_cond(NULL),
  m_release_resources_done(false),
  cleanup_done(false),
  main_da(false),
  m_stmt_da(&main_da),
  m_user_name(NULL)
{
  Global_THD_manager *thd_manager= Global_THD_manager::get_instance();
  m_connection_id = thd_manager->get_new_thread_id();

  statement_id_counter= 0UL;
  killed = NOT_KILLED;
  net.vio = 0;
  active_vio = 0;
  set_command(COM_CONNECT);
  *scramble= '\0';

  init();

  // m_SSL = NULL;

  pthread_mutex_init(&LOCK_thd_data, MY_MUTEX_INIT_FAST);
  pthread_mutex_init(&LOCK_current_cond, MY_MUTEX_INIT_FAST);

  /* Protocol */
  m_protocol= &protocol_text;			// Default protocol
  protocol_text.init(this);
  protocol_binary.init(this);
  protocol_text.set_client_capabilities(0); // minimalistic client
}
THD::~THD() {
    if (!m_release_resources_done)
        release_resources();

    /* Ensure that no one is using THD */
    mysql_mutex_lock(&LOCK_thd_data);
    mysql_mutex_unlock(&LOCK_thd_data);

    mysql_mutex_destroy(&LOCK_thd_data);
    mysql_mutex_destroy(&LOCK_current_cond);
}

/**
  Release most resources, prior to THD destruction.
 */
void THD::release_resources() {
    DBUG_ASSERT(m_release_resources_done == false);

    if (m_user_name) {
        free(m_user_name) ;
        m_user_name = NULL;
    }

    Global_THD_manager::get_instance()->release_thread_id(m_connection_id);

    /* Ensure that no one is using THD */
    mysql_mutex_lock(&LOCK_thd_data);

    /* Close connection */
    if (is_classic_protocol() && get_protocol_classic()->get_vio())
    {
        vio_delete(get_protocol_classic()->get_vio());
        get_protocol_classic()->end_net();
    }

    mysql_mutex_unlock(&LOCK_thd_data);

    if (!cleanup_done)
        cleanup();

    plugin_thdvar_cleanup(this);

    if (current_thd == this)
        restore_globals();

    m_release_resources_done = true;
}
/*
  Init common variables that has to be reset on start and on cleanup_connection
*/

void THD::init(void)
{
  // pthread_mutex_lock(&LOCK_global_system_variables);
  plugin_thdvar_init(this);
  /*
    variables= global_system_variables above has reset
    variables.pseudo_thread_id to 0. We need to correct it here to
    avoid temporary tables replication failure.
  */
  // variables.pseudo_thread_id= m_thread_id;
  // pthread_mutex_unlock(&LOCK_global_system_variables);

  /*
    NOTE: reset_connection command will reset the THD to its default state.
    All system variables whose scope is SESSION ONLY should be set to their
    default values here.
  */
  // reset_first_successful_insert_id();
  // user_time.tv_sec= user_time.tv_usec= 0;
  start_time.tv_sec= start_time.tv_usec= 0;
  set_time();
  // auto_inc_intervals_forced.empty();
  // {
  //   ulong tmp;
  //   tmp= sql_rnd_with_mutex();
  //   randominit(&rand, tmp + (ulong) &rand, tmp + (ulong) ::global_query_id);
  // }

  server_status= SERVER_STATUS_AUTOCOMMIT;
  // if (variables.sql_mode & MODE_NO_BACKSLASH_ESCAPES)
  //   server_status|= SERVER_STATUS_NO_BACKSLASH_ESCAPES;

  // get_transaction()->reset_unsafe_rollback_flags(Transaction_ctx::SESSION);
  // get_transaction()->reset_unsafe_rollback_flags(Transaction_ctx::STMT);
  // open_options=ha_open_options;
  // update_lock_default= (variables.low_priority_updates ?
		// 	TL_WRITE_LOW_PRIORITY :
		// 	TL_WRITE);
  // insert_lock_default= (variables.low_priority_updates ?
  //                       TL_WRITE_LOW_PRIORITY :
  //                       TL_WRITE_CONCURRENT_INSERT);
  // tx_isolation= (enum_tx_isolation) variables.tx_isolation;
  // tx_read_only= variables.tx_read_only;
  // tx_priority= 0;
  // thd_tx_priority= 0;
  // update_charset();
  // reset_current_stmt_binlog_format_row();
  // reset_binlog_local_stmt_filter();
  // memset(&status_var, 0, sizeof(status_var));
  // binlog_row_event_extra_data= 0;

  // if (variables.sql_log_bin)
  //   variables.option_bits|= OPTION_BIN_LOG;
  // else
  //   variables.option_bits&= ~OPTION_BIN_LOG;

#if defined(ENABLED_DEBUG_SYNC)
  /* Initialize the Debug Sync Facility. See debug_sync.cc. */
  debug_sync_init_thread(this);
#endif /* defined(ENABLED_DEBUG_SYNC) */

  /* Initialize session_tracker and create all tracker objects */
  session_tracker.init(/*this->charset()*/);
  // session_tracker.enable(this);

  // owned_gtid.clear();
  // owned_sid.clear();
  // owned_gtid.dbug_print(NULL, "set owned_gtid (clear) in THD::init");

  // rpl_thd_ctx.dependency_tracker_ctx().set_last_session_sequence_number(0);
}

/*
  Do what's needed when one invokes change user

  SYNOPSIS
    cleanup_connection()

  IMPLEMENTATION
    Reset all resources that are connection specific
*/
void THD::cleanup_connection(void)
{
    cleanup();
    killed = NOT_KILLED;
    cleanup_done= 0;
    init();

    clear_error();
}

/*
  Do what's needed when one invokes change user.
  Also used during THD::release_resources, i.e. prior to THD destruction.
*/
void THD::cleanup(void)
{
    killed= KILL_CONNECTION;
    /*
    Destroy trackers only after finishing manipulations with transaction
    state to avoid issues with Transaction_state_tracker.
    */
    session_tracker.deinit();

    cleanup_done=1;
    DBUG_VOID_RETURN;
}

/**
  Awake a thread.

  @param[in]  state_to_set    value for THD::killed

  This is normally called from another thread's THD object.

  @note Do always call this while holding LOCK_thd_data.
*/

void THD::awake(THD::killed_state state_to_set)
{
    DBUG_ENTER("THD::awake");
    // DBUG_PRINT("enter", ("this: %p current_thd: %p", this, current_thd));
    // THD_CHECK_SENTRY(this);
    mysql_mutex_assert_owner(&LOCK_thd_data);

    /*
      Set killed flag if the connection is being killed (state_to_set
      is KILL_CONNECTION) or the connection is processing a query
      (state_to_set is KILL_QUERY and m_server_idle flag is not set).
      If the connection is idle and state_to_set is KILL QUERY, the
      the killed flag is not set so that it doesn't affect the next
      command incorrectly.
    */
    if (this->m_server_idle && state_to_set == KILL_QUERY)
    { /* nothing */ }
    else
    {
        killed= state_to_set;
    }

    if (state_to_set != THD::KILL_QUERY && state_to_set != THD::KILL_TIMEOUT)
    {
        if (this != current_thd)
        {
            /*
              Before sending a signal, let's close the socket of the thread
              that is being killed ("this", which is not the current thread).
              This is to make sure it does not block if the signal is lost.
              This needs to be done only on platforms where signals are not
              a reliable interruption mechanism.

              Note that the downside of this mechanism is that we could close
              the connection while "this" target thread is in the middle of
              sending a result to the application, thus violating the client-
              server protocol.

              On the other hand, without closing the socket we have a race
              condition. If "this" target thread passes the check of
              thd->killed, and then the current thread runs through
              THD::awake(), sets the 'killed' flag and completes the
              signaling, and then the target thread runs into read(), it will
              block on the socket. As a result of the discussions around
              Bug#37780, it has been decided that we accept the race
              condition. A second KILL awakes the target from read().

              If we are killing ourselves, we know that we are not blocked.
              We also know that we will check thd->killed before we go for
              reading the next statement.
            */

            shutdown_active_vio();
        }

        /* Send an event to the scheduler that a thread should be killed. */
        // if (!slave_thread)
        //     MYSQL_CALLBACK(Connection_handler_manager::event_functions,
        //                    post_kill_notification, (this));
    }

    /* Interrupt target waiting inside a storage engine. */
    // if (state_to_set != THD::NOT_KILLED)
    //     ha_kill_connection(this);

    // if (state_to_set == THD::KILL_TIMEOUT)
    // {
    //     DBUG_ASSERT(!status_var_aggregated);
    //     status_var.max_execution_time_exceeded++;
    // }


    /* Broadcast a condition to kick the target if it is waiting on it. */
    if (is_killable)
    {
        mysql_mutex_lock(&LOCK_current_cond);
        /*
          This broadcast could be up in the air if the victim thread
          exits the cond in the time between read and broadcast, but that is
          ok since all we want to do is to make the victim thread get out
          of waiting on current_cond.
          If we see a non-zero current_cond: it cannot be an old value (because
          then exit_cond() should have run and it can't because we have mutex); so
          it is the true value but maybe current_mutex is not yet non-zero (we're
          in the middle of enter_cond() and there is a "memory order
          inversion"). So we test the mutex too to not lock 0.

          Note that there is a small chance we fail to kill. If victim has locked
          current_mutex, but hasn't yet entered enter_cond() (which means that
          current_cond and current_mutex are 0), then the victim will not get
          a signal and it may wait "forever" on the cond (until
          we issue a second KILL or the status it's waiting for happens).
          It's true that we have set its thd->killed but it may not
          see it immediately and so may have time to reach the cond_wait().

          However, where possible, we test for killed once again after
          enter_cond(). This should make the signaling as safe as possible.
          However, there is still a small chance of failure on platforms with
          instruction or memory write reordering.
        */
        if (current_cond && current_mutex)
        {
            // DBUG_EXECUTE_IF("before_dump_thread_acquires_current_mutex",
            //                 {
            //                         const char act[]=
            //                         "now signal dump_thread_signal wait_for go_dump_thread";
            //                         DBUG_ASSERT(!debug_sync_set_action(current_thd,
            //                                                            STRING_WITH_LEN(act)));
            //                 };);
            mysql_mutex_lock(current_mutex);
            mysql_cond_broadcast(current_cond);
            mysql_mutex_unlock(current_mutex);
        }
        mysql_mutex_unlock(&LOCK_current_cond);
    }
    DBUG_VOID_RETURN;
}

bool THD::send_result_metadata(vector<Send_field*> *list, uint flags)
{
  // DBUG_ENTER("send_result_metadata");
  vector<Send_field*>::iterator it = list->begin();
  // List_iterator_fast<Item> it(*list);
  Send_field* item;
  // String tmp((char *) buff, sizeof(buff), &my_charset_bin);

  if (m_protocol->start_result_metadata(list->size(), flags
          /*variables.character_set_results*/))
    goto err;

  for (it = list->begin(); it != list->end(); it++)
  {
    item = *it;
    m_protocol->start_row();
    if (m_protocol->send_field_metadata(item, /*item->charset_for_protocol())*/default_charset_info))
      goto err;
    if (flags & Protocol_classic::SEND_DEFAULTS)
    {
      // item->send(m_protocol, &tmp);
        if ( item->has_default )
            m_protocol->store( item->default_val_str.c_str(),
                               item->default_val_str.length(),
                               default_charset_info);
        else
            m_protocol->store( item->default_val_str.c_str(),
                               0,
                               default_charset_info);

    }
    if (m_protocol->end_row())
      DBUG_RETURN(true);
  }

  DBUG_RETURN(m_protocol->end_result_metadata());

err:
  LOG(ERROR) << "metadata error.";
  my_error(ER_OUT_OF_RESOURCES, MYF(0));        /* purecov: inspected */
  DBUG_RETURN(1);                               /* purecov: inspected */
}

void THD::send_statement_status()
{
    // DBUG_ENTER("send_statement_status");
    DBUG_ASSERT(!get_stmt_da()->is_sent());
    bool error= false;
    Diagnostics_area *da= get_stmt_da();

    /* Can not be true, but do not take chances in production. */
    if (da->is_sent())
        DBUG_VOID_RETURN;

    switch (da->status())
    {
        case Diagnostics_area::DA_ERROR:
            /* The query failed, send error to log and abort bootstrap. */
            error= m_protocol->send_error(
                    da->mysql_errno(), da->message_text(), da->returned_sqlstate());
            break;
        case Diagnostics_area::DA_EOF:
            error= m_protocol->send_eof(
                    server_status, /* da->last_statement_cond_count() */0);
            break;
        case Diagnostics_area::DA_OK:
            error= m_protocol->send_ok(
                    server_status, /* da->last_statement_cond_count() */ 0,
                    da->affected_rows(), da->last_insert_id(), da->message_text());
            break;
        case Diagnostics_area::DA_DISABLED:
            break;
        case Diagnostics_area::DA_EMPTY:
        default:
            // DBUG_ASSERT(0);
            error= m_protocol->send_ok(server_status, 0, 0, 0, NULL);
            break;
    }
    if (!error)
        da->set_is_sent(true);
    DBUG_VOID_RETURN;
}

void THD::set_command(enum enum_server_command command)
{
  m_command= command;
#ifdef HAVE_PSI_THREAD_INTERFACE
  PSI_STATEMENT_CALL(set_thread_command)(m_command);
#endif
}

/**
  This is only called from items that is not of type item_field.
*/

// bool THD::send_item(AEDataBufferSPtr column)
// {
//   bool result= false;                       // Will be set if null_value == 0
//   // enum_field_types f_type;
//   switch (column->GetDataType().BaseType)
//   {
//   case AriesBaseDataType::CHAR:
//   {
//     std::string columnData = column->GetString(tid);
//     result = protocol->store(columnData.c_str(), columnData->length(), /* res->charset()*/&my_charset_latin1);
//     break;
//   }
//   case AriesBaseDataType::INT32:
//   {
//     int columnData = column->GetInt(tid);
//     result = protocol->store_short(columnData);
//     break;
//   }
//   case AriesBaseDataType::DOUBLE:
//   {
//     double columnData = column->GetDouble(tid);
//     char buff[100];
//     snprintf(buff, 100, "%f", columnData);
//     cout << column->GetDouble(tid);
//     send(cliFd, buff, strlen(buff), 0);
//     // result = protocol->store(columnData, decimals, buff);
//     break;
//   }
//   default:
//     break;
//   }
// }

// switch ((f_type = field_type()))
// {
// default:
// case MYSQL_TYPE_NULL:
// case MYSQL_TYPE_DECIMAL:
// case MYSQL_TYPE_ENUM:
// case MYSQL_TYPE_SET:
// case MYSQL_TYPE_TINY_BLOB:
// case MYSQL_TYPE_MEDIUM_BLOB:
// case MYSQL_TYPE_LONG_BLOB:
// case MYSQL_TYPE_BLOB:
// case MYSQL_TYPE_GEOMETRY:
// case MYSQL_TYPE_STRING:
// case MYSQL_TYPE_VAR_STRING:
// case MYSQL_TYPE_VARCHAR:
// case MYSQL_TYPE_BIT:
// case MYSQL_TYPE_NEWDECIMAL:
// case MYSQL_TYPE_JSON:
// {
//   String *res;
//   if ((res = val_str(buffer)))
//     result = protocol->store(res->ptr(), res->length(), res->charset());
//   else
//   {
//     DBUG_ASSERT(null_value);
//   }
//   break;
//   }
//   case MYSQL_TYPE_TINY:
//   {
//     longlong nr;
//     nr= val_int();
//     if (!null_value)
//       result= protocol->store_tiny(nr);
//     break;
//   }
//   case MYSQL_TYPE_SHORT:
//   case MYSQL_TYPE_YEAR:
//   {
//     longlong nr;
//     nr= val_int();
//     if (!null_value)
//       result= protocol->store_short(nr);
//     break;
//   }
//   case MYSQL_TYPE_INT24:
//   case MYSQL_TYPE_LONG:
//   {
//     longlong nr;
//     nr= val_int();
//     if (!null_value)
//       result= protocol->store_long(nr);
//     break;
//   }
//   case MYSQL_TYPE_LONGLONG:
//   {
//     longlong nr;
//     nr= val_int();
//     if (!null_value)
//       result= protocol->store_longlong(nr, unsigned_flag);
//     break;
//   }
//   case MYSQL_TYPE_FLOAT:
//   {
//     float nr;
//     nr= (float) val_real();
//     if (!null_value)
//       result= protocol->store(nr, decimals, buffer);
//     break;
//   }
//   case MYSQL_TYPE_DOUBLE:
//   {
//     double nr= val_real();
//     if (!null_value)
//       result= protocol->store(nr, decimals, buffer);
//     break;
//   }
//   case MYSQL_TYPE_DATETIME:
//   case MYSQL_TYPE_DATE:
//   case MYSQL_TYPE_TIMESTAMP:
//   {
//     MYSQL_TIME tm;
//     get_date(&tm, TIME_FUZZY_DATE);
//     if (!null_value)
//       result= (f_type == MYSQL_TYPE_DATE) ? protocol->store_date(&tm) :
//                                             protocol->store(&tm, decimals);
//     break;
//   }
//   case MYSQL_TYPE_TIME:
//   {
//     MYSQL_TIME tm;
//     get_time(&tm);
//     if (!null_value)
//       result= protocol->store_time(&tm, decimals);
//     break;
//   }
//   }
//   if (null_value)
//     result= protocol->store_null();
  // return result;
// }

// bool THD::send_result_set_row(vector<AEDataBufferSPtr> *row_items)
// {
//   // char buffer[MAX_FIELD_WIDTH];
//   // String str_buffer(buffer, sizeof (buffer), &my_charset_bin);
//   // List_iterator_fast<Item> it(*row_items);
//
//   DBUG_ENTER("send_result_set_row");
//
//   vector<AEDataBufferSPtr>::iterator it = row_items->begin();
//   // for (Item *item= it++; item; item= it++)
//   for (; it != row_items.end(); it++)
//   {
//     if (send_item(*it))
//     // if (item->send(m_protocol, &str_buffer) || is_error())
//       DBUG_RETURN(true);
//     /*
//       Reset str_buffer to its original state, as it may have been altered in
//       Item::send().
//     */
//     // str_buffer.set(buffer, sizeof(buffer), &my_charset_bin);
//   }
//   DBUG_RETURN(false);
// }
//
// /* Send data to client. Returns 0 if ok */
//
// bool THD::send_data(vector<Item> &items)
// {
//   // Protocol *protocol= thd->get_protocol();
//   // DBUG_ENTER("Query_result_send::send_data");
//
//   // if (unit->offset_limit_cnt)
//   // {						// using limit offset,count
//   //   unit->offset_limit_cnt--;
//   //   DBUG_RETURN(FALSE);
//   // }
//
//   // /*
//   //   We may be passing the control from mysqld to the client: release the
//   //   InnoDB adaptive hash S-latch to avoid thread deadlocks if it was reserved
//   //   by thd
//   // */
//   // ha_release_temporary_latches(thd);
//
//   // protocol->start_row();
//   if (send_result_set_row(&items))
//   {
//     // protocol->abort_row();
//     DBUG_RETURN(TRUE);
//   }
//
//   // thd->inc_sent_row_count(1);
//   DBUG_RETURN(protocol->end_row());
// }

/**
  Close the Vio associated this session.

  @remark LOCK_thd_data is taken due to the fact that
          the Vio might be disassociated concurrently.
*/

void THD::shutdown_active_vio()
{
  // DBUG_ENTER("shutdown_active_vio");
  // mysql_mutex_assert_owner(&LOCK_thd_data);
#ifndef EMBEDDED_LIBRARY
  if (active_vio)
  {
    vio_shutdown(active_vio);
    active_vio = 0;
    // m_SSL = NULL;
  }
#endif
  DBUG_VOID_RETURN;
}

void THD::disconnect(bool server_shutdown)
{
  Vio *vio= NULL;

  mysql_mutex_lock(&LOCK_thd_data);

  killed= THD::KILL_CONNECTION;

  /*
    Since a active vio might might have not been set yet, in
    any case save a reference to avoid closing a inexistent
    one or closing the vio twice if there is a active one.
  */
  vio= active_vio;
  shutdown_active_vio();

  /* Disconnect even if a active vio is not associated. */
  if (is_classic_protocol() &&
      get_protocol_classic()->get_vio() != vio &&
      get_protocol_classic()->connection_alive())
  {
    m_protocol->shutdown(server_shutdown);
  }

  mysql_mutex_unlock(&LOCK_thd_data);
}

void THD::reset_for_next_command()
{
    // TODO: Why on earth is this here?! We should probably fix this
    // function and move it to the proper file. /Matz
    THD *thd= this;
    /*
      Those two lines below are theoretically unneeded as
      THD::cleanup_after_query() should take care of this already.
    */
    thd->is_fatal_error= thd->time_zone_used= 0;
    /*
      Clear the status flag that are expected to be cleared at the
      beginning of each SQL statement.
    */
    thd->server_status&= ~SERVER_STATUS_CLEAR_SET;

    thd->get_stmt_da()->reset_diagnostics_area();

    DBUG_VOID_RETURN;
}

Sql_condition* THD::raise_condition(uint sql_errno,
                                    const char* sqlstate,
                                    Sql_condition::enum_severity_level level,
                                    const char* msg,
                                    bool use_condition_handler)
{

    DBUG_ASSERT(sql_errno != 0);
    if (sql_errno == 0) /* Safety in release build */
        sql_errno= ER_UNKNOWN_ERROR;
    if (msg == NULL)
        msg= ER(sql_errno);
    if (sqlstate == NULL)
        sqlstate= mysql_errno_to_sqlstate(sql_errno);


    // if (level == Sql_condition::SL_NOTE || level == Sql_condition::SL_WARNING)
    //     got_warning= true;

    // query_cache.abort(&query_cache_tls);

    Diagnostics_area *da= get_stmt_da();
    if (level == Sql_condition::SL_ERROR)
    {
        // is_slave_error= true; // needed to catch query errors during replication

        if (!da->is_error())
        {
            // set_row_count_func(-1);
            da->set_error_status(sql_errno, msg, sqlstate);
        }
    }

    /*
      Avoid pushing a condition for fatal out of memory errors as this will
      require memory allocation and therefore might fail. Non fatal out of
      memory errors can occur if raised by SIGNAL/RESIGNAL statement.
    */
    Sql_condition *cond= NULL;
    // if (!(is_fatal_error && (sql_errno == EE_OUTOFMEMORY ||
    //                          sql_errno == ER_OUTOFMEMORY)))
    // {
    //     cond= da->push_warning(this, sql_errno, sqlstate, level, msg);
    // }
    DBUG_RETURN(cond);
}

/*
  Remember the location of thread info, the structure needed for
  sql_alloc() and the structure for the net buffer
*/

bool THD::store_globals()
{
    if (my_thread_set_THR_THD(this))
        return true;
    /*
      is_killable is concurrently readable by a killer thread.
      It is protected by LOCK_thd_data, it is not needed to lock while the
      value is changing from false not true. If the kill thread reads
      true we need to ensure that the thread doesn't proceed to assign
      another thread to the same TLS reference.
    */
    is_killable= true;
#ifndef DBUG_OFF
    /*
      Let mysqld define the thread id (not mysys)
      This allows us to move THD to different threads if needed.
    */
    // set_my_thread_var_id(m_thread_id);
#endif
    real_id= pthread_self();                      // For debugging

    return false;
}

/*
  Remove the thread specific info (THD and mem_root pointer) stored during
  store_global call for this thread.
*/
void THD::restore_globals()
{
    /* Undocking the thread specific data. */
    my_thread_set_THR_THD(NULL);
}

bool THD::store_user_var(const user_var_entry_ptr& userVarEntryPtr) {
    auto it = m_user_vars.find(userVarEntryPtr->entry_name);
    if (m_user_vars.end() != it) {
        m_user_vars.erase(it);
    }
    m_user_vars.insert(std::make_pair(userVarEntryPtr->entry_name, userVarEntryPtr));
    return true;
}

user_var_entry_ptr THD::get_user_var(const string& name) {
    string lowerName = name;
    lowerName = aries_utils::to_lower(lowerName);
    auto it = m_user_vars.find(lowerName);
    if (m_user_vars.end() == it) {
        return nullptr;
    }
    return it->second;
}
Prepared_statement_map::Prepared_statement_map()
        :m_last_found_statement(NULL)
{
}

std::shared_ptr<user_var_entry> user_var_entry::create(THD *thd, const std::string &name/*, const CHARSET_INFO *cs*/)
{
    if (check_column_name(name.data()))
    {
        my_error(ER_ILLEGAL_USER_VAR, MYF(0), name.data());
        return nullptr;
    }

    std::shared_ptr<user_var_entry> entry = std::make_shared<user_var_entry>();
    string lowerName = name;
    lowerName = aries_utils::to_lower(lowerName);
    entry->init(thd, lowerName/*, cs*/);
    return entry;
}

int Prepared_statement_map::insert(THD *thd, Prepared_statement_ptr statement)
{
    auto pair = st_hash.insert(std::make_pair(statement->id, statement));
    if (!pair.second)
    {
        my_error(ER_OUT_OF_RESOURCES, MYF(0));
        goto err_st_hash;
    }
    if (!statement->name().empty()) {
        auto pair2 = names_hash.insert(std::make_pair(statement->name(), statement));
        if (names_hash.end() == pair2.first) {
            my_error(ER_OUT_OF_RESOURCES, MYF(0));
            goto err_names_hash;
        }
    }
    mysql_mutex_lock(&LOCK_prepared_stmt_count);
    /*
      We don't check that prepared_stmt_count is <= max_prepared_stmt_count
      because we would like to allow to lower the total limit
      of prepared statements below the current count. In that case
      no new statements can be added until prepared_stmt_count drops below
      the limit.
    */
    if (prepared_stmt_count >= max_prepared_stmt_count)
    {
        mysql_mutex_unlock(&LOCK_prepared_stmt_count);
        my_error(ER_MAX_PREPARED_STMT_COUNT_REACHED, MYF(0),
                 max_prepared_stmt_count);
        goto err_max;
    }
    prepared_stmt_count++;
    mysql_mutex_unlock(&LOCK_prepared_stmt_count);

    m_last_found_statement= statement;
    return 0;

err_max:
    if (!statement->name().empty())
        names_hash.erase(statement->name());
err_names_hash:
    st_hash.erase(statement->id);
err_st_hash:
    return 1;
}


Prepared_statement_ptr
Prepared_statement_map::find_by_name(const string &name)
{
    auto it = names_hash.find(name);
    if (names_hash.end() == it) {
        return nullptr;
    } else {
        return it->second;
    }
}

Prepared_statement_ptr Prepared_statement_map::find(ulong id)
{
    if (m_last_found_statement == NULL || id != m_last_found_statement->id)
    {
        auto it = st_hash.find(id);
        if (st_hash.end() == it) {
            return nullptr;
        } else {
            Prepared_statement_ptr stmt= it->second;
            if (stmt && !stmt->name().empty())
                return NULL;
            m_last_found_statement= stmt;
        }
    }
    return m_last_found_statement;
}


void Prepared_statement_map::erase(Prepared_statement_ptr statement)
{
    if (statement == m_last_found_statement)
        m_last_found_statement= NULL;
    if (!statement->name().empty())
        names_hash.erase(statement->name());

    st_hash.erase(statement->id);
    mysql_mutex_lock(&LOCK_prepared_stmt_count);
    DBUG_ASSERT(prepared_stmt_count > 0);
    prepared_stmt_count--;
    mysql_mutex_unlock(&LOCK_prepared_stmt_count);
}


void Prepared_statement_map::reset()
{
    /* Must be first, hash_free will reset st_hash.records */
    if (st_hash.size() > 0)
    {
        mysql_mutex_lock(&LOCK_prepared_stmt_count);
        DBUG_ASSERT(prepared_stmt_count >= st_hash.size());
        prepared_stmt_count-= st_hash.size();
        mysql_mutex_unlock(&LOCK_prepared_stmt_count);
    }
    m_last_found_statement= NULL;
}

aries::BiaodashiPointer user_var_entry::ToBiaodashi() const {
    CommonBiaodashiPtr expression = nullptr;
    auto ariesExpr = getExpr();
    switch (ariesExpr->GetType()) {
        case AriesExprType::INTEGER: {
            if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT64) {
                int value = boost::get<int64_t>(ariesExpr->GetContent());
                expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                expression->SetValueType(BiaodashiValueType::LONG_INT);
            } else if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT16) {
                auto value = boost::get<int16_t>(ariesExpr->GetContent());
                expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                expression->SetValueType(BiaodashiValueType::INT);
            } else if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT8) {
                auto value = boost::get<int8_t>(ariesExpr->GetContent());
                expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                expression->SetValueType(BiaodashiValueType::INT);
            } else {
                int value = boost::get<int>(ariesExpr->GetContent());
                expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, value);
                expression->SetValueType(BiaodashiValueType::INT);
            }
            break;
        }
        case AriesExprType::FLOATING: {
            expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Fudianshu, boost::get<double>(ariesExpr->GetContent()));
            expression->SetValueType(BiaodashiValueType::DOUBLE);
            break;
        }
        case AriesExprType::DECIMAL: {
            auto decimal_content = boost::get<aries_acc::Decimal>(ariesExpr->GetContent());
            char tmp_buffer[64] = {0};
            decimal_content.GetDecimal(tmp_buffer);
            expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Decimal, string( tmp_buffer ));
            expression->SetValueType(BiaodashiValueType::DECIMAL);
            break;
        }
        case AriesExprType::STRING: {
            expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, boost::get<string>(ariesExpr->GetContent()));
            expression->SetValueType(BiaodashiValueType::TEXT);
            break;
        }
        case AriesExprType::TRUE_FALSE: {
            expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, boost::get<bool>(ariesExpr->GetContent()));
            expression->SetValueType(BiaodashiValueType::BOOL);
            break;
        }
        case AriesExprType::NULL_VALUE: {
            expression = std::make_shared<CommonBiaodashi>(BiaodashiType::Null, 0);
            break;
        }
        default: {
            assert(0);
            break;
        }
    }
    return expression;
}
