#include "./include/session_tracker.h"
#include "./include/sql_class.h"

void store_lenenc_string(String &to, const uchar *from, size_t length);
/**
  Current_schema_tracker
  ----------------------
  This is a tracker class that enables & manages the tracking of current
  schema for a particular connection.
*/

class Current_schema_tracker : public State_tracker
{
private:
  bool schema_track_inited;
  void reset();

public:

  /** Constructor */
  Current_schema_tracker()
  {
    schema_track_inited= false;
  }

  // bool enable(THD *thd)
  // { return update(thd); }
  // bool check(THD *thd, set_var *var)
  // { return false; }
  // bool update(THD *thd);
  bool store(THD *thd, String &buf);
  void mark_as_changed(THD *thd, LEX_CSTRING *tracked_item_name);
};

///////////////////////////////////////////////////////////////////////////////

/**
  @brief Enable/disable the tracker based on @@session_track_schema's value.

  @param thd [IN]           The thd handle.

  @return
    false (always)
*/

// bool Current_schema_tracker::update(THD *thd)
// {
//   m_enabled= (thd->variables.session_track_schema)? true: false;
//   return false;
// }


/**
  @brief Store the schema name as length-encoded string in the specified
         buffer.  Once the data is stored, we reset the flags related to
         state-change (see reset()).


  @param thd [IN]           The thd handle.
  @paran buf [INOUT]        Buffer to store the information to.

  @return
    false                   Success
    true                    Error
*/

bool Current_schema_tracker::store(THD *thd, String &buf)
{
  ulonglong db_length, length;

  length= db_length= thd->db().size();
  length += net_length_size(length);

  /* Session state type (SESSION_TRACK_SCHEMA) */
  uchar* lenBuf = new uchar[net_length_size(length) + 1];
  memset(lenBuf, 0, net_length_size(length) + 1);
  uchar* to = net_store_length(lenBuf, (ulonglong)SESSION_TRACK_SCHEMA);

  /* Length of the overall entity. */
  to= net_store_length(to, length);

  buf.append(lenBuf, to - lenBuf);

  delete[] lenBuf;

  /* Current schema name (length-encoded string). */
  store_lenenc_string(buf, (uchar*)thd->db().c_str(), thd->db().size());

  reset();

  return false;
}

/**
  @brief Mark the tracker as changed.

  @param name [IN]          Always null.

  @return void
*/

void Current_schema_tracker::mark_as_changed(THD *thd,
                                             LEX_CSTRING *tracked_item_name)
{
  m_changed= true;
  // thd->lex->safe_to_cache_query= 0;
}


/**
  @brief Reset the m_changed flag for next statement.

  @return                   void
*/

void Current_schema_tracker::reset()
{
  m_changed= false;
}

/**
  Session_sysvars_tracker
  -----------------------
  This is a tracker class that enables & manages the tracking of session
  system variables. It internally maintains a hash of user supplied variable
  names and a boolean field to store if the variable was changed by the last
  statement.
*/

class Session_sysvars_tracker : public State_tracker
{
private:

  struct sysvar_node_st {
    LEX_STRING m_sysvar_name;
    bool m_changed;
  };

  /**
    Two objects of vars_list type are maintained to manage
    various operations on variables_list.
  */
  // vars_list *orig_list, *tool_list;

public:
  /** Constructor */
  Session_sysvars_tracker(/*const CHARSET_INFO *char_set*/)
  {
    // orig_list= new (std::nothrow) vars_list(char_set);
    // tool_list= new (std::nothrow) vars_list(char_set);
  }

  /** Destructor */
  ~Session_sysvars_tracker()
  {
  }

  void reset(){};
  // bool enable(THD *thd) {}
  // bool update(THD *thd) {}
  bool store(THD *thd, String &buf){ return false; }
  void mark_as_changed(THD *thd, LEX_CSTRING *tracked_item_name){};
  /* callback */
};

///////////////////////////////////////////////////////////////////////////////

/**
  @brief Initialize session tracker objects.

  @param char_set [IN]      The character set info.

  @return                   void
*/

void Session_tracker::init(/*const CHARSET_INFO *char_set*/)
{
  m_trackers[SESSION_SYSVARS_TRACKER]=
     new (std::nothrow) Session_sysvars_tracker();
  m_trackers[CURRENT_SCHEMA_TRACKER]=
    new (std::nothrow) Current_schema_tracker;
  // m_trackers[SESSION_STATE_CHANGE_TRACKER]=
  //   new (std::nothrow) Session_state_change_tracker;
  // m_trackers[SESSION_GTIDS_TRACKER]=
  //   new (std::nothrow) Session_gtids_tracker;
  // m_trackers[TRANSACTION_INFO_TRACKER]=
  //   new (std::nothrow) Transaction_state_tracker;
}

/**
  @brief Store all change information in the specified buffer.

  @param thd [IN]           The thd handle.
  @param buf [OUT]          Reference to the string buffer to which the state
                            change data needs to be written.

  @return                   void
*/

void Session_tracker::store(THD *thd, String &buf)
{
  /* Temporary buffer to store all the changes. */
  String temp;
  size_t length;

  /* Get total length. */
  for (int i= 0; i <= SESSION_TRACKER_END; i ++)
  {
    if (NULL != m_trackers[i] && m_trackers[i]->is_changed())
      m_trackers[i]->store(thd, temp);
  }

  length= temp.length();
  /* Store length first.. */
  uchar lenBuff[9];
  memset(lenBuff, 0, 9);
  uchar* pos = net_store_length(lenBuff, length);
  buf.append(lenBuff, pos - lenBuff);

  /* .. and then the actual info. */
  buf.append(temp);
}

/**
  @brief Enables the tracker objects.

  @param thd [IN]    The thread handle.

  @return            void
*/
// void Session_tracker::enable(THD *thd)
// {
//   for (int i= 0; i <= SESSION_TRACKER_END; i ++)
//     if (NULL != m_trackers[i])
//     m_trackers[i]->enable(thd);
// }

/**
  @brief Returns the pointer to the tracker object for the specified tracker.

  @param tracker [IN]       Tracker type.

  @return                   Pointer to the tracker object.
*/

State_tracker *
Session_tracker::get_tracker(enum_session_tracker tracker) const
{
  return m_trackers[tracker];
}

/**
  @brief Checks if m_enabled flag is set for any of the tracker objects.

  @return
    true  - At least one of the trackers is enabled.
    false - None of the trackers is enabled.

*/

bool Session_tracker::enabled_any()
{
  for (int i= 0; i <= SESSION_TRACKER_END; i ++)
  {
    if (m_trackers[i]->is_enabled())
      return true;
  }
  return false;
}

/**
  @brief Checks if m_changed flag is set for any of the tracker objects.

  @return
    true                    At least one of the entities being tracker has
                            changed.
    false                   None of the entities being tracked has changed.
*/

bool Session_tracker::changed_any()
{
  for (int i= 0; i <= SESSION_TRACKER_END; i ++)
  {
    if (m_trackers[i]->is_changed())
      return true;
  }
  return false;
}
