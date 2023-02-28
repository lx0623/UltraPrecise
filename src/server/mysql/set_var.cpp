//
// Created by tengjp on 19-8-13.
//

#include <unordered_map>
#include <glog/logging.h>
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/set_var.h"
#include "server/mysql/include/mysqld.h"
#include "server/mysql/include/mysql_def.h"
using namespace mysql;

using std::unordered_map;

unordered_map<string, sys_var*> global_variables_hash;
unordered_map<string, sys_var*> session_variables_hash;
extern system_variables global_system_variables;
std::mutex LOCK_global_system_variables; 

sys_var_chain all_sys_vars = { NULL, NULL };

extern THD *createPseudoThd();
bool InsertSysVarRows() {
    THD* thd = createPseudoThd();
    thd->peer_port = USHRT_MAX;
    sys_var *var = all_sys_vars.first;

    std::string sql_global = "insert into `global_variables` values";
    std::string sql_session = "insert into `session_variables` values";
    bool has_global_data = false;
    bool has_session_data = false;
    for (; var; var= var->next) {

        if (var->is_global_var()) {
            std::string value = var->to_string(OPT_GLOBAL);

            if ( has_global_data )
            {
                sql_global += ", ";
            }
            if ( has_session_data )
            {
                sql_session += ", ";
            }
            has_global_data = true;
            has_session_data = true;
            sql_global += "('" + var->name + "', '" + value + "')";
            sql_session += "('" + var->name + "', '" + value + "')";
        } else if (var->is_session_var()) {
            std::string value = var->to_string(OPT_SESSION);
            if ( has_global_data )
            {
                sql_global += ", ";
            }
            if ( has_session_data )
            {
                sql_session += ", ";
            }
            has_global_data = true;
            has_session_data = true;
            sql_global += "('" + var->name + "', '" + value + "')";
            sql_session += "('" + var->name + "', '" + value + "')";

        } else if (var->is_session_only_var()) {
            std::string value = var->to_string(OPT_SESSION);

            if ( has_session_data )
            {
                sql_session += ", ";
            }
            has_session_data = true;
            sql_session += "('" + var->name + "', '" + value + "')";
        }
    }

    SQLExecutor::GetInstance()->ExecuteSQL( sql_global, "information_schema" );
    SQLExecutor::GetInstance()->ExecuteSQL( sql_session, "information_schema" );
    return true;
}

/****************************************************************************
  Main handling of variables:
  - Initialisation
  - Searching during parsing
  - Update loop
****************************************************************************/

/**
  Add variables to the dynamic hash of system variables

  @param first       Pointer to first system variable to add

  @retval
    0           SUCCESS
  @retval
    otherwise   FAILURE
*/


int mysql_add_sys_var_chain(sys_var *first)
{
    sys_var *var;

    /* A write lock should be held on LOCK_system_variables_hash */

    for (var= first; var; var= var->next)
    {
        if (var->is_session_only_var()) {
            session_variables_hash[var->name] = var;
        } else {
            global_variables_hash[var->name] = var;
            session_variables_hash[var->name] = var;
        }
    }

    /* Update system_variable_hash version. */
    // system_variable_hash_version++;
    return 0;
}


int sys_var_init()
{
    /* Must be already initialized. */
    // DBUG_ASSERT(system_charset_info != NULL);

    mysql_add_sys_var_chain(all_sys_vars.first);

    DBUG_RETURN(0);
}

void sys_var_end()
{
    // my_hash_free(&system_variable_hash);

    // for (sys_var *var=all_sys_vars.first; var; var= var->next)
    //     var->cleanup();

    DBUG_VOID_RETURN;
}

/**
  Find a user set-table variable.

  @param str       Name of system variable to find
  @param length    Length of variable.  zero means that we should use strlen()
                   on the variable

  @retval
    pointer     pointer to variable definitions
  @retval
    0           Unknown variable (error message is given)
*/

sys_var *find_sys_var(const char *str)
{
    sys_var *var = NULL;

    auto it = global_variables_hash.find(str);
    if (it != global_variables_hash.end()) {
        var = it->second;
    }

    if (!var) {
        it = session_variables_hash.find(str);
        if (it != session_variables_hash.end()) {
            var = it->second;
        }
    }

    /* Don't show non-visible variables. */
    if (var && var->not_visible())
        return NULL;

    return var;
}

/**
  sys_var constructor

  @param chain     variables are linked into chain for mysql_add_sys_var_chain()
  @param name_arg  the name of the variable. Must be 0-terminated and exist
                   for the liftime of the sys_var object. @sa my_option::name
  @param comment   shown in mysqld --help, @sa my_option::comment
  @param flags_arg or'ed flag_enum values
  @param off       offset of the global variable value from the
                   &global_system_variables.
  @param getopt_id -1 for no command-line option, otherwise @sa my_option::id
  @param getopt_arg_type @sa my_option::arg_type
  @param show_val_type_arg what value_ptr() returns for sql_show.cc
  @param def_val   default value, @sa my_option::def_value
  @param lock      mutex or rw_lock that protects the global variable
                   *in addition* to LOCK_global_system_variables.
  @param binlog_status_enum @sa binlog_status_enum
  @param on_check_func a function to be called at the end of sys_var::check,
                   put your additional checks here
  @param on_update_func a function to be called at the end of sys_var::update,
                   any post-update activity should happen here
  @param substitute If non-NULL, this variable is deprecated and the
  string describes what one should use instead. If an empty string,
  the variable is deprecated but no replacement is offered.
  @param parse_flag either PARSE_EARLY or PARSE_NORMAL
*/
sys_var::sys_var(sys_var_chain *chain, const char *name_arg,
                 const char *comment, int flags_arg, ptrdiff_t off,
                 // int getopt_id, enum get_opt_arg_type getopt_arg_type,
                 SHOW_TYPE show_val_type_arg, longlong def_val,
                 PolyLock *lock, enum binlog_status_enum binlog_status_arg,
                 on_check_function on_check_func,
                 on_update_function on_update_func,
                 const char *substitute, int parse_flag) :
        next(0),
        binlog_status(binlog_status_arg),
        flags(flags_arg), m_parse_flag(parse_flag), show_val_type(show_val_type_arg),
        guard(lock), offset(off), on_check(on_check_func), on_update(on_update_func),
        deprecation_substitute(substitute),
        is_os_charset(FALSE)
{
    /*
      There is a limitation in handle_options() related to short options:
      - either all short options should be declared when parsing in multiple stages,
      - or none should be declared.
      Because a lot of short options are used in the normal parsing phase
      for mysqld, we enforce here that no short option is present
      in the first (PARSE_EARLY) stage.
      See handle_options() for details.
    */
    // DBUG_ASSERT(parse_flag == PARSE_NORMAL || getopt_id <= 0 || getopt_id >= 255);

    name.assign(name_arg);     // ER_NO_DEFAULT relies on 0-termination of name_arg
    DBUG_ASSERT(name.size() <= NAME_CHAR_LEN);

    memset(&option, 0, sizeof(option));
    option.name= name_arg;
    // option.id= getopt_id;
    option.comment= comment;
    // option.arg_type= getopt_arg_type;
    option.value= (uchar **)global_var_ptr();
    option.def_value= def_val;

    if (chain->last)
        chain->last->next= this;
    else
        chain->first= this;
    chain->last= this;
}

bool sys_var::update(THD *thd, const SetSysVarStructure* var)
{
  enum_var_type type= var->m_sysVarStructurePtr->varScope;
  if (type == OPT_GLOBAL || scope() == GLOBAL)
  {
    /*
      Yes, both locks need to be taken before an update, just as
      both are taken to get a value. If we'll take only 'guard' here,
      then value_ptr() for strings won't be safe in SHOW VARIABLES anymore,
      to make it safe we'll need value_ptr_unlock().
    */
    // AutoWLock lock2(guard);
    std::unique_lock< std::mutex > lock( LOCK_global_system_variables );
    return global_update(thd, var) ||
      (on_update && on_update(this, thd, OPT_GLOBAL));
  }
  else
  {
    std::unique_lock< std::mutex > lock( thd->m_lock_thd_sysvar );
  
    bool ret= session_update(thd, var) ||
      (on_update && on_update(this, thd, OPT_SESSION));
  
    /*
      Make sure we don't session-track variables that are not actually
      part of the session. tx_isolation and and tx_read_only for example
      exist as GLOBAL, SESSION, and one-shot ("for next transaction only").
    */
    // if ((var->m_sysVarStructurePtr->varScope == OPT_SESSION) || !is_trilevel())
    // {
    //   if ((!ret) &&
    //       thd->session_tracker.get_tracker(SESSION_SYSVARS_TRACKER)->is_enabled())
    //     thd->session_tracker.get_tracker(SESSION_SYSVARS_TRACKER)->mark_as_changed(thd, &(var->var->name));

    //   if ((!ret) &&
    //       thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->is_enabled())
    //     thd->session_tracker.get_tracker(SESSION_STATE_CHANGE_TRACKER)->mark_as_changed(thd, &var->var->name);
    // }

        return ret;
  }
}

void sys_var::set_default(THD *thd, SetSysVarStructure* var)
{
  if (var->m_sysVarStructurePtr->varScope == OPT_GLOBAL || scope() == GLOBAL)
    global_save_default(thd, var);
  else
    session_save_default(thd, var);

  update(thd, var);
}

uchar *sys_var::session_value_ptr(THD *running_thd, THD *target_thd)
{
    return session_var_ptr(target_thd);
}

uchar *sys_var::global_value_ptr(THD *thd)
{
    return global_var_ptr();
}

uchar *sys_var::session_var_ptr(THD *thd)
{ return ((uchar*)&(thd->variables)) + offset; }

uchar *sys_var::global_var_ptr()
{ return ((uchar*)&global_system_variables) + offset; }

void sys_var::check(THD *thd, SetSysVarStructure* var)
{
  // if ((var->value_expr && do_check(thd, var))
  //     || (on_check && on_check(this, thd, var)))
  // {
  //   if (!thd->is_error())
  //   {
  //     // char buff[STRING_BUFFER_USUAL_SIZE];
  //     // String str(buff, sizeof(buff), system_charset_info), *res;

  //     // if (!var->value_expr)
  //     // {
  //     //   str.set(STRING_WITH_LEN("DEFAULT"), &my_charset_latin1);
  //     //   res= &str;
  //     // }
  //     // else if (!(res=var->value->val_str(&str)))
  //     // {
  //     //   str.set(STRING_WITH_LEN("NULL"), &my_charset_latin1);
  //     //   res= &str;
  //     // }
  //     // ErrConvString err(res);
  //     // my_error(ER_WRONG_VALUE_FOR_VAR, MYF(0), name.str, err.ptr());
  //   }
  //   return true;
  // }
  do_check( thd, var );
}

uchar *sys_var::value_ptr(THD *running_thd, THD *target_thd, enum_var_type type)
{
    if (type == OPT_GLOBAL || scope() == GLOBAL)
    {
        AutoRLock lock(guard);
        return global_value_ptr(running_thd);
    }
    else
        return session_value_ptr(running_thd, target_thd);
}

uchar *sys_var::value_ptr(THD *thd, enum_var_type type)
{
    return value_ptr(thd, thd, type);
}
template <typename T, typename R = T>
R get_sys_var_value(sys_var* var, enum_var_type type);
std::string sys_var::to_string(enum_var_type scope) {
    std::string str;
    switch (show_type()) {
        case SHOW_BOOL: {
            bool value = get_sys_var_value<bool>(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_MY_BOOL: {
            my_bool value = get_sys_var_value<my_bool >(this, scope);
            str = value == 1 ? "true" : "false";
            break;
        }
        case SHOW_INT: {
            int32_t value = get_sys_var_value<int32_t >(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_LONG: {
            ulong value = get_sys_var_value<ulong>(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_HA_ROWS:
        case SHOW_SIGNED_LONG: {
            long value = get_sys_var_value<long>(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_LONGLONG: {
            ulonglong value = get_sys_var_value<ulonglong >(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_DOUBLE: {
            double value = get_sys_var_value<double>(this, scope);
            str = std::to_string(value);
            break;
        }
        case SHOW_CHAR: {
            char* value = (char *) get_sys_var_value<uchar *>(this, scope);
            if (value) {
                str.assign(value);
            }
            break;
        }
        case SHOW_CHAR_PTR: {
            char* value = (char *) get_sys_var_value<uchar **, uchar *>(this, scope);
            if (value) {
                str.assign(value);
            }
            break;
        }
        default:
            ARIES_EXCEPTION( ER_UNKNOWN_SYSTEM_VARIABLE, name.data() );
            break;
    }
    return str;
}

// int set_var::check(THD *thd)
// {
//   // if ((type == OPT_GLOBAL && check_global_access(thd, SUPER_ACL)))
//   //   DBUG_RETURN(1);
//   /* value is a NULL pointer if we are using SET ... = DEFAULT */
//   if ( !value_expr )
//     DBUG_RETURN(0);
// 
//   int ret= var->check(thd, this) ? -1 : 0;
// 
//   return ret;
// }