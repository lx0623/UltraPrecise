//
// Created by tengjp on 19-8-13.
//

#ifndef AIRES_SYS_VARS_H
#define AIRES_SYS_VARS_H

#include <utils/string_util.h>
#include "server/mysql/include/typelib.h"
#include "server/mysql/include/my_getopt.h"
#include "server/mysql/include/sql_class.h"
#include "server/mysql/include/set_var.h"
#include "mysqld.h"

#include "frontend/VariableStructure.h"

extern sys_var_chain all_sys_vars;
extern system_variables max_system_variables;

static const char *bool_values[3]= {"OFF", "ON", 0};
enum charset_enum {IN_SYSTEM_CHARSET, IN_FS_CHARSET};

template <typename T, typename R = T>
R get_sys_var_value(sys_var* var, enum_var_type type);

template <typename T, typename R>
R get_sys_var_value(const char* varName, enum_var_type scope );

/*
  a set of mostly trivial (as in f(X)=X) defines below to make system variable
  declarations more readable
*/
#define VALID_RANGE(X,Y) X,Y
// #define DEFAULT(X) X
#define BLOCK_SIZE(X) X
#define GLOBAL_VAR(X) sys_var::GLOBAL, (((char*)&(X))-(char*)&global_system_variables), sizeof(X)
#define SESSION_VAR(X) sys_var::SESSION, offsetof(SV, X), sizeof(((SV *)0)->X)
#define SESSION_ONLY(X) sys_var::ONLY_SESSION, offsetof(SV, X), sizeof(((SV *)0)->X)

/*
  the define below means that there's no *second* mutex guard,
  LOCK_global_system_variables always guards all system variables
*/
#define NO_MUTEX_GUARD ((PolyLock*)0)
#define IN_BINLOG sys_var::SESSION_VARIABLE_IN_BINLOG
#define NOT_IN_BINLOG sys_var::VARIABLE_NOT_IN_BINLOG
#define ON_READ(X) X
#define ON_CHECK(X) X
#define ON_UPDATE(X) X
#define READ_ONLY sys_var::READONLY+
#define NOT_VISIBLE sys_var::INVISIBLE+
#define UNTRACKED_DEFAULT sys_var::TRI_LEVEL+
// this means that Sys_var_charptr initial value was malloc()ed
#define PREALLOCATED sys_var::ALLOCATED+
/*
  Sys_var_bit meaning is reversed, like in
  @@foreign_key_checks <-> OPTION_NO_FOREIGN_KEY_CHECKS
*/
// #define REVERSE(X) ~(X)
#define DEPRECATED(X) X

#define session_var(THD, TYPE) (*(TYPE*)session_var_ptr(THD))
#define global_var(TYPE) (*(TYPE*)global_var_ptr())

/**
  Sys_var_integer template is used to generate Sys_var_* classes
  for variables that represent the value as a signed or unsigned integer.
  They are Sys_var_uint, Sys_var_ulong, Sys_var_harows, Sys_var_ulonglong,
  and Sys_var_long.

  An integer variable has a minimal and maximal values, and a "block_size"
  (any valid value of the variable must be divisible by the block_size).

  Class specific constructor arguments: min, max, block_size
  Backing store: uint, ulong, ha_rows, ulonglong, long, depending on the
  Sys_var_*
*/
template
        <typename T, ulong ARGT, enum enum_mysql_show_type SHOWT, bool SIGNED>
class Sys_var_integer: public sys_var
{
public:
    Sys_var_integer(const char *name_arg,
                    const char *comment, int flag_args, ptrdiff_t off, size_t size,
                    // CMD_LINE getopt,
                    T min_val, T max_val, T def_val, uint block_size, PolyLock *lock=0,
                    enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
                    on_check_function on_check_func=0,
                    on_update_function on_update_func=0,
                    const char *substitute=0,
                    int parse_flag= PARSE_NORMAL)
            : sys_var(&all_sys_vars, name_arg, comment, flag_args, off, //getopt.id,
                      // getopt.arg_type,
                      SHOWT, def_val, lock, binlog_status_arg,
                      on_check_func, on_update_func,
                      substitute, parse_flag)
    {
        option.var_type= ARGT;
        option.min_value= min_val;
        option.max_value= max_val;
        option.block_size= block_size;
        option.u_max_value= (uchar**)max_var_ptr();
        if (max_var_ptr())
            *max_var_ptr()= max_val;

        // Do not set global_var for Sys_var_keycache objects
        if (offset >= 0)
            global_var(T)= def_val;

        DBUG_ASSERT(size == sizeof(T));
        DBUG_ASSERT(min_val < max_val);
        DBUG_ASSERT(min_val <= def_val);
        DBUG_ASSERT(max_val >= def_val);
        DBUG_ASSERT(block_size > 0);
        DBUG_ASSERT(def_val % block_size == 0);
    }
    void do_check(THD *thd, SetSysVarStructure* var)
    {
        // my_bool fixed= FALSE;
        // longlong v;
        // ulonglong uv;

        // v= var->value->val_int();
        // if (SIGNED) /* target variable has signed type */
        // {
        //     if (var->value->unsigned_flag)
        //     {
        //         /*
        //           Input value is such a large positive number that MySQL used an
        //           unsigned item to hold it. When cast to a signed longlong, if the
        //           result is negative there is "cycling" and this is incorrect (large
        //           positive input value should not end up as a large negative value in
        //           the session signed variable to be set); instead, we need to pick the
        //           allowed number closest to the positive input value, i.e. pick the
        //           biggest allowed positive integer.
        //         */
        //         if (v < 0)
        //             uv= max_of_int_range(ARGT);
        //         else /* no cycling, longlong can hold true value */
        //             uv= (ulonglong) v;
        //     }
        //     else
        //         uv= v;
        //     /* This will further restrict with VALID_RANGE, BLOCK_SIZE */
        //     var->save_result.ulonglong_value=
        //             getopt_ll_limit_value(uv, &option, &fixed);
        // }
        // else
        // {
        //     if (var->value->unsigned_flag)
        //     {
        //         /* Guaranteed positive input value, ulonglong can hold it */
        //         uv= (ulonglong) v;
        //     }
        //     else
        //     {
        //         /*
        //           Maybe negative input value; in this case, cast to ulonglong makes it
        //           positive, which is wrong. Pick the closest allowed value i.e. 0.
        //         */
        //         uv= (ulonglong) (v < 0 ? 0 : v);
        //     }
        //     var->save_result.ulonglong_value=
        //             getopt_ull_limit_value(uv, &option, &fixed);
        // }

        // if (max_var_ptr())
        // {
        //     /* check constraint set with --maximum-...=X */
        //     if (SIGNED)
        //     {
        //         longlong max_val= *max_var_ptr();
        //         if (((longlong)(var->save_result.ulonglong_value)) > max_val)
        //             var->save_result.ulonglong_value= max_val;
        //         /*
        //           Signed variable probably has some kind of symmetry. Then it's good
        //           to limit negative values just as we limit positive values.
        //         */
        //         max_val= -max_val;
        //         if (((longlong)(var->save_result.ulonglong_value)) < max_val)
        //             var->save_result.ulonglong_value= max_val;
        //     }
        //     else
        //     {
        //         ulonglong max_val= *max_var_ptr();
        //         if (var->save_result.ulonglong_value > max_val)
        //             var->save_result.ulonglong_value= max_val;
        //     }
        // }

        // return throw_bounds_warning(thd, name.str,
        //                             var->save_result.ulonglong_value !=
        //                             (ulonglong)v,
        //                             var->value->unsigned_flag, v);
    }
    void session_save_default(THD *thd, SetSysVarStructure* var)
    {

    }
    void global_save_default(THD *thd, SetSysVarStructure* var)
    {

    }
    bool session_update(THD *thd, const SetSysVarStructure* var)
    {
      return true;
    }
    bool global_update(THD *thd, const SetSysVarStructure* var)
    {
      return true;
    }
    // bool session_update(THD *thd, set_var *var)
    // {
    //     session_var(thd, T)= static_cast<T>(var->save_result.ulonglong_value);
    //     return false;
    // }
    // bool global_update(THD *thd, set_var *var)
    // {
    //     global_var(T)= static_cast<T>(var->save_result.ulonglong_value);
    //     return false;
    // }
    bool check_update_type( BiaodashiValueType value_type )
    {
        return !schema::ColumnEntry::IsIntegerType( value_type );
    }
    // void session_save_default(THD *thd, set_var *var)
    // {
    //     var->save_result.ulonglong_value=
    //             static_cast<ulonglong>(*(T*)global_value_ptr(thd, 0));
    // }
    // void global_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= option.def_value; }
private:
    T *max_var_ptr()
    {
        return scope() == SESSION ? (T*)(((uchar*)&max_system_variables) + offset)
                                  : 0;
    }
};

typedef Sys_var_integer<int32, GET_UINT, SHOW_INT, 0> Sys_var_int32;
typedef Sys_var_integer<uint, GET_UINT, SHOW_INT, 0> Sys_var_uint;
typedef Sys_var_integer<ulong, GET_ULONG, SHOW_LONG, 0> Sys_var_ulong;
// typedef Sys_var_integer<ha_rows, GET_HA_ROWS, SHOW_HA_ROWS, FALSE>
//         Sys_var_harows;
typedef Sys_var_integer<ulonglong, GET_ULL, SHOW_LONGLONG, 0>
        Sys_var_ulonglong;
typedef Sys_var_integer<long, GET_LONG, SHOW_SIGNED_LONG, 1> Sys_var_long;

/**
  Helper class for variables that take values from a TYPELIB
*/
class Sys_var_typelib: public sys_var
{
protected:
    TYPELIB typelib;
public:
    Sys_var_typelib(const char *name_arg,
                    const char *comment, int flag_args, ptrdiff_t off,
                    // CMD_LINE getopt,
                    SHOW_TYPE show_val_type_arg, const char *values[],
                    ulonglong def_val, PolyLock *lock,
                    enum binlog_status_enum binlog_status_arg,
                    on_check_function on_check_func, on_update_function on_update_func,
                    const char *substitute, int parse_flag= PARSE_NORMAL)
            : sys_var(&all_sys_vars, name_arg, comment, flag_args, off, // getopt.id,
                      // getopt.arg_type,
                      show_val_type_arg, def_val, lock,
                      binlog_status_arg, on_check_func,
                      on_update_func, substitute, parse_flag)
    {
        for (typelib.count= 0; values[typelib.count]; typelib.count++) /*no-op */;
        typelib.name="";
        typelib.type_names= values;
        typelib.type_lengths= 0;    // only used by Fields_enum and Field_set
        option.typelib= &typelib;
    }
    void do_check(THD *thd, SetSysVarStructure* var); // works for enums and my_bool
    bool check_update_type( BiaodashiValueType value_type );

};

/**
  The class for ENUM variables - variables that take one value from a fixed
  list of values.

  Class specific constructor arguments:
    char* values[]    - 0-terminated list of strings of valid values

  Backing store: uint

  @note
  Do *not* use "enum FOO" variables as a backing store, there is no
  guarantee that sizeof(enum FOO) == sizeof(uint), there is no guarantee
  even that sizeof(enum FOO) == sizeof(enum BAR)
*/
class Sys_var_enum: public Sys_var_typelib
{
public:
    Sys_var_enum(const char *name_arg,
                 const char *comment, int flag_args, ptrdiff_t off, size_t size,
                 // CMD_LINE getopt,
                 const char *values[], uint def_val, PolyLock *lock=0,
                 enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
                 on_check_function on_check_func=0,
                 on_update_function on_update_func=0,
                 const char *substitute=0)
            : Sys_var_typelib(name_arg, comment, flag_args, off, // getopt,
                              SHOW_CHAR, values, def_val, lock,
                              binlog_status_arg, on_check_func, on_update_func,
                              substitute)
    {
        option.var_type= GET_ENUM;
        global_var(ulong)= def_val;
        DBUG_ASSERT(def_val < typelib.count);
        DBUG_ASSERT(size == sizeof(ulong));
    }
    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    // bool session_update(THD *thd, set_var *var)
    // {
    //     session_var(thd, ulong)=
    //             static_cast<ulong>(var->save_result.ulonglong_value);
    //     return false;
    // }
    // bool global_update(THD *thd, set_var *var)
    // {
    //     global_var(ulong)=
    //             static_cast<ulong>(var->save_result.ulonglong_value);
    //     return false;
    // }
    // void session_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= global_var(ulong); }
    // void global_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= option.def_value; }
    uchar *session_value_ptr(THD *running_thd, THD *target_thd)
    { return (uchar*)typelib.type_names[session_var(target_thd, ulong)]; }
    uchar *global_value_ptr(THD *thd)
    { return (uchar*)typelib.type_names[global_var(ulong)]; }
};


/**
  The class for boolean variables - a variant of ENUM variables
  with the fixed list of values of { OFF , ON }

  Backing store: my_bool
*/
class Sys_var_mybool: public Sys_var_typelib
{
public:
    Sys_var_mybool(const char *name_arg,
                   const char *comment, int flag_args, ptrdiff_t off, size_t size,
                   // CMD_LINE getopt,
                   my_bool def_val, PolyLock *lock=0,
                   enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
                   on_check_function on_check_func=0,
                   on_update_function on_update_func=0,
                   const char *substitute=0,
                   int parse_flag= PARSE_NORMAL)
            : Sys_var_typelib(name_arg, comment, flag_args, off, // getopt,
                              SHOW_MY_BOOL,
                              bool_values, def_val, lock,
                              binlog_status_arg, on_check_func, on_update_func,
                              substitute, parse_flag)
    {
        option.var_type= GET_BOOL;
        global_var(my_bool)= def_val;
        DBUG_ASSERT(def_val < 2);
        // DBUG_ASSERT(getopt.arg_type == OPT_ARG || getopt.id == -1);
        DBUG_ASSERT(size == sizeof(my_bool));
    }
    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    // bool session_update(THD *thd, set_var *var)
    // {
    //     session_var(thd, my_bool)=
    //             static_cast<my_bool>(var->save_result.ulonglong_value);
    //     return false;
    // }
    // bool global_update(THD *thd, set_var *var)
    // {
    //     global_var(my_bool)=
    //             static_cast<my_bool>(var->save_result.ulonglong_value);
    //     return false;
    // }
    // void session_save_default(THD *thd, set_var *var)
    // {
    //     var->save_result.ulonglong_value=
    //             static_cast<ulonglong>(*(my_bool *)global_value_ptr(thd, 0));
    // }
    // void global_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= option.def_value; }
};

/**
  Generic class for variables for storing entities that are internally
  represented as structures, have names, and possibly can be referred to by
  numbers.  Examples: character sets, collations, locales,

  Class specific constructor arguments:
    ptrdiff_t name_offset  - offset of the 'name' field in the structure

  Backing store: void*

  @note
  As every such a structure requires special treatment from my_getopt,
  these variables don't support command-line equivalents, any such
  command-line options should be added manually to my_long_options in mysqld.cc
*/
class Sys_var_struct: public sys_var
{
    ptrdiff_t name_offset; // offset to the 'name' property in the structure
public:
    Sys_var_struct(const char *name_arg,
                   const char *comment, int flag_args, ptrdiff_t off, size_t size,
                   // CMD_LINE getopt,
                   ptrdiff_t name_off, void *def_val, PolyLock *lock=0,
                   enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
                   on_check_function on_check_func=0,
                   on_update_function on_update_func=0,
                   const char *substitute=0,
                   int parse_flag= PARSE_NORMAL)
            : sys_var(&all_sys_vars, name_arg, comment, flag_args, off, //getopt.id,
                      // getopt.arg_type,
                      SHOW_CHAR, (intptr)def_val,
                      lock, binlog_status_arg, on_check_func, on_update_func,
                      substitute, parse_flag),
              name_offset(name_off)
    {
        option.var_type= GET_STR;
        /*
          struct variables are special on the command line - often (e.g. for
          charsets) the name cannot be immediately resolved, but only after all
          options (in particular, basedir) are parsed.

          thus all struct command-line options should be added manually
          to my_long_options in mysqld.cc
        */
        // DBUG_ASSERT(getopt.id == -1);
        DBUG_ASSERT(size == sizeof(void *));
    }
    // bool do_check(THD *thd, set_var *var)
    // { return false; }
    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    // bool session_update(THD *thd, set_var *var)
    // {
    //     session_var(thd, const void*)= var->save_result.ptr;
    //     return false;
    // }
    // bool global_update(THD *thd, set_var *var)
    // {
    //     global_var(const void*)= var->save_result.ptr;
    //     return false;
    // }
    // void session_save_default(THD *thd, set_var *var)
    // { var->save_result.ptr= global_var(void*); }
    // void global_save_default(THD *thd, set_var *var)
    // {
    //     void **default_value= reinterpret_cast<void**>(option.def_value);
    //     var->save_result.ptr= *default_value;
    // }
    uchar *session_value_ptr(THD *running_thd, THD *target_thd)
    {
        uchar *ptr= session_var(target_thd, uchar*);
        return ptr ? *(uchar**)(ptr+name_offset) : 0;
    }
    uchar *global_value_ptr(THD *thd)
    {
        uchar *ptr= global_var(uchar*);
        return ptr ? *(uchar**)(ptr+name_offset) : 0;
    }
    bool check_update_type( BiaodashiValueType value_type );
    void do_check(THD *thd, SetSysVarStructure* var)
    {
    }
};

/**
  The class for string variables. The string can be in character_set_filesystem
  or in character_set_system. The string can be allocated with my_malloc()
  or not. The state of the initial value is specified in the constructor,
  after that it's managed automatically. The value of NULL is supported.

  Class specific constructor arguments:
    enum charset_enum is_os_charset_arg

  Backing store: char*

*/
class Sys_var_charptr: public sys_var
{
public:
    Sys_var_charptr(const char *name_arg,
                    const char *comment, int flag_args, ptrdiff_t off, size_t size,
                    // CMD_LINE getopt,
                    enum charset_enum is_os_charset_arg,
                    const char *def_val, PolyLock *lock= 0,
                    enum binlog_status_enum binlog_status_arg= VARIABLE_NOT_IN_BINLOG,
                    on_check_function on_check_func= 0,
                    on_update_function on_update_func= 0,
                    const char *substitute= 0,
                    int parse_flag= PARSE_NORMAL)
            : sys_var(&all_sys_vars, name_arg, comment, flag_args, off, // getopt.id,
                      // getopt.arg_type,
                      SHOW_CHAR_PTR, (intptr) def_val,
                      lock, binlog_status_arg, on_check_func, on_update_func,
                      substitute, parse_flag)
    {
        is_os_charset= is_os_charset_arg == IN_FS_CHARSET;
        option.var_type= (flags & ALLOCATED) ? GET_STR_ALLOC : GET_STR;
        global_var(const char*)= def_val;
        DBUG_ASSERT(size == sizeof(char *));
    }

    void cleanup()
    {
        if (flags & ALLOCATED)
            free(global_var(char*));
        flags&= ~ALLOCATED;
    }

    void do_check(THD *thd, SetSysVarStructure* var);
    // {
    //     char buff[STRING_BUFFER_USUAL_SIZE], buff2[STRING_BUFFER_USUAL_SIZE];
    //     String str(buff, sizeof(buff), charset(thd));
    //     String str2(buff2, sizeof(buff2), charset(thd)), *res;

    //     if (!(res=var->value->val_str(&str)))
    //         var->save_result.string_value.str= 0;
    //     else
    //     {
    //         size_t unused;
    //         if (String::needs_conversion(res->length(), res->charset(),
    //                                      charset(thd), &unused))
    //         {
    //             uint errors;
    //             str2.copy(res->ptr(), res->length(), res->charset(), charset(thd),
    //                       &errors);
    //             res= &str2;

    //         }
    //         var->save_result.string_value.str= thd->strmake(res->ptr(), res->length());
    //         var->save_result.string_value.length= res->length();
    //     }

    //     return false;
    // }

    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    // bool session_update(THD *thd, set_var *var)
    // {
    //     char *new_val=  var->save_result.string_value.str;
    //     size_t new_val_len= var->save_result.string_value.length;
    //     char *ptr= ((char *)&thd->variables + offset);

    //     return thd->session_sysvar_res_mgr.update((char **) ptr, new_val,
    //                                               new_val_len);
    // }

    // bool global_update(THD *thd, set_var *var)
    // {
    //     char *new_val, *ptr= var->save_result.string_value.str;
    //     size_t len=var->save_result.string_value.length;
    //     if (ptr)
    //     {
    //         new_val= (char*) my_memdup(key_memory_Sys_var_charptr_value,
    //                                    ptr, len+1, MYF(MY_WME));
    //         if (!new_val) return true;
    //         new_val[len]= 0;
    //     }
    //     else
    //         new_val= 0;
    //     if (flags & ALLOCATED)
    //         my_free(global_var(char*));
    //     flags |= ALLOCATED;
    //     global_var(char*)= new_val;
    //     return false;
    // }

    // void session_save_default(THD *thd, set_var *var)
    // {
    //     char *ptr= (char*)(intptr)option.def_value;
    //     var->save_result.string_value.str= ptr;
    //     var->save_result.string_value.length= ptr ? strlen(ptr) : 0;
    // }

    // void global_save_default(THD *thd, set_var *var)
    // {
    //     char *ptr= (char*)(intptr)option.def_value;
    //     var->save_result.string_value.str= ptr;
    //     var->save_result.string_value.length= ptr ? strlen(ptr) : 0;
    // }
    bool check_update_type( BiaodashiValueType value_type );
};

class Sys_var_version : public Sys_var_charptr
{
public:
    Sys_var_version(const char *name_arg,
                    const char *comment, int flag_args, ptrdiff_t off, size_t size,
                    // CMD_LINE getopt,
                    enum charset_enum is_os_charset_arg,
                    const char *def_val)
            : Sys_var_charptr(name_arg, comment, flag_args, off, size, /*getopt,*/ is_os_charset_arg, def_val)
    {}

    ~Sys_var_version()
    {}

    virtual uchar *global_value_ptr(THD *thd)
    {
        uchar *value= Sys_var_charptr::global_value_ptr(thd);

        // DBUG_EXECUTE_IF("alter_server_version_str",
        //                 {
        //                         static const char *altered_value= "some-other-version";
        //                         uchar *altered_value_ptr= reinterpret_cast<uchar*> (& altered_value);
        //                         value= altered_value_ptr;
        //                 });

        return value;
    }
};

// overflow-safe (1 << X)-1
#define MAX_SET(X) ((((1UL << ((X)-1))-1) << 1) | 1)
/**
  The class for SET variables - variables taking zero or more values
  from the given list. Example: @@sql_mode

  Class specific constructor arguments:
    char* values[]    - 0-terminated list of strings of valid values

  Backing store: ulonglong
*/
class Sys_var_set: public Sys_var_typelib
{
public:
    Sys_var_set(const char *name_arg,
                const char *comment, int flag_args, ptrdiff_t off, size_t size,
                // CMD_LINE getopt,
                const char *values[], ulonglong def_val, PolyLock *lock=0,
                enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
                on_check_function on_check_func=0,
                on_update_function on_update_func=0,
                const char *substitute=0)
            : Sys_var_typelib(name_arg, comment, flag_args, off, // getopt,
                              SHOW_CHAR, values, def_val, lock,
                              binlog_status_arg, on_check_func, on_update_func,
                              substitute)
    {
        option.var_type= GET_SET;
        global_var(ulonglong)= def_val;
        DBUG_ASSERT(typelib.count > 0);
        DBUG_ASSERT(typelib.count <= 64);
        DBUG_ASSERT(def_val < MAX_SET(typelib.count));
        DBUG_ASSERT(size == sizeof(ulonglong));
    }
    void do_check(THD *thd, SetSysVarStructure* var);
    // {
    //     char buff[STRING_BUFFER_USUAL_SIZE];
    //     String str(buff, sizeof(buff), system_charset_info), *res;

    //     if (var->value->result_type() == STRING_RESULT)
    //     {
    //         if (!(res=var->value->val_str(&str)))
    //             return true;
    //         else
    //         {
    //             char *error;
    //             uint error_len;
    //             bool not_used;

    //             var->save_result.ulonglong_value=
    //                     find_set(&typelib, res->ptr(),
    //                              static_cast<uint>(res->length()), NULL,
    //                              &error, &error_len, &not_used);
    //             /*
    //               note, we only issue an error if error_len > 0.
    //               That is even while empty (zero-length) values are considered
    //               errors by find_set(), these errors are ignored here
    //             */
    //             if (error_len)
    //             {
    //                 ErrConvString err(error, error_len, res->charset());
    //                 my_error(ER_WRONG_VALUE_FOR_VAR, MYF(0), name.str, err.ptr());
    //                 return true;
    //             }
    //         }
    //     }
    //     else
    //     {
    //         longlong tmp=var->value->val_int();
    //         if ((tmp < 0 && ! var->value->unsigned_flag)
    //             || (ulonglong)tmp > MAX_SET(typelib.count))
    //             return true;
    //         else
    //             var->save_result.ulonglong_value= tmp;
    //     }

    //     return false;
    // }
    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    // bool session_update(THD *thd, set_var *var)
    // {
    //     session_var(thd, ulonglong)= var->save_result.ulonglong_value;
    //     return false;
    // }
    // bool global_update(THD *thd, set_var *var)
    // {
    //     global_var(ulonglong)= var->save_result.ulonglong_value;
    //     return false;
    // }
    // void session_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= global_var(ulonglong); }
    // void global_save_default(THD *thd, set_var *var)
    // { var->save_result.ulonglong_value= option.def_value; }
    uchar *session_value_ptr(THD *running_thd, THD *target_thd, LEX_STRING *base)
    {
        return (uchar*)aries_utils::set_to_string(running_thd, &value, session_var(target_thd, ulonglong),
                                                  typelib.type_names);
    }
    uchar *global_value_ptr(THD *thd, LEX_STRING *base)
    {
        return (uchar*)aries_utils::set_to_string(thd, &value, global_var(ulonglong),
                                                  typelib.type_names);
    }

private:
    string value;
};
/**
  The class for variables that store time zones

  Backing store: Time_zone*

  @note
  Time zones cannot be supported directly by my_getopt, thus
  these variables don't support command-line equivalents, any such
  command-line options should be added manually to my_long_options in mysqld.cc
*/
class Sys_var_tz: public sys_var
{
public:
    Sys_var_tz(const char *name_arg,
               const char *comment, int flag_args, ptrdiff_t off, size_t size,
               // CMD_LINE getopt,
               Time_zone **def_val, PolyLock *lock=0,
               enum binlog_status_enum binlog_status_arg=VARIABLE_NOT_IN_BINLOG,
               on_check_function on_check_func=0,
               on_update_function on_update_func=0,
               const char *substitute=0,
               int parse_flag= PARSE_NORMAL)
            : sys_var(&all_sys_vars, name_arg, comment, flag_args, off,
                      //getopt.id, getopt.arg_type,
                      SHOW_CHAR, (intptr)def_val,
                      lock, binlog_status_arg, on_check_func, on_update_func,
                      substitute, parse_flag)
    {
        // DBUG_ASSERT(getopt.id == -1);
        DBUG_ASSERT(size == sizeof(Time_zone *));
    }
    void session_save_default(THD *thd, SetSysVarStructure* var);
    void global_save_default(THD *thd, SetSysVarStructure* var);
    bool session_update(THD *thd, const SetSysVarStructure* var);
    bool global_update(THD *thd, const SetSysVarStructure* var);
    uchar *session_value_ptr(THD *running_thd, THD *target_thd)
    {
        /*
          This is an ugly fix for replication: we don't replicate properly queries
          invoking system variables' values to update tables; but
          CONVERT_TZ(,,@@session.time_zone) is so popular that we make it
          replicable (i.e. we tell the binlog code to store the session
          timezone). If it's the global value which was used we can't replicate
          (binlog code stores session value only).
        */
        target_thd->time_zone_used= 1;
        return (uchar *)(session_var(target_thd, Time_zone*)->get_name().data());
    }
    uchar *global_value_ptr(THD *thd)
    {
        return (uchar *)(global_var(Time_zone*)->get_name().data());
    }
    bool check_update_type( BiaodashiValueType value_type )
    {
        return !schema::ColumnEntry::IsStringType( value_type );
    }
    void do_check(THD *thd, SetSysVarStructure* var);
};

/**
  Class representing the 'tx_isolation' system variable. This
  variable can also be indirectly set using 'SET TRANSACTION ISOLATION
  LEVEL'. This variable is deprecated and will be removed in a
  future release. 'transaction_isolation' is used an alternative
  instead.
*/
class Sys_var_tx_isolation: public Sys_var_enum
{
public:
    Sys_var_tx_isolation(const char *name_arg,
                         const char *comment, int flag_args, ptrdiff_t off, size_t size,
                         // CMD_LINE getopt,
                         const char *values[], uint def_val, PolyLock *lock,
                         enum binlog_status_enum binlog_status_arg,
                         on_check_function on_check_func,
                         on_update_function on_update_func=0,
                         const char *substitute=0)
            :Sys_var_enum(name_arg, comment, flag_args, off, size, // getopt,
                          values, def_val, lock, binlog_status_arg, on_check_func,
                          on_update_func, substitute)
    {}
    bool session_update(THD *thd, const SetSysVarStructure* var);
};

/**
  The class for @test_flags (core_file for now).
  It's derived from Sys_var_mybool.

  Class specific constructor arguments:
    Caller need not pass in a variable as we make up the value on the
    fly, that is, we derive it from the global test_flags bit vector.

  Backing store: my_bool
*/
class Sys_var_test_flag: public Sys_var_mybool
{
private:
    my_bool test_flag_value;
    uint    test_flag_mask;
public:
    Sys_var_test_flag(const char *name_arg, const char *comment, uint mask)
            : Sys_var_mybool(name_arg, comment, READ_ONLY GLOBAL_VAR(test_flag_value),
                             // NO_CMD_LINE,
                             (0))
    {
        test_flag_mask= mask;
    }
    uchar *global_value_ptr(THD *thd)
    {
        test_flag_value= ((test_flags & test_flag_mask) > 0);
        return (uchar*) &test_flag_value;
    }
};


#endif //AIRES_SYS_VARS_H
