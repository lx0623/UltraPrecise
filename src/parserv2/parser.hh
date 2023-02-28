// A Bison parser, made by GNU Bison 3.5.

// Skeleton interface for Bison LALR(1) parsers in C++

// Copyright (C) 2002-2015, 2018-2019 Free Software Foundation, Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// As a special exception, you may create a larger work that contains
// part or all of the Bison parser skeleton and distribute that work
// under terms of your choice, so long as that work isn't itself a
// parser generator using the skeleton or a modified version thereof
// as a parser skeleton.  Alternatively, if you modify or redistribute
// the parser skeleton itself, you may (at your option) remove this
// special exception, which will cause the skeleton and the resulting
// Bison output files to be licensed under the GNU General Public
// License without this special exception.

// This special exception was added by the Free Software Foundation in
// version 2.2 of Bison.


/**
 ** \file parser.hh
 ** Define the aries_parser::parser class.
 */

// C++ LALR(1) parser skeleton written by Akim Demaille.

// Undocumented macros, especially those whose name start with YY_,
// are private implementation details.  Do not rely on them.

#ifndef YY_ARIES_PARSER_PARSER_HH_INCLUDED
# define YY_ARIES_PARSER_PARSER_HH_INCLUDED
// "%code requires" blocks.
#line 45 "parser.yy"
 #include "location.h" 

#line 51 "parser.hh"


# include <cstdlib> // std::abort
# include <iostream>
# include <stdexcept>
# include <string>
# include <vector>

#if defined __cplusplus
# define YY_CPLUSPLUS __cplusplus
#else
# define YY_CPLUSPLUS 199711L
#endif

// Support move semantics when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_MOVE           std::move
# define YY_MOVE_OR_COPY   move
# define YY_MOVE_REF(Type) Type&&
# define YY_RVREF(Type)    Type&&
# define YY_COPY(Type)     Type
#else
# define YY_MOVE
# define YY_MOVE_OR_COPY   copy
# define YY_MOVE_REF(Type) Type&
# define YY_RVREF(Type)    const Type&
# define YY_COPY(Type)     const Type&
#endif

// Support noexcept when possible.
#if 201103L <= YY_CPLUSPLUS
# define YY_NOEXCEPT noexcept
# define YY_NOTHROW
#else
# define YY_NOEXCEPT
# define YY_NOTHROW throw ()
#endif

// Support constexpr when possible.
#if 201703 <= YY_CPLUSPLUS
# define YY_CONSTEXPR constexpr
#else
# define YY_CONSTEXPR
#endif


#ifndef YY_ASSERT
# include <cassert>
# define YY_ASSERT assert
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Debug traces.  */
#ifndef ARIES_PARSERDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define ARIES_PARSERDEBUG 1
#  else
#   define ARIES_PARSERDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define ARIES_PARSERDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined ARIES_PARSERDEBUG */

namespace aries_parser {
#line 193 "parser.hh"




  /// A Bison parser.
  class Parser
  {
  public:
#ifndef ARIES_PARSERSTYPE
  /// A buffer to store and retrieve objects.
  ///
  /// Sort of a variant, but does not keep track of the nature
  /// of the stored data, since that knowledge is available
  /// via the current parser state.
  class semantic_type
  {
  public:
    /// Type of *this.
    typedef semantic_type self_type;

    /// Empty construction.
    semantic_type () YY_NOEXCEPT
      : yybuffer_ ()
    {}

    /// Construct and fill.
    template <typename T>
    semantic_type (YY_RVREF (T) t)
    {
      YY_ASSERT (sizeof (T) <= size);
      new (yyas_<T> ()) T (YY_MOVE (t));
    }

    /// Destruction, allowed only if empty.
    ~semantic_type () YY_NOEXCEPT
    {}

# if 201103L <= YY_CPLUSPLUS
    /// Instantiate a \a T in here from \a t.
    template <typename T, typename... U>
    T&
    emplace (U&&... u)
    {
      return *new (yyas_<T> ()) T (std::forward <U>(u)...);
    }
# else
    /// Instantiate an empty \a T in here.
    template <typename T>
    T&
    emplace ()
    {
      return *new (yyas_<T> ()) T ();
    }

    /// Instantiate a \a T in here from \a t.
    template <typename T>
    T&
    emplace (const T& t)
    {
      return *new (yyas_<T> ()) T (t);
    }
# endif

    /// Instantiate an empty \a T in here.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build ()
    {
      return emplace<T> ();
    }

    /// Instantiate a \a T in here from \a t.
    /// Obsolete, use emplace.
    template <typename T>
    T&
    build (const T& t)
    {
      return emplace<T> (t);
    }

    /// Accessor to a built \a T.
    template <typename T>
    T&
    as () YY_NOEXCEPT
    {
      return *yyas_<T> ();
    }

    /// Const accessor to a built \a T (for %printer).
    template <typename T>
    const T&
    as () const YY_NOEXCEPT
    {
      return *yyas_<T> ();
    }

    /// Swap the content with \a that, of same type.
    ///
    /// Both variants must be built beforehand, because swapping the actual
    /// data requires reading it (with as()), and this is not possible on
    /// unconstructed variants: it would require some dynamic testing, which
    /// should not be the variant's responsibility.
    /// Swapping between built and (possibly) non-built is done with
    /// self_type::move ().
    template <typename T>
    void
    swap (self_type& that) YY_NOEXCEPT
    {
      std::swap (as<T> (), that.as<T> ());
    }

    /// Move the content of \a that to this.
    ///
    /// Destroys \a that.
    template <typename T>
    void
    move (self_type& that)
    {
# if 201103L <= YY_CPLUSPLUS
      emplace<T> (std::move (that.as<T> ()));
# else
      emplace<T> ();
      swap<T> (that);
# endif
      that.destroy<T> ();
    }

# if 201103L <= YY_CPLUSPLUS
    /// Move the content of \a that to this.
    template <typename T>
    void
    move (self_type&& that)
    {
      emplace<T> (std::move (that.as<T> ()));
      that.destroy<T> ();
    }
#endif

    /// Copy the content of \a that to this.
    template <typename T>
    void
    copy (const self_type& that)
    {
      emplace<T> (that.as<T> ());
    }

    /// Destroy the stored \a T.
    template <typename T>
    void
    destroy ()
    {
      as<T> ().~T ();
    }

  private:
    /// Prohibit blind copies.
    self_type& operator= (const self_type&);
    semantic_type (const self_type&);

    /// Accessor to raw memory as \a T.
    template <typename T>
    T*
    yyas_ () YY_NOEXCEPT
    {
      void *yyp = yybuffer_.yyraw;
      return static_cast<T*> (yyp);
     }

    /// Const accessor to raw memory as \a T.
    template <typename T>
    const T*
    yyas_ () const YY_NOEXCEPT
    {
      const void *yyp = yybuffer_.yyraw;
      return static_cast<const T*> (yyp);
     }

    /// An auxiliary type to compute the largest semantic type.
    union union_type
    {
      // drop_table_stmt
      // drop_user_stmt
      // drop_view_stmt
      // drop_database_stmt
      // create
      // create_table_stmt
      // view_or_trigger_or_sp_or_event
      // no_definer_tail
      // view_tail
      // use
      char dummy1[sizeof (AbstractCommandPointer)];

      // user
      // create_user
      char dummy2[sizeof (AccountSPtr)];

      // kill
      // shutdown_stmt
      char dummy3[sizeof (AdminStmtStructurePtr)];

      // cast_type
      char dummy4[sizeof (CastType)];

      // opt_column_attribute_list
      // column_attribute_list
      char dummy5[sizeof (ColAttrList)];

      // opt_create_table_options_etc
      char dummy6[sizeof (CreateTableOptions)];

      // delete_stmt
      char dummy7[sizeof (DeleteStructurePtr)];

      // expr
      // bool_pri
      // predicate
      // bit_expr
      // simple_expr
      // case_expr
      // function_call_keyword
      // function_call_nonkeyword
      // function_call_conflict
      // function_call_generic
      // udf_expr
      // set_function_specification
      // sum_expr
      // row_subquery
      // param_marker
      // variable
      // set_expr_or_default
      // in_sum_expr
      // opt_expr
      // opt_else
      // opt_where_clause
      // opt_where_clause_expr
      // opt_having_clause
      // table_wild
      // grouping_expr
      // func_datetime_precision
      // now_or_signed_literal
      // now
      // expr_or_default
      char dummy8[sizeof (Expression)];

      // field_def
      char dummy9[sizeof (Field_def_ptr)];

      // field_options
      // field_opt_list
      // field_option
      char dummy10[sizeof (Field_option)];

      // opt_field_term
      // field_term_list
      // field_term
      char dummy11[sizeof (Field_separators)];

      // opt_from_clause
      // from_clause
      char dummy12[sizeof (FromPartStructurePointer)];

      // opt_group_clause
      char dummy13[sizeof (GroupbyStructurePointer)];

      // insert_stmt
      char dummy14[sizeof (InsertStructurePtr)];

      // table_reference
      // joined_table
      // table_factor
      // single_table_parens
      // single_table
      // joined_table_parens
      // derived_table
      char dummy15[sizeof (JoinStructurePointer)];

      // natural_join_type
      // inner_join_type
      // outer_join_type
      char dummy16[sizeof (JoinType)];

      // opt_limit_clause
      // limit_clause
      // limit_options
      // opt_simple_limit
      char dummy17[sizeof (LimitStructurePointer)];

      // opt_line_term
      // line_term_list
      // line_term
      char dummy18[sizeof (Line_separators)];

      // signed_literal
      // literal
      // NUM_literal
      char dummy19[sizeof (Literal)];

      // load_stmt
      char dummy20[sizeof (LoadDataStructurePtr)];

      // duplicate
      // opt_duplicate
      char dummy21[sizeof (On_duplicate)];

      // order_expr
      char dummy22[sizeof (OrderItem)];

      // opt_order_clause
      // order_clause
      char dummy23[sizeof (OrderbyStructurePointer)];

      // type
      char dummy24[sizeof (PT_ColumnType_ptr)];

      // column_attribute
      char dummy25[sizeof (PT_column_attr_base_ptr)];

      // option_value_following_option_type
      char dummy26[sizeof (PT_option_value_following_option_type_ptr)];

      // part_definition
      char dummy27[sizeof (PartDef)];

      // opt_part_defs
      // part_def_list
      char dummy28[sizeof (PartDefList)];

      // part_type_def
      char dummy29[sizeof (PartTypeDef)];

      // part_value_item
      char dummy30[sizeof (PartValueItem)];

      // part_func_max
      // part_value_item_list_paren
      // part_value_item_list
      char dummy31[sizeof (PartValueItemsSPtr)];

      // opt_part_values
      char dummy32[sizeof (PartValuesSPtr)];

      // opt_create_partitioning_etc
      // partition_clause
      char dummy33[sizeof (PartitionStructureSPtr)];

      // float_options
      // precision
      // opt_precision
      char dummy34[sizeof (Precision_ptr)];

      // prepare_src
      char dummy35[sizeof (PrepareSrcPtr)];

      // prepare
      // execute
      // deallocate
      char dummy36[sizeof (PreparedStmtStructurePtr)];

      // show_engine_param
      char dummy37[sizeof (SHOW_CMD)];

      // insert_ident
      // simple_ident
      // simple_ident_nospvar
      // simple_ident_q
      char dummy38[sizeof (SQLIdentPtr)];

      // select_item_list
      char dummy39[sizeof (SelectPartStructurePointer)];

      // select_stmt
      // query_expression
      // query_expression_body
      // query_expression_parens
      // query_primary
      // query_specification
      // table_subquery
      // subquery
      // view_select
      // query_expression_or_parens
      // explain_stmt
      char dummy40[sizeof (SelectStructurePointer)];

      // union_option
      char dummy41[sizeof (SetOperationType)];

      // option_value
      // option_value_no_option_type
      char dummy42[sizeof (SetStructurePtr)];

      // describe_stmt
      // show
      // show_param
      char dummy43[sizeof (ShowStructurePtr)];

      // opt_show_cmd_type
      char dummy44[sizeof (Show_cmd_type)];

      // opt_table_list
      // table_list
      // table_alias_ref_list
      char dummy45[sizeof (TABLE_LIST)];

      // table_element
      // column_def
      // table_constraint_def
      char dummy46[sizeof (TableElementDescriptionPtr)];

      // start
      // begin_stmt
      // commit
      // rollback
      char dummy47[sizeof (TransactionStructurePtr)];

      // update_stmt
      char dummy48[sizeof (UpdateStructurePtr)];

      // row_value
      // opt_values
      // values
      char dummy49[sizeof (VALUES)];

      // variable_aux
      char dummy50[sizeof (VariableStructurePtr)];

      // opt_wild_or_where
      // opt_wild_or_where_for_show
      char dummy51[sizeof (WildOrWhere_ptr)];

      // opt_distinct
      // opt_if_not_exists
      // opt_not
      // if_exists
      // opt_temporary
      // opt_linear
      // visibility
      // opt_full
      // opt_extended
      // opt_ignore
      // opt_local
      char dummy52[sizeof (bool)];

      // data_or_xml
      char dummy53[sizeof (enum_filetype)];

      // option_type
      // opt_var_type
      // opt_var_ident_type
      // opt_set_var_ident_type
      char dummy54[sizeof (enum_var_type)];

      // opt_chain
      // opt_release
      char dummy55[sizeof (enum_yes_no_unknown)];

      // select_options
      // select_option_list
      // select_option
      // comp_op
      // query_spec_option
      // kill_option
      char dummy56[sizeof (int)];

      // constraint_key_type
      char dummy57[sizeof (keytype)];

      // update_elem
      char dummy58[sizeof (pair< BiaodashiPointer, BiaodashiPointer >)];

      // insert_query_expression
      char dummy59[sizeof (pair< EXPR_LIST, SelectStructurePointer >)];

      // encode_type
      char dummy60[sizeof (pair< aries::EncodeType, string >)];

      // opt_references
      // references
      char dummy61[sizeof (pair< shared_ptr<BasicRel>, vector< string > >)];

      // opt_insert_update_list
      // update_list
      char dummy62[sizeof (pair< vector< BiaodashiPointer >, vector< BiaodashiPointer > > )];

      // user_list
      // create_user_list
      char dummy63[sizeof (shared_ptr< vector< AccountSPtr > >)];

      // table_ident
      // sp_name
      // table_ident_opt_wild
      char dummy64[sizeof (shared_ptr<BasicRel>)];

      // option_value_list_continued
      // start_option_value_list_following_option_type
      // option_value_list
      // set
      // start_option_value_list
      char dummy65[sizeof (shared_ptr<vector<SetStructurePtr>>)];

      // table_element_list
      char dummy66[sizeof (shared_ptr<vector<TableElementDescriptionPtr>>)];

      // ACCOUNT_SYM
      // ACTION
      // ADDDATE_SYM
      // AFTER_SYM
      // AGAINST
      // AGGREGATE_SYM
      // ALGORITHM_SYM
      // ALWAYS_SYM
      // ANY_SYM
      // ASC
      // ASCII_SYM
      // AT_SYM
      // AUTOEXTEND_SIZE_SYM
      // AUTO_INC
      // AVG_ROW_LENGTH
      // AVG_SYM
      // BACKUP_SYM
      // BEGIN_SYM
      // BINLOG_SYM
      // BIT_SYM
      // BLOCK_SYM
      // BOOLEAN_SYM
      // BOOL_SYM
      // BTREE_SYM
      // BYTE_SYM
      // CACHE_SYM
      // CASCADED
      // CATALOG_NAME_SYM
      // CHAIN_SYM
      // CHANGED
      // CHANNEL_SYM
      // CHARSET
      // CHECKSUM_SYM
      // CIPHER_SYM
      // CLASS_ORIGIN_SYM
      // CLIENT_SYM
      // CLOSE_SYM
      // COALESCE
      // CODE_SYM
      // COLLATION_SYM
      // COLUMNS
      // COLUMN_FORMAT_SYM
      // COLUMN_NAME_SYM
      // COMMENT_SYM
      // COMMITTED_SYM
      // COMMIT_SYM
      // COMPACT_SYM
      // COMPLETION_SYM
      // COMPRESSED_SYM
      // COMPRESSION_SYM
      // ENCRYPTION_SYM
      // CONCURRENT
      // CONNECTION_SYM
      // CONSISTENT_SYM
      // CONSTRAINT_CATALOG_SYM
      // CONSTRAINT_NAME_SYM
      // CONSTRAINT_SCHEMA_SYM
      // CONTAINS_SYM
      // CONTEXT_SYM
      // CPU_SYM
      // CREATE
      // CURRENT_SYM
      // CURSOR_NAME_SYM
      // DATAFILE_SYM
      // DATA_SYM
      // DATETIME_SYM
      // DATE_SYM
      // DAY_SYM
      // DEALLOCATE_SYM
      // DECIMAL_NUM
      // DEFAULT_AUTH_SYM
      // DEFINER_SYM
      // DELAY_KEY_WRITE_SYM
      // DESC
      // DIAGNOSTICS_SYM
      // BYTEDICT_SYM
      // SHORTDICT_SYM
      // INTDICT_SYM
      // DIRECTORY_SYM
      // DISABLE_SYM
      // DISCARD_SYM
      // DISK_SYM
      // DO_SYM
      // DUMPFILE
      // DUPLICATE_SYM
      // DYNAMIC_SYM
      // ENABLE_SYM
      // END
      // ENDS_SYM
      // ENGINES_SYM
      // ENGINE_SYM
      // ENUM_SYM
      // ERROR_SYM
      // ERRORS
      // ESCAPE_SYM
      // EVENTS_SYM
      // EVENT_SYM
      // EVERY_SYM
      // EXCHANGE_SYM
      // EXECUTE_SYM
      // EXPANSION_SYM
      // EXPIRE_SYM
      // EXPORT_SYM
      // EXTENDED_SYM
      // EXTENT_SIZE_SYM
      // FAST_SYM
      // FAULTS_SYM
      // FILE_SYM
      // FILE_BLOCK_SIZE_SYM
      // FILTER_SYM
      // FIRST_SYM
      // FIXED_SYM
      // FLOAT_NUM
      // FLUSH_SYM
      // FOLLOWS_SYM
      // FORMAT_SYM
      // FOUND_SYM
      // FULL
      // GENERAL
      // GROUP_REPLICATION
      // GEOMETRYCOLLECTION_SYM
      // GEOMETRY_SYM
      // GET_FORMAT
      // GLOBAL_SYM
      // GRANTS
      // HANDLER_SYM
      // HASH_SYM
      // HELP_SYM
      // HEX_NUM
      // HOST_SYM
      // HOSTS_SYM
      // HOUR_SYM
      // IDENT
      // IDENTIFIED_SYM
      // IGNORE_SERVER_IDS_SYM
      // IMPORT
      // INDEXES
      // INITIAL_SIZE_SYM
      // INSERT_METHOD
      // INSTANCE_SYM
      // INSTALL_SYM
      // INTERVAL_SYM
      // INVOKER_SYM
      // IO_SYM
      // IPC_SYM
      // ISOLATION
      // ISSUER_SYM
      // JSON_SYM
      // KEY_BLOCK_SIZE
      // LANGUAGE_SYM
      // LAST_SYM
      // LEAVES
      // LESS_SYM
      // LEVEL_SYM
      // LEX_HOSTNAME
      // LINESTRING_SYM
      // LIST_SYM
      // LOCAL_SYM
      // LOCKS_SYM
      // LOGFILE_SYM
      // LOGS_SYM
      // LONG_NUM
      // MASTER_AUTO_POSITION_SYM
      // MASTER_CONNECT_RETRY_SYM
      // MASTER_DELAY_SYM
      // MASTER_HOST_SYM
      // MASTER_LOG_FILE_SYM
      // MASTER_LOG_POS_SYM
      // MASTER_PASSWORD_SYM
      // MASTER_PORT_SYM
      // MASTER_RETRY_COUNT_SYM
      // MASTER_SERVER_ID_SYM
      // MASTER_SSL_CAPATH_SYM
      // MASTER_TLS_VERSION_SYM
      // MASTER_SSL_CA_SYM
      // MASTER_SSL_CERT_SYM
      // MASTER_SSL_CIPHER_SYM
      // MASTER_SSL_CRL_SYM
      // MASTER_SSL_CRLPATH_SYM
      // MASTER_SSL_KEY_SYM
      // MASTER_SSL_SYM
      // MASTER_SYM
      // MASTER_USER_SYM
      // MASTER_HEARTBEAT_PERIOD_SYM
      // MAX_CONNECTIONS_PER_HOUR
      // MAX_QUERIES_PER_HOUR
      // MAX_ROWS
      // MAX_SIZE_SYM
      // MAX_UPDATES_PER_HOUR
      // MAX_USER_CONNECTIONS_SYM
      // MEDIUM_SYM
      // MEMORY_SYM
      // MERGE_SYM
      // MESSAGE_TEXT_SYM
      // MICROSECOND_SYM
      // MIGRATE_SYM
      // MINUTE_SYM
      // MIN_ROWS
      // MODE_SYM
      // MODIFY_SYM
      // MONTH_SYM
      // MULTILINESTRING_SYM
      // MULTIPOINT_SYM
      // MULTIPOLYGON_SYM
      // MUTEX_SYM
      // MYSQL_ERRNO_SYM
      // NAMES_SYM
      // NAME_SYM
      // NATIONAL_SYM
      // NCHAR_SYM
      // NDBCLUSTER_SYM
      // NEVER_SYM
      // NEW_SYM
      // NEXT_SYM
      // NODEGROUP_SYM
      // NONE_SYM
      // NO_SYM
      // NO_WAIT_SYM
      // NUM
      // NUMBER_SYM
      // NVARCHAR_SYM
      // OFFSET_SYM
      // ONE_SYM
      // ONLY_SYM
      // OPEN_SYM
      // OPTIONS_SYM
      // OWNER_SYM
      // PACK_KEYS_SYM
      // PAGE_SYM
      // PARSER_SYM
      // PARTIAL
      // PARTITIONS_SYM
      // PARTITIONING_SYM
      // PASSWORD
      // PHASE_SYM
      // PLUGIN_DIR_SYM
      // PLUGIN_SYM
      // PLUGINS_SYM
      // "."
      // POLYGON_SYM
      // PORT_SYM
      // PRECEDES_SYM
      // PREPARE_SYM
      // PRESERVE_SYM
      // PREV_SYM
      // PRIVILEGES
      // PROCESS
      // PROCESSLIST_SYM
      // PROFILE_SYM
      // PROFILES_SYM
      // PROXY_SYM
      // QUARTER_SYM
      // QUERY_SYM
      // QUICK
      // READ_ONLY_SYM
      // REBUILD_SYM
      // RECOVER_SYM
      // REDO_BUFFER_SIZE_SYM
      // REDUNDANT_SYM
      // RELAY
      // RELAYLOG_SYM
      // RELAY_LOG_FILE_SYM
      // RELAY_LOG_POS_SYM
      // RELAY_THREAD
      // RELOAD
      // REMOVE_SYM
      // REORGANIZE_SYM
      // REPAIR
      // REPEATABLE_SYM
      // REPLICATION
      // REPLICATE_DO_DB
      // REPLICATE_IGNORE_DB
      // REPLICATE_DO_TABLE
      // REPLICATE_IGNORE_TABLE
      // REPLICATE_WILD_DO_TABLE
      // REPLICATE_WILD_IGNORE_TABLE
      // REPLICATE_REWRITE_DB
      // RESET_SYM
      // RESOURCES
      // RESTORE_SYM
      // RESUME_SYM
      // RETURNED_SQLSTATE_SYM
      // RETURNS_SYM
      // REVERSE_SYM
      // ROLLBACK_SYM
      // ROLLUP_SYM
      // ROTATE_SYM
      // ROUTINE_SYM
      // ROW_FORMAT_SYM
      // ROW_COUNT_SYM
      // RTREE_SYM
      // SAVEPOINT_SYM
      // SCHEDULE_SYM
      // SCHEMA_NAME_SYM
      // SECOND_SYM
      // SECURITY_SYM
      // SERIALIZABLE_SYM
      // SERIAL_SYM
      // SESSION_SYM
      // SERVER_SYM
      // SHARE_SYM
      // SHARES_SYM
      // SHUTDOWN
      // SIGNED_SYM
      // SIMPLE_SYM
      // SLAVE
      // SLOW
      // SNAPSHOT_SYM
      // SOCKET_SYM
      // SONAME_SYM
      // SOUNDS_SYM
      // SOURCE_SYM
      // SQL_AFTER_GTIDS
      // SQL_AFTER_MTS_GAPS
      // SQL_BEFORE_GTIDS
      // SQL_BUFFER_RESULT
      // SQL_NO_CACHE_SYM
      // SQL_THREAD
      // STACKED_SYM
      // STARTS_SYM
      // START_SYM
      // STATS_AUTO_RECALC_SYM
      // STATS_PERSISTENT_SYM
      // STATS_SAMPLE_PAGES_SYM
      // STATUS_SYM
      // STOP_SYM
      // STORAGE_SYM
      // STRING_SYM
      // SUBCLASS_ORIGIN_SYM
      // SUBDATE_SYM
      // SUBJECT_SYM
      // SUBPARTITIONS_SYM
      // SUBPARTITION_SYM
      // SUPER_SYM
      // SUSPEND_SYM
      // SWAPS_SYM
      // SWITCHES_SYM
      // TABLES
      // VIEWS
      // TABLESPACE_SYM
      // TABLE_CHECKSUM_SYM
      // TABLE_NAME_SYM
      // TEMPORARY
      // TEMPTABLE_SYM
      // TEXT_STRING
      // TEXT_SYM
      // THAN_SYM
      // TIMESTAMP_SYM
      // TIMESTAMP_ADD
      // TIMESTAMP_DIFF
      // TIME_SYM
      // TRANSACTION_SYM
      // TRIGGERS_SYM
      // TRUNCATE_SYM
      // TYPES_SYM
      // TYPE_SYM
      // ULONGLONG_NUM
      // UNCOMMITTED_SYM
      // UNDEFINED_SYM
      // UNDOFILE_SYM
      // UNDO_BUFFER_SIZE_SYM
      // UNICODE_SYM
      // UNINSTALL_SYM
      // UNKNOWN_SYM
      // UNTIL_SYM
      // UPGRADE_SYM
      // USER
      // USE_FRM
      // VALIDATION_SYM
      // VALUE_SYM
      // VARBINARY_SYM
      // VARIABLES
      // VIEW_SYM
      // WAIT_SYM
      // WARNINGS
      // WEEK_SYM
      // WEIGHT_STRING_SYM
      // WITHOUT_SYM
      // WORK_SYM
      // WRAPPER_SYM
      // X509_SYM
      // XA_SYM
      // XID_SYM
      // XML_SYM
      // YEAR_SYM
      // PERSIST_SYM
      // ROLE_SYM
      // ADMIN_SYM
      // INVISIBLE_SYM
      // VISIBLE_SYM
      // COMPONENT_SYM
      // SKIP_SYM
      // LOCKED_SYM
      // NOWAIT_SYM
      // PERSIST_ONLY_SYM
      // HISTOGRAM_SYM
      // BUCKETS_SYM
      // OBSOLETE_TOKEN_930
      // CLONE_SYM
      // EXCLUDE_SYM
      // FOLLOWING_SYM
      // NULLS_SYM
      // OTHERS_SYM
      // PRECEDING_SYM
      // RESPECT_SYM
      // TIES_SYM
      // UNBOUNDED_SYM
      // NESTED_SYM
      // ORDINALITY_SYM
      // PATH_SYM
      // HISTORY_SYM
      // REUSE_SYM
      // SRID_SYM
      // THREAD_PRIORITY_SYM
      // RESOURCE_SYM
      // VCPU_SYM
      // MASTER_PUBLIC_KEY_PATH_SYM
      // GET_MASTER_PUBLIC_KEY_SYM
      // RESTART_SYM
      // DEFINITION_SYM
      // DESCRIPTION_SYM
      // ORGANIZATION_SYM
      // REFERENCE_SYM
      // ACTIVE_SYM
      // INACTIVE_SYM
      // OPTIONAL_SYM
      // SECONDARY_SYM
      // SECONDARY_ENGINE_SYM
      // SECONDARY_LOAD_SYM
      // SECONDARY_UNLOAD_SYM
      // RETAIN_SYM
      // OLD_SYM
      // ENFORCED_SYM
      // OJ_SYM
      // NETWORK_NAMESPACE_SYM
      // select_alias
      // opt_constraint_name
      // execute_var_ident
      // text_literal
      // text_string
      // opt_index_name_and_type
      // key_part
      // key_part_with_expression
      // opt_ident
      // opt_component
      // charset_name
      // internal_variable_name
      // interval
      // interval_time_stamp
      // opt_table_alias
      // opt_ordering_direction
      // ordering_direction
      // limit_option
      // IDENT_sys
      // TEXT_STRING_sys
      // TEXT_STRING_literal
      // TEXT_STRING_filesystem
      // TEXT_STRING_password
      // TEXT_STRING_hash
      // ident
      // ident_or_text
      // nchar
      // varchar
      // nvarchar
      // int_type
      // real_type
      // numeric_type
      // type_datetime_precision
      // field_length
      // opt_field_length
      // ident_keyword
      // ident_keywords_ambiguous_2_labels
      // ident_keywords_unambiguous
      // lvalue_keyword
      // TEXT_STRING_sys_nonewline
      // opt_describe_column
      // opt_db
      // lvalue_ident
      // password
      // opt_load_data_charset
      // opt_xml_rows_identified_by
      char dummy67[sizeof (string)];

      // load_data_lock
      char dummy68[sizeof (thr_lock_type)];

      // select_item
      char dummy69[sizeof (tuple<Expression, string>)];

      // opt_num_parts
      // opt_num_subparts
      // ulong_num
      // opt_ignore_lines
      char dummy70[sizeof (ulong)];

      // size_number
      // real_ulong_num
      // ulonglong_num
      // real_ulonglong_num
      char dummy71[sizeof (ulonglong)];

      // part_values_in
      // part_value_list
      char dummy72[sizeof (vector< PartValueItemsSPtr >)];

      // opt_udf_expr_list
      // udf_expr_list
      // opt_expr_list
      // expr_list
      // group_list
      // insert_from_constructor
      // fields
      char dummy73[sizeof (vector<Expression>)];

      // from_tables
      // table_reference_list
      char dummy74[sizeof (vector<JoinStructurePointer>)];

      // order_list
      char dummy75[sizeof (vector<OrderItem>)];

      // execute_using
      // execute_var_list
      // key_list
      // key_list_with_expression
      // opt_derived_column_list
      // simple_ident_list
      char dummy76[sizeof (vector<string>)];

      // when_list
      char dummy77[sizeof (vector<tuple<Expression, Expression>>)];
    };

    /// The size of the largest semantic type.
    enum { size = sizeof (union_type) };

    /// A buffer to store semantic values.
    union
    {
      /// Strongest alignment constraints.
      long double yyalign_me;
      /// A buffer large enough to store any of the semantic values.
      char yyraw[size];
    } yybuffer_;
  };

#else
    typedef ARIES_PARSERSTYPE semantic_type;
#endif
    /// Symbol locations.
    typedef aries_parser::location location_type;

    /// Syntax errors thrown from user actions.
    struct syntax_error : std::runtime_error
    {
      syntax_error (const location_type& l, const std::string& m)
        : std::runtime_error (m)
        , location (l)
      {}

      syntax_error (const syntax_error& s)
        : std::runtime_error (s.what ())
        , location (s.location)
      {}

      ~syntax_error () YY_NOEXCEPT YY_NOTHROW;

      location_type location;
    };

    /// Tokens.
    struct token
    {
      enum yytokentype
      {
        END_OF_INPUT = 0,
        ABORT_SYM = 258,
        ACCESSIBLE_SYM = 259,
        ACCOUNT_SYM = 260,
        ACTION = 261,
        ADD = 262,
        ADDDATE_SYM = 263,
        AFTER_SYM = 264,
        AGAINST = 265,
        AGGREGATE_SYM = 266,
        ALGORITHM_SYM = 267,
        ALL = 268,
        ALTER = 269,
        ALWAYS_SYM = 270,
        OBSOLETE_TOKEN_271 = 271,
        ANALYZE_SYM = 272,
        AND_AND_SYM = 273,
        AND_SYM = 274,
        ANY_SYM = 275,
        AS = 276,
        ASC = 277,
        ASCII_SYM = 278,
        ASENSITIVE_SYM = 279,
        AT_SYM = 280,
        AUTOEXTEND_SIZE_SYM = 281,
        AUTO_INC = 282,
        AVG_ROW_LENGTH = 283,
        AVG_SYM = 284,
        BACKUP_SYM = 285,
        BEFORE_SYM = 286,
        BEGIN_SYM = 287,
        BETWEEN_SYM = 288,
        BIGINT_SYM = 289,
        BINARY_SYM = 290,
        BINLOG_SYM = 291,
        BIN_NUM = 292,
        BIT_AND = 293,
        BIT_OR = 294,
        BIT_SYM = 295,
        BIT_XOR = 296,
        BLOB_SYM = 297,
        BLOCK_SYM = 298,
        BOOLEAN_SYM = 299,
        BOOL_SYM = 300,
        BOTH = 301,
        BTREE_SYM = 302,
        BY = 303,
        BYTE_SYM = 304,
        CACHE_SYM = 305,
        CALL_SYM = 306,
        CASCADE = 307,
        CASCADED = 308,
        CASE_SYM = 309,
        CAST_SYM = 310,
        CATALOG_NAME_SYM = 311,
        CHAIN_SYM = 312,
        CHANGE = 313,
        CHANGED = 314,
        CHANNEL_SYM = 315,
        CHARSET = 316,
        CHAR_SYM = 317,
        CHECKSUM_SYM = 318,
        CHECK_SYM = 319,
        CIPHER_SYM = 320,
        CLASS_ORIGIN_SYM = 321,
        CLIENT_SYM = 322,
        CLOSE_SYM = 323,
        COALESCE = 324,
        CODE_SYM = 325,
        COLLATE_SYM = 326,
        COLLATION_SYM = 327,
        COLUMNS = 328,
        COLUMN_SYM = 329,
        COLUMN_FORMAT_SYM = 330,
        COLUMN_NAME_SYM = 331,
        COMMENT_SYM = 332,
        COMMITTED_SYM = 333,
        COMMIT_SYM = 334,
        COMPACT_SYM = 335,
        COMPLETION_SYM = 336,
        COMPRESSED_SYM = 337,
        COMPRESSION_SYM = 338,
        ENCRYPTION_SYM = 339,
        CONCURRENT = 340,
        CONDITION_SYM = 341,
        CONNECTION_ID_SYM = 342,
        CONNECTION_SYM = 343,
        CONSISTENT_SYM = 344,
        CONSTRAINT = 345,
        CONSTRAINT_CATALOG_SYM = 346,
        CONSTRAINT_NAME_SYM = 347,
        CONSTRAINT_SCHEMA_SYM = 348,
        CONTAINS_SYM = 349,
        CONTEXT_SYM = 350,
        CONTINUE_SYM = 351,
        CONVERT_SYM = 352,
        COUNT_SYM = 353,
        CPU_SYM = 354,
        CREATE = 355,
        CROSS = 356,
        CUBE_SYM = 357,
        CURDATE = 358,
        CURRENT_SYM = 359,
        CURRENT_USER = 360,
        CURSOR_SYM = 361,
        CURSOR_NAME_SYM = 362,
        CURTIME = 363,
        DATABASE = 364,
        DATABASES = 365,
        DATAFILE_SYM = 366,
        DATA_SYM = 367,
        DATETIME_SYM = 368,
        DATE_ADD_INTERVAL = 369,
        DATE_SUB_INTERVAL = 370,
        DATE_SYM = 371,
        DAY_HOUR_SYM = 372,
        DAY_MICROSECOND_SYM = 373,
        DAY_MINUTE_SYM = 374,
        DAY_SECOND_SYM = 375,
        DAY_SYM = 376,
        DEALLOCATE_SYM = 377,
        DECIMAL_NUM = 378,
        REAL_NUM = 379,
        DECIMAL_SYM = 380,
        DECLARE_SYM = 381,
        DEFAULT_SYM = 382,
        DEFAULT_AUTH_SYM = 383,
        DEFINER_SYM = 384,
        DELAYED_SYM = 385,
        DELAY_KEY_WRITE_SYM = 386,
        DELETE_SYM = 387,
        DESC = 388,
        DESCRIBE = 389,
        OBSOLETE_TOKEN_388 = 390,
        DETERMINISTIC_SYM = 391,
        DIAGNOSTICS_SYM = 392,
        BYTEDICT_SYM = 393,
        SHORTDICT_SYM = 394,
        INTDICT_SYM = 395,
        DICT_INDEX_SYM = 396,
        DIRECTORY_SYM = 397,
        DISABLE_SYM = 398,
        DISCARD_SYM = 399,
        DISK_SYM = 400,
        DISTINCT = 401,
        DIV_SYM = 402,
        DOUBLE_SYM = 403,
        DO_SYM = 404,
        DROP = 405,
        DUAL_SYM = 406,
        DUMPFILE = 407,
        DUPLICATE_SYM = 408,
        DYNAMIC_SYM = 409,
        EACH_SYM = 410,
        ELSE = 411,
        ELSEIF_SYM = 412,
        ENABLE_SYM = 413,
        ENCLOSED = 414,
        ENCODING = 415,
        END = 416,
        ENDS_SYM = 417,
        ENGINES_SYM = 418,
        ENGINE_SYM = 419,
        ENUM_SYM = 420,
        EQ = 421,
        EQUAL_SYM = 422,
        ERROR_SYM = 423,
        ERRORS = 424,
        ESCAPED = 425,
        ESCAPE_SYM = 426,
        EVENTS_SYM = 427,
        EVENT_SYM = 428,
        EVERY_SYM = 429,
        EXCHANGE_SYM = 430,
        EXECUTE_SYM = 431,
        EXISTS = 432,
        EXIT_SYM = 433,
        EXPANSION_SYM = 434,
        EXPIRE_SYM = 435,
        EXPORT_SYM = 436,
        EXTENDED_SYM = 437,
        EXTENT_SIZE_SYM = 438,
        EXTRACT_SYM = 439,
        FALSE_SYM = 440,
        FAST_SYM = 441,
        FAULTS_SYM = 442,
        FETCH_SYM = 443,
        FILE_SYM = 444,
        FILE_BLOCK_SIZE_SYM = 445,
        FILTER_SYM = 446,
        FIRST_SYM = 447,
        FIXED_SYM = 448,
        FLOAT_NUM = 449,
        FLOAT_SYM = 450,
        FLUSH_SYM = 451,
        FOLLOWS_SYM = 452,
        FORCE_SYM = 453,
        FOREIGN = 454,
        FOR_SYM = 455,
        FORMAT_SYM = 456,
        FOUND_SYM = 457,
        FROM = 458,
        FULL = 459,
        FULLTEXT_SYM = 460,
        FUNCTION_SYM = 461,
        GE = 462,
        GENERAL = 463,
        GENERATED = 464,
        GROUP_REPLICATION = 465,
        GEOMETRYCOLLECTION_SYM = 466,
        GEOMETRY_SYM = 467,
        GET_FORMAT = 468,
        GET_SYM = 469,
        GLOBAL_SYM = 470,
        GRANT = 471,
        GRANTS = 472,
        GROUP_SYM = 473,
        GROUP_CONCAT_SYM = 474,
        GT_SYM = 475,
        HANDLER_SYM = 476,
        HASH_SYM = 477,
        HAVING = 478,
        HELP_SYM = 479,
        HEX_NUM = 480,
        HIGH_PRIORITY = 481,
        HOST_SYM = 482,
        HOSTS_SYM = 483,
        HOUR_MICROSECOND_SYM = 484,
        HOUR_MINUTE_SYM = 485,
        HOUR_SECOND_SYM = 486,
        HOUR_SYM = 487,
        IDENT = 488,
        IDENTIFIED_SYM = 489,
        IDENT_QUOTED = 490,
        IF = 491,
        IGNORE_SYM = 492,
        IGNORE_SERVER_IDS_SYM = 493,
        IMPORT = 494,
        INDEXES = 495,
        INDEX_SYM = 496,
        INFILE = 497,
        INITIAL_SIZE_SYM = 498,
        INNER_SYM = 499,
        INOUT_SYM = 500,
        INSENSITIVE_SYM = 501,
        INSERT_SYM = 502,
        INSERT_METHOD = 503,
        INSTANCE_SYM = 504,
        INSTALL_SYM = 505,
        INTERVAL_SYM = 506,
        INTO = 507,
        INT_SYM = 508,
        INTEGER_SYM = 509,
        INVOKER_SYM = 510,
        IN_SYM = 511,
        IO_AFTER_GTIDS = 512,
        IO_BEFORE_GTIDS = 513,
        IO_SYM = 514,
        IPC_SYM = 515,
        IS = 516,
        ISOLATION = 517,
        ISSUER_SYM = 518,
        ITERATE_SYM = 519,
        JOIN_SYM = 520,
        JSON_SEPARATOR_SYM = 521,
        JSON_SYM = 522,
        KEYS = 523,
        KEY_BLOCK_SIZE = 524,
        KEY_SYM = 525,
        KILL_SYM = 526,
        LANGUAGE_SYM = 527,
        LAST_SYM = 528,
        LE = 529,
        LEADING = 530,
        LEAVES = 531,
        LEAVE_SYM = 532,
        LEFT = 533,
        LESS_SYM = 534,
        LEVEL_SYM = 535,
        LEX_HOSTNAME = 536,
        LIKE = 537,
        LIMIT = 538,
        LINEAR_SYM = 539,
        LINES = 540,
        LINESTRING_SYM = 541,
        LIST_SYM = 542,
        LOAD = 543,
        LOCAL_SYM = 544,
        OBSOLETE_TOKEN_538 = 545,
        LOCKS_SYM = 546,
        LOCK_SYM = 547,
        LOGFILE_SYM = 548,
        LOGS_SYM = 549,
        LONGBLOB_SYM = 550,
        LONGTEXT_SYM = 551,
        LONG_NUM = 552,
        LONG_SYM = 553,
        LOOP_SYM = 554,
        LOW_PRIORITY = 555,
        LT = 556,
        MASTER_AUTO_POSITION_SYM = 557,
        MASTER_BIND_SYM = 558,
        MASTER_CONNECT_RETRY_SYM = 559,
        MASTER_DELAY_SYM = 560,
        MASTER_HOST_SYM = 561,
        MASTER_LOG_FILE_SYM = 562,
        MASTER_LOG_POS_SYM = 563,
        MASTER_PASSWORD_SYM = 564,
        MASTER_PORT_SYM = 565,
        MASTER_RETRY_COUNT_SYM = 566,
        MASTER_SERVER_ID_SYM = 567,
        MASTER_SSL_CAPATH_SYM = 568,
        MASTER_TLS_VERSION_SYM = 569,
        MASTER_SSL_CA_SYM = 570,
        MASTER_SSL_CERT_SYM = 571,
        MASTER_SSL_CIPHER_SYM = 572,
        MASTER_SSL_CRL_SYM = 573,
        MASTER_SSL_CRLPATH_SYM = 574,
        MASTER_SSL_KEY_SYM = 575,
        MASTER_SSL_SYM = 576,
        MASTER_SSL_VERIFY_SERVER_CERT_SYM = 577,
        MASTER_SYM = 578,
        MASTER_USER_SYM = 579,
        MASTER_HEARTBEAT_PERIOD_SYM = 580,
        MATCH = 581,
        MAX_CONNECTIONS_PER_HOUR = 582,
        MAX_QUERIES_PER_HOUR = 583,
        MAX_ROWS = 584,
        MAX_SIZE_SYM = 585,
        MAX_SYM = 586,
        MAX_UPDATES_PER_HOUR = 587,
        MAX_USER_CONNECTIONS_SYM = 588,
        MAX_VALUE_SYM = 589,
        MEDIUMBLOB_SYM = 590,
        MEDIUMINT_SYM = 591,
        MEDIUMTEXT_SYM = 592,
        MEDIUM_SYM = 593,
        MEMORY_SYM = 594,
        MERGE_SYM = 595,
        MESSAGE_TEXT_SYM = 596,
        MICROSECOND_SYM = 597,
        MIGRATE_SYM = 598,
        MINUTE_MICROSECOND_SYM = 599,
        MINUTE_SECOND_SYM = 600,
        MINUTE_SYM = 601,
        MIN_ROWS = 602,
        MIN_SYM = 603,
        MODE_SYM = 604,
        MODIFIES_SYM = 605,
        MODIFY_SYM = 606,
        MOD_SYM = 607,
        MONTH_SYM = 608,
        MULTILINESTRING_SYM = 609,
        MULTIPOINT_SYM = 610,
        MULTIPOLYGON_SYM = 611,
        MUTEX_SYM = 612,
        MYSQL_ERRNO_SYM = 613,
        NAMES_SYM = 614,
        NAME_SYM = 615,
        NATIONAL_SYM = 616,
        NATURAL = 617,
        NCHAR_STRING = 618,
        NCHAR_SYM = 619,
        NDBCLUSTER_SYM = 620,
        NE = 621,
        NEG = 622,
        NEVER_SYM = 623,
        NEW_SYM = 624,
        NEXT_SYM = 625,
        NODEGROUP_SYM = 626,
        NONE_SYM = 627,
        NOT2_SYM = 628,
        NOT_SYM = 629,
        NOW_SYM = 630,
        NO_SYM = 631,
        NO_WAIT_SYM = 632,
        NO_WRITE_TO_BINLOG = 633,
        NULL_SYM = 634,
        NUM = 635,
        NUMBER_SYM = 636,
        NUMERIC_SYM = 637,
        NVARCHAR_SYM = 638,
        OFFSET_SYM = 639,
        ON_SYM = 640,
        ONE_SYM = 641,
        ONLY_SYM = 642,
        OPEN_SYM = 643,
        OPTIMIZE = 644,
        OPTIMIZER_COSTS_SYM = 645,
        OPTIONS_SYM = 646,
        OPTION = 647,
        OPTIONALLY = 648,
        OR2_SYM = 649,
        ORDER_SYM = 650,
        OR_OR_SYM = 651,
        OR_SYM = 652,
        OUTER = 653,
        OUTFILE = 654,
        OUT_SYM = 655,
        OWNER_SYM = 656,
        PACK_KEYS_SYM = 657,
        PAGE_SYM = 658,
        PARAM_MARKER = 659,
        PARSER_SYM = 660,
        OBSOLETE_TOKEN_654 = 661,
        PARTIAL = 662,
        PARTITION_SYM = 663,
        PARTITIONS_SYM = 664,
        PARTITIONING_SYM = 665,
        PASSWORD = 666,
        PHASE_SYM = 667,
        PLUGIN_DIR_SYM = 668,
        PLUGIN_SYM = 669,
        PLUGINS_SYM = 670,
        POINT_SYM = 671,
        POLYGON_SYM = 672,
        PORT_SYM = 673,
        POSITION_SYM = 674,
        PRECEDES_SYM = 675,
        PRECISION = 676,
        PREPARE_SYM = 677,
        PRESERVE_SYM = 678,
        PREV_SYM = 679,
        PRIMARY_SYM = 680,
        PRIVILEGES = 681,
        PROCEDURE_SYM = 682,
        PROCESS = 683,
        PROCESSLIST_SYM = 684,
        PROFILE_SYM = 685,
        PROFILES_SYM = 686,
        PROXY_SYM = 687,
        PURGE = 688,
        QUARTER_SYM = 689,
        QUERY_SYM = 690,
        QUICK = 691,
        RANGE_SYM = 692,
        READS_SYM = 693,
        READ_ONLY_SYM = 694,
        READ_SYM = 695,
        READ_WRITE_SYM = 696,
        REAL_SYM = 697,
        REBUILD_SYM = 698,
        RECOVER_SYM = 699,
        OBSOLETE_TOKEN_693 = 700,
        REDO_BUFFER_SIZE_SYM = 701,
        REDUNDANT_SYM = 702,
        REFERENCES = 703,
        REGEXP = 704,
        RELAY = 705,
        RELAYLOG_SYM = 706,
        RELAY_LOG_FILE_SYM = 707,
        RELAY_LOG_POS_SYM = 708,
        RELAY_THREAD = 709,
        RELEASE_SYM = 710,
        RELOAD = 711,
        REMOVE_SYM = 712,
        RENAME = 713,
        REORGANIZE_SYM = 714,
        REPAIR = 715,
        REPEATABLE_SYM = 716,
        REPEAT_SYM = 717,
        REPLACE_SYM = 718,
        REPLICATION = 719,
        REPLICATE_DO_DB = 720,
        REPLICATE_IGNORE_DB = 721,
        REPLICATE_DO_TABLE = 722,
        REPLICATE_IGNORE_TABLE = 723,
        REPLICATE_WILD_DO_TABLE = 724,
        REPLICATE_WILD_IGNORE_TABLE = 725,
        REPLICATE_REWRITE_DB = 726,
        REQUIRE_SYM = 727,
        RESET_SYM = 728,
        RESIGNAL_SYM = 729,
        RESOURCES = 730,
        RESTORE_SYM = 731,
        RESTRICT = 732,
        RESUME_SYM = 733,
        RETURNED_SQLSTATE_SYM = 734,
        RETURNS_SYM = 735,
        RETURN_SYM = 736,
        REVERSE_SYM = 737,
        REVOKE = 738,
        RIGHT = 739,
        ROLLBACK_SYM = 740,
        ROLLUP_SYM = 741,
        ROTATE_SYM = 742,
        ROUTINE_SYM = 743,
        ROWS_SYM = 744,
        ROW_FORMAT_SYM = 745,
        ROW_SYM = 746,
        ROW_COUNT_SYM = 747,
        RTREE_SYM = 748,
        SAVEPOINT_SYM = 749,
        SCHEDULE_SYM = 750,
        SCHEMA_NAME_SYM = 751,
        SCHEMA = 752,
        SECOND_MICROSECOND_SYM = 753,
        SECOND_SYM = 754,
        SECURITY_SYM = 755,
        SELECT_SYM = 756,
        SENSITIVE_SYM = 757,
        SEPARATOR_SYM = 758,
        SERIALIZABLE_SYM = 759,
        SERIAL_SYM = 760,
        SESSION_SYM = 761,
        SERVER_SYM = 762,
        OBSOLETE_TOKEN_755 = 763,
        SET = 764,
        SET_VAR = 765,
        SHARE_SYM = 766,
        SHARES_SYM = 767,
        SHIFT_LEFT = 768,
        SHIFT_RIGHT = 769,
        SHOW = 770,
        SHUTDOWN = 771,
        SIGNAL_SYM = 772,
        SIGNED_SYM = 773,
        SIMPLE_SYM = 774,
        SLAVE = 775,
        SLOW = 776,
        SMALLINT_SYM = 777,
        SNAPSHOT_SYM = 778,
        SOCKET_SYM = 779,
        SONAME_SYM = 780,
        SOUNDS_SYM = 781,
        SOURCE_SYM = 782,
        SPATIAL_SYM = 783,
        SPECIFIC_SYM = 784,
        SQLEXCEPTION_SYM = 785,
        SQLSTATE_SYM = 786,
        SQLWARNING_SYM = 787,
        SQL_AFTER_GTIDS = 788,
        SQL_AFTER_MTS_GAPS = 789,
        SQL_BEFORE_GTIDS = 790,
        SQL_BIG_RESULT = 791,
        SQL_BUFFER_RESULT = 792,
        OBSOLETE_TOKEN_784 = 793,
        SQL_CALC_FOUND_ROWS = 794,
        SQL_NO_CACHE_SYM = 795,
        SQL_SMALL_RESULT = 796,
        SQL_SYM = 797,
        SQL_THREAD = 798,
        SSL_SYM = 799,
        STACKED_SYM = 800,
        STARTING = 801,
        STARTS_SYM = 802,
        START_SYM = 803,
        STATS_AUTO_RECALC_SYM = 804,
        STATS_PERSISTENT_SYM = 805,
        STATS_SAMPLE_PAGES_SYM = 806,
        STATUS_SYM = 807,
        STDDEV_SAMP_SYM = 808,
        STD_SYM = 809,
        STOP_SYM = 810,
        STORAGE_SYM = 811,
        STORED_SYM = 812,
        STRAIGHT_JOIN = 813,
        STRING_SYM = 814,
        SUBCLASS_ORIGIN_SYM = 815,
        SUBDATE_SYM = 816,
        SUBJECT_SYM = 817,
        SUBPARTITIONS_SYM = 818,
        SUBPARTITION_SYM = 819,
        SUBSTRING = 820,
        SUM_SYM = 821,
        SUPER_SYM = 822,
        SUSPEND_SYM = 823,
        SWAPS_SYM = 824,
        SWITCHES_SYM = 825,
        SYSDATE = 826,
        TABLES = 827,
        VIEWS = 828,
        TABLESPACE_SYM = 829,
        OBSOLETE_TOKEN_820 = 830,
        TABLE_SYM = 831,
        TABLE_CHECKSUM_SYM = 832,
        TABLE_NAME_SYM = 833,
        TEMPORARY = 834,
        TEMPTABLE_SYM = 835,
        TERMINATED = 836,
        TEXT_STRING = 837,
        TEXT_SYM = 838,
        THAN_SYM = 839,
        THEN_SYM = 840,
        TIMESTAMP_SYM = 841,
        TIMESTAMP_ADD = 842,
        TIMESTAMP_DIFF = 843,
        TIME_SYM = 844,
        TINYBLOB_SYM = 845,
        TINYINT_SYM = 846,
        TINYTEXT_SYN = 847,
        TO_SYM = 848,
        TRAILING = 849,
        TRANSACTION_SYM = 850,
        TRIGGERS_SYM = 851,
        TRIGGER_SYM = 852,
        TRIM = 853,
        TRUE_SYM = 854,
        TRUNCATE_SYM = 855,
        TYPES_SYM = 856,
        TYPE_SYM = 857,
        OBSOLETE_TOKEN_848 = 858,
        ULONGLONG_NUM = 859,
        UNCOMMITTED_SYM = 860,
        UNDEFINED_SYM = 861,
        UNDERSCORE_CHARSET = 862,
        UNDOFILE_SYM = 863,
        UNDO_BUFFER_SIZE_SYM = 864,
        UNDO_SYM = 865,
        UNICODE_SYM = 866,
        UNINSTALL_SYM = 867,
        UNION_SYM = 868,
        UNIQUE_SYM = 869,
        UNKNOWN_SYM = 870,
        UNLOCK_SYM = 871,
        UNSIGNED_SYM = 872,
        UNTIL_SYM = 873,
        UPDATE_SYM = 874,
        UPGRADE_SYM = 875,
        USAGE = 876,
        USER = 877,
        USE_FRM = 878,
        USE_SYM = 879,
        USING = 880,
        UTC_DATE_SYM = 881,
        UTC_TIMESTAMP_SYM = 882,
        UTC_TIME_SYM = 883,
        VALIDATION_SYM = 884,
        VALUES = 885,
        VALUE_SYM = 886,
        VARBINARY_SYM = 887,
        VARCHAR_SYM = 888,
        VARIABLES = 889,
        VARIANCE_SYM = 890,
        VARYING = 891,
        VAR_SAMP_SYM = 892,
        VERSION_SYM = 893,
        VIEW_SYM = 894,
        VIRTUAL_SYM = 895,
        WAIT_SYM = 896,
        WARNINGS = 897,
        WEEK_SYM = 898,
        WEIGHT_STRING_SYM = 899,
        WHEN_SYM = 900,
        WHERE = 901,
        WHILE_SYM = 902,
        WITH = 903,
        OBSOLETE_TOKEN_893 = 904,
        WITH_ROLLUP_SYM = 905,
        WITHOUT_SYM = 906,
        WORK_SYM = 907,
        WRAPPER_SYM = 908,
        WRITE_SYM = 909,
        X509_SYM = 910,
        XA_SYM = 911,
        XID_SYM = 912,
        XML_SYM = 913,
        XOR = 914,
        YEAR_MONTH_SYM = 915,
        YEAR_SYM = 916,
        ZEROFILL_SYM = 917,
        EXPLAIN_SYM = 918,
        TREE_SYM = 919,
        TRADITIONAL_SYM = 920,
        JSON_UNQUOTED_SEPARATOR_SYM = 921,
        PERSIST_SYM = 922,
        ROLE_SYM = 923,
        ADMIN_SYM = 924,
        INVISIBLE_SYM = 925,
        VISIBLE_SYM = 926,
        EXCEPT_SYM = 927,
        COMPONENT_SYM = 928,
        RECURSIVE_SYM = 929,
        GRAMMAR_SELECTOR_EXPR = 930,
        GRAMMAR_SELECTOR_GCOL = 931,
        GRAMMAR_SELECTOR_PART = 932,
        GRAMMAR_SELECTOR_CTE = 933,
        JSON_OBJECTAGG = 934,
        JSON_ARRAYAGG = 935,
        OF_SYM = 936,
        SKIP_SYM = 937,
        LOCKED_SYM = 938,
        NOWAIT_SYM = 939,
        GROUPING_SYM = 940,
        PERSIST_ONLY_SYM = 941,
        HISTOGRAM_SYM = 942,
        BUCKETS_SYM = 943,
        OBSOLETE_TOKEN_930 = 944,
        CLONE_SYM = 945,
        CUME_DIST_SYM = 946,
        DENSE_RANK_SYM = 947,
        EXCLUDE_SYM = 948,
        FIRST_VALUE_SYM = 949,
        FOLLOWING_SYM = 950,
        GROUPS_SYM = 951,
        LAG_SYM = 952,
        LAST_VALUE_SYM = 953,
        LEAD_SYM = 954,
        NTH_VALUE_SYM = 955,
        NTILE_SYM = 956,
        NULLS_SYM = 957,
        OTHERS_SYM = 958,
        OVER_SYM = 959,
        PERCENT_RANK_SYM = 960,
        PRECEDING_SYM = 961,
        RANK_SYM = 962,
        RESPECT_SYM = 963,
        ROW_NUMBER_SYM = 964,
        TIES_SYM = 965,
        UNBOUNDED_SYM = 966,
        WINDOW_SYM = 967,
        EMPTY_SYM = 968,
        JSON_TABLE_SYM = 969,
        NESTED_SYM = 970,
        ORDINALITY_SYM = 971,
        PATH_SYM = 972,
        HISTORY_SYM = 973,
        REUSE_SYM = 974,
        SRID_SYM = 975,
        THREAD_PRIORITY_SYM = 976,
        RESOURCE_SYM = 977,
        SYSTEM_SYM = 978,
        VCPU_SYM = 979,
        MASTER_PUBLIC_KEY_PATH_SYM = 980,
        GET_MASTER_PUBLIC_KEY_SYM = 981,
        RESTART_SYM = 982,
        DEFINITION_SYM = 983,
        DESCRIPTION_SYM = 984,
        ORGANIZATION_SYM = 985,
        REFERENCE_SYM = 986,
        ACTIVE_SYM = 987,
        INACTIVE_SYM = 988,
        LATERAL_SYM = 989,
        OPTIONAL_SYM = 990,
        SECONDARY_SYM = 991,
        SECONDARY_ENGINE_SYM = 992,
        SECONDARY_LOAD_SYM = 993,
        SECONDARY_UNLOAD_SYM = 994,
        RETAIN_SYM = 995,
        OLD_SYM = 996,
        ENFORCED_SYM = 997,
        OJ_SYM = 998,
        NETWORK_NAMESPACE_SYM = 999,
        ADD_SYM = 1000,
        MINUS_SYM = 1001,
        CONDITIONLESS_JOIN = 1002,
        SUBQUERY_AS_EXPR = 1003,
        EMPTY_FROM_CLAUSE = 1004
      };
    };

    /// (External) token type, as returned by yylex.
    typedef token::yytokentype token_type;

    /// Symbol type: an internal symbol number.
    typedef int symbol_number_type;

    /// The symbol type number to denote an empty symbol.
    enum { empty_symbol = -2 };

    /// Internal symbol number for tokens (subsumed by symbol_number_type).
    typedef short token_number_type;

    /// A complete symbol.
    ///
    /// Expects its Base type to provide access to the symbol type
    /// via type_get ().
    ///
    /// Provide access to semantic value and location.
    template <typename Base>
    struct basic_symbol : Base
    {
      /// Alias to Base.
      typedef Base super_type;

      /// Default constructor.
      basic_symbol ()
        : value ()
        , location ()
      {}

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      basic_symbol (basic_symbol&& that);
#endif

      /// Copy constructor.
      basic_symbol (const basic_symbol& that);

      /// Constructor for valueless symbols, and symbols from each type.
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, location_type&& l)
        : Base (t)
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const location_type& l)
        : Base (t)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, AbstractCommandPointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const AbstractCommandPointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, AccountSPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const AccountSPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, AdminStmtStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const AdminStmtStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, CastType&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const CastType& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, ColAttrList&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const ColAttrList& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, CreateTableOptions&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const CreateTableOptions& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, DeleteStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const DeleteStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Expression&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Expression& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Field_def_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Field_def_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Field_option&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Field_option& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Field_separators&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Field_separators& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, FromPartStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const FromPartStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, GroupbyStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const GroupbyStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, InsertStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const InsertStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, JoinStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const JoinStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, JoinType&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const JoinType& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, LimitStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const LimitStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Line_separators&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Line_separators& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Literal&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Literal& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, LoadDataStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const LoadDataStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, On_duplicate&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const On_duplicate& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, OrderItem&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const OrderItem& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, OrderbyStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const OrderbyStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PT_ColumnType_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PT_ColumnType_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PT_column_attr_base_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PT_column_attr_base_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PT_option_value_following_option_type_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PT_option_value_following_option_type_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartDef&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartDef& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartDefList&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartDefList& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartTypeDef&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartTypeDef& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartValueItem&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartValueItem& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartValueItemsSPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartValueItemsSPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartValuesSPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartValuesSPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PartitionStructureSPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PartitionStructureSPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Precision_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Precision_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PrepareSrcPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PrepareSrcPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, PreparedStmtStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const PreparedStmtStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SHOW_CMD&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SHOW_CMD& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SQLIdentPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SQLIdentPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SelectPartStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SelectPartStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SelectStructurePointer&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SelectStructurePointer& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SetOperationType&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SetOperationType& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, SetStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const SetStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, ShowStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const ShowStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, Show_cmd_type&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const Show_cmd_type& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, TABLE_LIST&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const TABLE_LIST& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, TableElementDescriptionPtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const TableElementDescriptionPtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, TransactionStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const TransactionStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, UpdateStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const UpdateStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, VALUES&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const VALUES& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, VariableStructurePtr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const VariableStructurePtr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, WildOrWhere_ptr&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const WildOrWhere_ptr& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, bool&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const bool& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, enum_filetype&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const enum_filetype& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, enum_var_type&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const enum_var_type& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, enum_yes_no_unknown&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const enum_yes_no_unknown& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, int&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const int& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, keytype&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const keytype& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, pair< BiaodashiPointer, BiaodashiPointer >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const pair< BiaodashiPointer, BiaodashiPointer >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, pair< EXPR_LIST, SelectStructurePointer >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const pair< EXPR_LIST, SelectStructurePointer >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, pair< aries::EncodeType, string >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const pair< aries::EncodeType, string >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, pair< shared_ptr<BasicRel>, vector< string > >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const pair< shared_ptr<BasicRel>, vector< string > >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, pair< vector< BiaodashiPointer >, vector< BiaodashiPointer > > && v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const pair< vector< BiaodashiPointer >, vector< BiaodashiPointer > > & v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, shared_ptr< vector< AccountSPtr > >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const shared_ptr< vector< AccountSPtr > >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, shared_ptr<BasicRel>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const shared_ptr<BasicRel>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, shared_ptr<vector<SetStructurePtr>>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const shared_ptr<vector<SetStructurePtr>>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, shared_ptr<vector<TableElementDescriptionPtr>>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const shared_ptr<vector<TableElementDescriptionPtr>>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, string&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const string& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, thr_lock_type&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const thr_lock_type& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, tuple<Expression, string>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const tuple<Expression, string>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, ulong&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const ulong& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, ulonglong&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const ulonglong& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector< PartValueItemsSPtr >&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector< PartValueItemsSPtr >& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector<Expression>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector<Expression>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector<JoinStructurePointer>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector<JoinStructurePointer>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector<OrderItem>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector<OrderItem>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector<string>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector<string>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif
#if 201103L <= YY_CPLUSPLUS
      basic_symbol (typename Base::kind_type t, vector<tuple<Expression, Expression>>&& v, location_type&& l)
        : Base (t)
        , value (std::move (v))
        , location (std::move (l))
      {}
#else
      basic_symbol (typename Base::kind_type t, const vector<tuple<Expression, Expression>>& v, const location_type& l)
        : Base (t)
        , value (v)
        , location (l)
      {}
#endif

      /// Destroy the symbol.
      ~basic_symbol ()
      {
        clear ();
      }

      /// Destroy contents, and record that is empty.
      void clear ()
      {
        // User destructor.
        symbol_number_type yytype = this->type_get ();
        basic_symbol<Base>& yysym = *this;
        (void) yysym;
        switch (yytype)
        {
       default:
          break;
        }

        // Type destructor.
switch (yytype)
    {
      case 775: // drop_table_stmt
      case 776: // drop_user_stmt
      case 777: // drop_view_stmt
      case 778: // drop_database_stmt
      case 827: // create
      case 871: // create_table_stmt
      case 875: // view_or_trigger_or_sp_or_event
      case 877: // no_definer_tail
      case 946: // view_tail
      case 1328: // use
        value.template destroy< AbstractCommandPointer > ();
        break;

      case 837: // user
      case 839: // create_user
        value.template destroy< AccountSPtr > ();
        break;

      case 1329: // kill
      case 1336: // shutdown_stmt
        value.template destroy< AdminStmtStructurePtr > ();
        break;

      case 1110: // cast_type
        value.template destroy< CastType > ();
        break;

      case 1204: // opt_column_attribute_list
      case 1205: // column_attribute_list
        value.template destroy< ColAttrList > ();
        break;

      case 990: // opt_create_table_options_etc
        value.template destroy< CreateTableOptions > ();
        break;

      case 1320: // delete_stmt
        value.template destroy< DeleteStructurePtr > ();
        break;

      case 796: // expr
      case 797: // bool_pri
      case 798: // predicate
      case 799: // bit_expr
      case 806: // simple_expr
      case 807: // case_expr
      case 808: // function_call_keyword
      case 809: // function_call_nonkeyword
      case 810: // function_call_conflict
      case 811: // function_call_generic
      case 814: // udf_expr
      case 815: // set_function_specification
      case 817: // sum_expr
      case 823: // row_subquery
      case 1029: // param_marker
      case 1057: // variable
      case 1109: // set_expr_or_default
      case 1111: // in_sum_expr
      case 1117: // opt_expr
      case 1118: // opt_else
      case 1145: // opt_where_clause
      case 1146: // opt_where_clause_expr
      case 1147: // opt_having_clause
      case 1170: // table_wild
      case 1172: // grouping_expr
      case 1200: // func_datetime_precision
      case 1210: // now_or_signed_literal
      case 1211: // now
      case 1311: // expr_or_default
        value.template destroy< Expression > ();
        break;

      case 1185: // field_def
        value.template destroy< Field_def_ptr > ();
        break;

      case 868: // field_options
      case 869: // field_opt_list
      case 870: // field_option
        value.template destroy< Field_option > ();
        break;

      case 1343: // opt_field_term
      case 1344: // field_term_list
      case 1345: // field_term
        value.template destroy< Field_separators > ();
        break;

      case 785: // opt_from_clause
      case 786: // from_clause
        value.template destroy< FromPartStructurePointer > ();
        break;

      case 1157: // opt_group_clause
        value.template destroy< GroupbyStructurePointer > ();
        break;

      case 1299: // insert_stmt
        value.template destroy< InsertStructurePtr > ();
        break;

      case 1120: // table_reference
      case 1121: // joined_table
      case 1129: // table_factor
      case 1131: // single_table_parens
      case 1132: // single_table
      case 1133: // joined_table_parens
      case 1134: // derived_table
        value.template destroy< JoinStructurePointer > ();
        break;

      case 1122: // natural_join_type
      case 1123: // inner_join_type
      case 1124: // outer_join_type
        value.template destroy< JoinType > ();
        break;

      case 1165: // opt_limit_clause
      case 1166: // limit_clause
      case 1167: // limit_options
      case 1321: // opt_simple_limit
        value.template destroy< LimitStructurePointer > ();
        break;

      case 1346: // opt_line_term
      case 1347: // line_term_list
      case 1348: // line_term
        value.template destroy< Line_separators > ();
        break;

      case 1030: // signed_literal
      case 1031: // literal
      case 1032: // NUM_literal
        value.template destroy< Literal > ();
        break;

      case 1337: // load_stmt
        value.template destroy< LoadDataStructurePtr > ();
        break;

      case 844: // duplicate
      case 1341: // opt_duplicate
        value.template destroy< On_duplicate > ();
        break;

      case 1171: // order_expr
        value.template destroy< OrderItem > ();
        break;

      case 1160: // opt_order_clause
      case 1161: // order_clause
        value.template destroy< OrderbyStructurePointer > ();
        break;

      case 1188: // type
        value.template destroy< PT_ColumnType_ptr > ();
        break;

      case 1206: // column_attribute
        value.template destroy< PT_column_attr_base_ptr > ();
        break;

      case 1106: // option_value_following_option_type
        value.template destroy< PT_option_value_following_option_type_ptr > ();
        break;

      case 1005: // part_definition
        value.template destroy< PartDef > ();
        break;

      case 1003: // opt_part_defs
      case 1004: // part_def_list
        value.template destroy< PartDefList > ();
        break;

      case 995: // part_type_def
        value.template destroy< PartTypeDef > ();
        break;

      case 1012: // part_value_item
        value.template destroy< PartValueItem > ();
        break;

      case 1007: // part_func_max
      case 1010: // part_value_item_list_paren
      case 1011: // part_value_item_list
        value.template destroy< PartValueItemsSPtr > ();
        break;

      case 1006: // opt_part_values
        value.template destroy< PartValuesSPtr > ();
        break;

      case 991: // opt_create_partitioning_etc
      case 994: // partition_clause
        value.template destroy< PartitionStructureSPtr > ();
        break;

      case 1197: // float_options
      case 1198: // precision
      case 1203: // opt_precision
        value.template destroy< Precision_ptr > ();
        break;

      case 1020: // prepare_src
        value.template destroy< PrepareSrcPtr > ();
        break;

      case 1019: // prepare
      case 1021: // execute
      case 1025: // deallocate
        value.template destroy< PreparedStmtStructurePtr > ();
        break;

      case 1259: // show_engine_param
        value.template destroy< SHOW_CMD > ();
        break;

      case 1169: // insert_ident
      case 1173: // simple_ident
      case 1174: // simple_ident_nospvar
      case 1175: // simple_ident_q
        value.template destroy< SQLIdentPtr > ();
        break;

      case 792: // select_item_list
        value.template destroy< SelectPartStructurePointer > ();
        break;

      case 779: // select_stmt
      case 780: // query_expression
      case 781: // query_expression_body
      case 782: // query_expression_parens
      case 783: // query_primary
      case 784: // query_specification
      case 824: // table_subquery
      case 825: // subquery
      case 947: // view_select
      case 1298: // query_expression_or_parens
      case 1317: // explain_stmt
        value.template destroy< SelectStructurePointer > ();
        break;

      case 822: // union_option
        value.template destroy< SetOperationType > ();
        break;

      case 1101: // option_value
      case 1107: // option_value_no_option_type
        value.template destroy< SetStructurePtr > ();
        break;

      case 1256: // describe_stmt
      case 1257: // show
      case 1258: // show_param
        value.template destroy< ShowStructurePtr > ();
        break;

      case 1267: // opt_show_cmd_type
        value.template destroy< Show_cmd_type > ();
        break;

      case 856: // opt_table_list
      case 857: // table_list
      case 1325: // table_alias_ref_list
        value.template destroy< TABLE_LIST > ();
        break;

      case 859: // table_element
      case 860: // column_def
      case 862: // table_constraint_def
        value.template destroy< TableElementDescriptionPtr > ();
        break;

      case 1358: // start
      case 1362: // begin_stmt
      case 1367: // commit
      case 1368: // rollback
        value.template destroy< TransactionStructurePtr > ();
        break;

      case 1313: // update_stmt
        value.template destroy< UpdateStructurePtr > ();
        break;

      case 1308: // row_value
      case 1309: // opt_values
      case 1310: // values
        value.template destroy< VALUES > ();
        break;

      case 1058: // variable_aux
        value.template destroy< VariableStructurePtr > ();
        break;

      case 1261: // opt_wild_or_where
      case 1262: // opt_wild_or_where_for_show
        value.template destroy< WildOrWhere_ptr > ();
        break;

      case 818: // opt_distinct
      case 845: // opt_if_not_exists
      case 865: // opt_not
      case 872: // if_exists
      case 873: // opt_temporary
      case 996: // opt_linear
      case 1080: // visibility
      case 1265: // opt_full
      case 1266: // opt_extended
      case 1301: // opt_ignore
      case 1339: // opt_local
        value.template destroy< bool > ();
        break;

      case 1338: // data_or_xml
        value.template destroy< enum_filetype > ();
        break;

      case 1102: // option_type
      case 1103: // opt_var_type
      case 1104: // opt_var_ident_type
      case 1105: // opt_set_var_ident_type
        value.template destroy< enum_var_type > ();
        break;

      case 1364: // opt_chain
      case 1365: // opt_release
        value.template destroy< enum_yes_no_unknown > ();
        break;

      case 789: // select_options
      case 790: // select_option_list
      case 791: // select_option
      case 804: // comp_op
      case 826: // query_spec_option
      case 1330: // kill_option
        value.template destroy< int > ();
        break;

      case 1064: // constraint_key_type
        value.template destroy< keytype > ();
        break;

      case 1315: // update_elem
        value.template destroy< pair< BiaodashiPointer, BiaodashiPointer > > ();
        break;

      case 1303: // insert_query_expression
        value.template destroy< pair< EXPR_LIST, SelectStructurePointer > > ();
        break;

      case 1207: // encode_type
        value.template destroy< pair< aries::EncodeType, string > > ();
        break;

      case 861: // opt_references
      case 1060: // references
        value.template destroy< pair< shared_ptr<BasicRel>, vector< string > > > ();
        break;

      case 1312: // opt_insert_update_list
      case 1314: // update_list
        value.template destroy< pair< vector< BiaodashiPointer >, vector< BiaodashiPointer > >  > ();
        break;

      case 838: // user_list
      case 840: // create_user_list
        value.template destroy< shared_ptr< vector< AccountSPtr > > > ();
        break;

      case 1176: // table_ident
      case 1242: // sp_name
      case 1326: // table_ident_opt_wild
        value.template destroy< shared_ptr<BasicRel> > ();
        break;

      case 1098: // option_value_list_continued
      case 1099: // start_option_value_list_following_option_type
      case 1100: // option_value_list
      case 1286: // set
      case 1287: // start_option_value_list
        value.template destroy< shared_ptr<vector<SetStructurePtr>> > ();
        break;

      case 858: // table_element_list
        value.template destroy< shared_ptr<vector<TableElementDescriptionPtr>> > ();
        break;

      case 5: // ACCOUNT_SYM
      case 6: // ACTION
      case 8: // ADDDATE_SYM
      case 9: // AFTER_SYM
      case 10: // AGAINST
      case 11: // AGGREGATE_SYM
      case 12: // ALGORITHM_SYM
      case 15: // ALWAYS_SYM
      case 20: // ANY_SYM
      case 22: // ASC
      case 23: // ASCII_SYM
      case 25: // AT_SYM
      case 26: // AUTOEXTEND_SIZE_SYM
      case 27: // AUTO_INC
      case 28: // AVG_ROW_LENGTH
      case 29: // AVG_SYM
      case 30: // BACKUP_SYM
      case 32: // BEGIN_SYM
      case 36: // BINLOG_SYM
      case 40: // BIT_SYM
      case 43: // BLOCK_SYM
      case 44: // BOOLEAN_SYM
      case 45: // BOOL_SYM
      case 47: // BTREE_SYM
      case 49: // BYTE_SYM
      case 50: // CACHE_SYM
      case 53: // CASCADED
      case 56: // CATALOG_NAME_SYM
      case 57: // CHAIN_SYM
      case 59: // CHANGED
      case 60: // CHANNEL_SYM
      case 61: // CHARSET
      case 63: // CHECKSUM_SYM
      case 65: // CIPHER_SYM
      case 66: // CLASS_ORIGIN_SYM
      case 67: // CLIENT_SYM
      case 68: // CLOSE_SYM
      case 69: // COALESCE
      case 70: // CODE_SYM
      case 72: // COLLATION_SYM
      case 73: // COLUMNS
      case 75: // COLUMN_FORMAT_SYM
      case 76: // COLUMN_NAME_SYM
      case 77: // COMMENT_SYM
      case 78: // COMMITTED_SYM
      case 79: // COMMIT_SYM
      case 80: // COMPACT_SYM
      case 81: // COMPLETION_SYM
      case 82: // COMPRESSED_SYM
      case 83: // COMPRESSION_SYM
      case 84: // ENCRYPTION_SYM
      case 85: // CONCURRENT
      case 88: // CONNECTION_SYM
      case 89: // CONSISTENT_SYM
      case 91: // CONSTRAINT_CATALOG_SYM
      case 92: // CONSTRAINT_NAME_SYM
      case 93: // CONSTRAINT_SCHEMA_SYM
      case 94: // CONTAINS_SYM
      case 95: // CONTEXT_SYM
      case 99: // CPU_SYM
      case 100: // CREATE
      case 104: // CURRENT_SYM
      case 107: // CURSOR_NAME_SYM
      case 111: // DATAFILE_SYM
      case 112: // DATA_SYM
      case 113: // DATETIME_SYM
      case 116: // DATE_SYM
      case 121: // DAY_SYM
      case 122: // DEALLOCATE_SYM
      case 123: // DECIMAL_NUM
      case 128: // DEFAULT_AUTH_SYM
      case 129: // DEFINER_SYM
      case 131: // DELAY_KEY_WRITE_SYM
      case 133: // DESC
      case 137: // DIAGNOSTICS_SYM
      case 138: // BYTEDICT_SYM
      case 139: // SHORTDICT_SYM
      case 140: // INTDICT_SYM
      case 142: // DIRECTORY_SYM
      case 143: // DISABLE_SYM
      case 144: // DISCARD_SYM
      case 145: // DISK_SYM
      case 149: // DO_SYM
      case 152: // DUMPFILE
      case 153: // DUPLICATE_SYM
      case 154: // DYNAMIC_SYM
      case 158: // ENABLE_SYM
      case 161: // END
      case 162: // ENDS_SYM
      case 163: // ENGINES_SYM
      case 164: // ENGINE_SYM
      case 165: // ENUM_SYM
      case 168: // ERROR_SYM
      case 169: // ERRORS
      case 171: // ESCAPE_SYM
      case 172: // EVENTS_SYM
      case 173: // EVENT_SYM
      case 174: // EVERY_SYM
      case 175: // EXCHANGE_SYM
      case 176: // EXECUTE_SYM
      case 179: // EXPANSION_SYM
      case 180: // EXPIRE_SYM
      case 181: // EXPORT_SYM
      case 182: // EXTENDED_SYM
      case 183: // EXTENT_SIZE_SYM
      case 186: // FAST_SYM
      case 187: // FAULTS_SYM
      case 189: // FILE_SYM
      case 190: // FILE_BLOCK_SIZE_SYM
      case 191: // FILTER_SYM
      case 192: // FIRST_SYM
      case 193: // FIXED_SYM
      case 194: // FLOAT_NUM
      case 196: // FLUSH_SYM
      case 197: // FOLLOWS_SYM
      case 201: // FORMAT_SYM
      case 202: // FOUND_SYM
      case 204: // FULL
      case 208: // GENERAL
      case 210: // GROUP_REPLICATION
      case 211: // GEOMETRYCOLLECTION_SYM
      case 212: // GEOMETRY_SYM
      case 213: // GET_FORMAT
      case 215: // GLOBAL_SYM
      case 217: // GRANTS
      case 221: // HANDLER_SYM
      case 222: // HASH_SYM
      case 224: // HELP_SYM
      case 225: // HEX_NUM
      case 227: // HOST_SYM
      case 228: // HOSTS_SYM
      case 232: // HOUR_SYM
      case 233: // IDENT
      case 234: // IDENTIFIED_SYM
      case 238: // IGNORE_SERVER_IDS_SYM
      case 239: // IMPORT
      case 240: // INDEXES
      case 243: // INITIAL_SIZE_SYM
      case 248: // INSERT_METHOD
      case 249: // INSTANCE_SYM
      case 250: // INSTALL_SYM
      case 251: // INTERVAL_SYM
      case 255: // INVOKER_SYM
      case 259: // IO_SYM
      case 260: // IPC_SYM
      case 262: // ISOLATION
      case 263: // ISSUER_SYM
      case 267: // JSON_SYM
      case 269: // KEY_BLOCK_SIZE
      case 272: // LANGUAGE_SYM
      case 273: // LAST_SYM
      case 276: // LEAVES
      case 279: // LESS_SYM
      case 280: // LEVEL_SYM
      case 281: // LEX_HOSTNAME
      case 286: // LINESTRING_SYM
      case 287: // LIST_SYM
      case 289: // LOCAL_SYM
      case 291: // LOCKS_SYM
      case 293: // LOGFILE_SYM
      case 294: // LOGS_SYM
      case 297: // LONG_NUM
      case 302: // MASTER_AUTO_POSITION_SYM
      case 304: // MASTER_CONNECT_RETRY_SYM
      case 305: // MASTER_DELAY_SYM
      case 306: // MASTER_HOST_SYM
      case 307: // MASTER_LOG_FILE_SYM
      case 308: // MASTER_LOG_POS_SYM
      case 309: // MASTER_PASSWORD_SYM
      case 310: // MASTER_PORT_SYM
      case 311: // MASTER_RETRY_COUNT_SYM
      case 312: // MASTER_SERVER_ID_SYM
      case 313: // MASTER_SSL_CAPATH_SYM
      case 314: // MASTER_TLS_VERSION_SYM
      case 315: // MASTER_SSL_CA_SYM
      case 316: // MASTER_SSL_CERT_SYM
      case 317: // MASTER_SSL_CIPHER_SYM
      case 318: // MASTER_SSL_CRL_SYM
      case 319: // MASTER_SSL_CRLPATH_SYM
      case 320: // MASTER_SSL_KEY_SYM
      case 321: // MASTER_SSL_SYM
      case 323: // MASTER_SYM
      case 324: // MASTER_USER_SYM
      case 325: // MASTER_HEARTBEAT_PERIOD_SYM
      case 327: // MAX_CONNECTIONS_PER_HOUR
      case 328: // MAX_QUERIES_PER_HOUR
      case 329: // MAX_ROWS
      case 330: // MAX_SIZE_SYM
      case 332: // MAX_UPDATES_PER_HOUR
      case 333: // MAX_USER_CONNECTIONS_SYM
      case 338: // MEDIUM_SYM
      case 339: // MEMORY_SYM
      case 340: // MERGE_SYM
      case 341: // MESSAGE_TEXT_SYM
      case 342: // MICROSECOND_SYM
      case 343: // MIGRATE_SYM
      case 346: // MINUTE_SYM
      case 347: // MIN_ROWS
      case 349: // MODE_SYM
      case 351: // MODIFY_SYM
      case 353: // MONTH_SYM
      case 354: // MULTILINESTRING_SYM
      case 355: // MULTIPOINT_SYM
      case 356: // MULTIPOLYGON_SYM
      case 357: // MUTEX_SYM
      case 358: // MYSQL_ERRNO_SYM
      case 359: // NAMES_SYM
      case 360: // NAME_SYM
      case 361: // NATIONAL_SYM
      case 364: // NCHAR_SYM
      case 365: // NDBCLUSTER_SYM
      case 368: // NEVER_SYM
      case 369: // NEW_SYM
      case 370: // NEXT_SYM
      case 371: // NODEGROUP_SYM
      case 372: // NONE_SYM
      case 376: // NO_SYM
      case 377: // NO_WAIT_SYM
      case 380: // NUM
      case 381: // NUMBER_SYM
      case 383: // NVARCHAR_SYM
      case 384: // OFFSET_SYM
      case 386: // ONE_SYM
      case 387: // ONLY_SYM
      case 388: // OPEN_SYM
      case 391: // OPTIONS_SYM
      case 401: // OWNER_SYM
      case 402: // PACK_KEYS_SYM
      case 403: // PAGE_SYM
      case 405: // PARSER_SYM
      case 407: // PARTIAL
      case 409: // PARTITIONS_SYM
      case 410: // PARTITIONING_SYM
      case 411: // PASSWORD
      case 412: // PHASE_SYM
      case 413: // PLUGIN_DIR_SYM
      case 414: // PLUGIN_SYM
      case 415: // PLUGINS_SYM
      case 416: // "."
      case 417: // POLYGON_SYM
      case 418: // PORT_SYM
      case 420: // PRECEDES_SYM
      case 422: // PREPARE_SYM
      case 423: // PRESERVE_SYM
      case 424: // PREV_SYM
      case 426: // PRIVILEGES
      case 428: // PROCESS
      case 429: // PROCESSLIST_SYM
      case 430: // PROFILE_SYM
      case 431: // PROFILES_SYM
      case 432: // PROXY_SYM
      case 434: // QUARTER_SYM
      case 435: // QUERY_SYM
      case 436: // QUICK
      case 439: // READ_ONLY_SYM
      case 443: // REBUILD_SYM
      case 444: // RECOVER_SYM
      case 446: // REDO_BUFFER_SIZE_SYM
      case 447: // REDUNDANT_SYM
      case 450: // RELAY
      case 451: // RELAYLOG_SYM
      case 452: // RELAY_LOG_FILE_SYM
      case 453: // RELAY_LOG_POS_SYM
      case 454: // RELAY_THREAD
      case 456: // RELOAD
      case 457: // REMOVE_SYM
      case 459: // REORGANIZE_SYM
      case 460: // REPAIR
      case 461: // REPEATABLE_SYM
      case 464: // REPLICATION
      case 465: // REPLICATE_DO_DB
      case 466: // REPLICATE_IGNORE_DB
      case 467: // REPLICATE_DO_TABLE
      case 468: // REPLICATE_IGNORE_TABLE
      case 469: // REPLICATE_WILD_DO_TABLE
      case 470: // REPLICATE_WILD_IGNORE_TABLE
      case 471: // REPLICATE_REWRITE_DB
      case 473: // RESET_SYM
      case 475: // RESOURCES
      case 476: // RESTORE_SYM
      case 478: // RESUME_SYM
      case 479: // RETURNED_SQLSTATE_SYM
      case 480: // RETURNS_SYM
      case 482: // REVERSE_SYM
      case 485: // ROLLBACK_SYM
      case 486: // ROLLUP_SYM
      case 487: // ROTATE_SYM
      case 488: // ROUTINE_SYM
      case 490: // ROW_FORMAT_SYM
      case 492: // ROW_COUNT_SYM
      case 493: // RTREE_SYM
      case 494: // SAVEPOINT_SYM
      case 495: // SCHEDULE_SYM
      case 496: // SCHEMA_NAME_SYM
      case 499: // SECOND_SYM
      case 500: // SECURITY_SYM
      case 504: // SERIALIZABLE_SYM
      case 505: // SERIAL_SYM
      case 506: // SESSION_SYM
      case 507: // SERVER_SYM
      case 511: // SHARE_SYM
      case 512: // SHARES_SYM
      case 516: // SHUTDOWN
      case 518: // SIGNED_SYM
      case 519: // SIMPLE_SYM
      case 520: // SLAVE
      case 521: // SLOW
      case 523: // SNAPSHOT_SYM
      case 524: // SOCKET_SYM
      case 525: // SONAME_SYM
      case 526: // SOUNDS_SYM
      case 527: // SOURCE_SYM
      case 533: // SQL_AFTER_GTIDS
      case 534: // SQL_AFTER_MTS_GAPS
      case 535: // SQL_BEFORE_GTIDS
      case 537: // SQL_BUFFER_RESULT
      case 540: // SQL_NO_CACHE_SYM
      case 543: // SQL_THREAD
      case 545: // STACKED_SYM
      case 547: // STARTS_SYM
      case 548: // START_SYM
      case 549: // STATS_AUTO_RECALC_SYM
      case 550: // STATS_PERSISTENT_SYM
      case 551: // STATS_SAMPLE_PAGES_SYM
      case 552: // STATUS_SYM
      case 555: // STOP_SYM
      case 556: // STORAGE_SYM
      case 559: // STRING_SYM
      case 560: // SUBCLASS_ORIGIN_SYM
      case 561: // SUBDATE_SYM
      case 562: // SUBJECT_SYM
      case 563: // SUBPARTITIONS_SYM
      case 564: // SUBPARTITION_SYM
      case 567: // SUPER_SYM
      case 568: // SUSPEND_SYM
      case 569: // SWAPS_SYM
      case 570: // SWITCHES_SYM
      case 572: // TABLES
      case 573: // VIEWS
      case 574: // TABLESPACE_SYM
      case 577: // TABLE_CHECKSUM_SYM
      case 578: // TABLE_NAME_SYM
      case 579: // TEMPORARY
      case 580: // TEMPTABLE_SYM
      case 582: // TEXT_STRING
      case 583: // TEXT_SYM
      case 584: // THAN_SYM
      case 586: // TIMESTAMP_SYM
      case 587: // TIMESTAMP_ADD
      case 588: // TIMESTAMP_DIFF
      case 589: // TIME_SYM
      case 595: // TRANSACTION_SYM
      case 596: // TRIGGERS_SYM
      case 600: // TRUNCATE_SYM
      case 601: // TYPES_SYM
      case 602: // TYPE_SYM
      case 604: // ULONGLONG_NUM
      case 605: // UNCOMMITTED_SYM
      case 606: // UNDEFINED_SYM
      case 608: // UNDOFILE_SYM
      case 609: // UNDO_BUFFER_SIZE_SYM
      case 611: // UNICODE_SYM
      case 612: // UNINSTALL_SYM
      case 615: // UNKNOWN_SYM
      case 618: // UNTIL_SYM
      case 620: // UPGRADE_SYM
      case 622: // USER
      case 623: // USE_FRM
      case 629: // VALIDATION_SYM
      case 631: // VALUE_SYM
      case 632: // VARBINARY_SYM
      case 634: // VARIABLES
      case 639: // VIEW_SYM
      case 641: // WAIT_SYM
      case 642: // WARNINGS
      case 643: // WEEK_SYM
      case 644: // WEIGHT_STRING_SYM
      case 651: // WITHOUT_SYM
      case 652: // WORK_SYM
      case 653: // WRAPPER_SYM
      case 655: // X509_SYM
      case 656: // XA_SYM
      case 657: // XID_SYM
      case 658: // XML_SYM
      case 661: // YEAR_SYM
      case 667: // PERSIST_SYM
      case 668: // ROLE_SYM
      case 669: // ADMIN_SYM
      case 670: // INVISIBLE_SYM
      case 671: // VISIBLE_SYM
      case 673: // COMPONENT_SYM
      case 682: // SKIP_SYM
      case 683: // LOCKED_SYM
      case 684: // NOWAIT_SYM
      case 686: // PERSIST_ONLY_SYM
      case 687: // HISTOGRAM_SYM
      case 688: // BUCKETS_SYM
      case 689: // OBSOLETE_TOKEN_930
      case 690: // CLONE_SYM
      case 693: // EXCLUDE_SYM
      case 695: // FOLLOWING_SYM
      case 702: // NULLS_SYM
      case 703: // OTHERS_SYM
      case 706: // PRECEDING_SYM
      case 708: // RESPECT_SYM
      case 710: // TIES_SYM
      case 711: // UNBOUNDED_SYM
      case 715: // NESTED_SYM
      case 716: // ORDINALITY_SYM
      case 717: // PATH_SYM
      case 718: // HISTORY_SYM
      case 719: // REUSE_SYM
      case 720: // SRID_SYM
      case 721: // THREAD_PRIORITY_SYM
      case 722: // RESOURCE_SYM
      case 724: // VCPU_SYM
      case 725: // MASTER_PUBLIC_KEY_PATH_SYM
      case 726: // GET_MASTER_PUBLIC_KEY_SYM
      case 727: // RESTART_SYM
      case 728: // DEFINITION_SYM
      case 729: // DESCRIPTION_SYM
      case 730: // ORGANIZATION_SYM
      case 731: // REFERENCE_SYM
      case 732: // ACTIVE_SYM
      case 733: // INACTIVE_SYM
      case 735: // OPTIONAL_SYM
      case 736: // SECONDARY_SYM
      case 737: // SECONDARY_ENGINE_SYM
      case 738: // SECONDARY_LOAD_SYM
      case 739: // SECONDARY_UNLOAD_SYM
      case 740: // RETAIN_SYM
      case 741: // OLD_SYM
      case 742: // ENFORCED_SYM
      case 743: // OJ_SYM
      case 744: // NETWORK_NAMESPACE_SYM
      case 794: // select_alias
      case 864: // opt_constraint_name
      case 1024: // execute_var_ident
      case 1027: // text_literal
      case 1028: // text_string
      case 1078: // opt_index_name_and_type
      case 1083: // key_part
      case 1085: // key_part_with_expression
      case 1086: // opt_ident
      case 1087: // opt_component
      case 1088: // charset_name
      case 1108: // internal_variable_name
      case 1139: // interval
      case 1140: // interval_time_stamp
      case 1143: // opt_table_alias
      case 1163: // opt_ordering_direction
      case 1164: // ordering_direction
      case 1168: // limit_option
      case 1177: // IDENT_sys
      case 1178: // TEXT_STRING_sys
      case 1179: // TEXT_STRING_literal
      case 1180: // TEXT_STRING_filesystem
      case 1181: // TEXT_STRING_password
      case 1182: // TEXT_STRING_hash
      case 1183: // ident
      case 1184: // ident_or_text
      case 1190: // nchar
      case 1191: // varchar
      case 1192: // nvarchar
      case 1193: // int_type
      case 1194: // real_type
      case 1196: // numeric_type
      case 1199: // type_datetime_precision
      case 1201: // field_length
      case 1202: // opt_field_length
      case 1212: // ident_keyword
      case 1214: // ident_keywords_ambiguous_2_labels
      case 1217: // ident_keywords_unambiguous
      case 1218: // lvalue_keyword
      case 1220: // TEXT_STRING_sys_nonewline
      case 1255: // opt_describe_column
      case 1264: // opt_db
      case 1285: // lvalue_ident
      case 1288: // password
      case 1342: // opt_load_data_charset
      case 1349: // opt_xml_rows_identified_by
        value.template destroy< string > ();
        break;

      case 1340: // load_data_lock
        value.template destroy< thr_lock_type > ();
        break;

      case 793: // select_item
        value.template destroy< tuple<Expression, string> > ();
        break;

      case 998: // opt_num_parts
      case 1002: // opt_num_subparts
      case 1271: // ulong_num
      case 1350: // opt_ignore_lines
        value.template destroy< ulong > ();
        break;

      case 989: // size_number
      case 1272: // real_ulong_num
      case 1273: // ulonglong_num
      case 1274: // real_ulonglong_num
        value.template destroy< ulonglong > ();
        break;

      case 1008: // part_values_in
      case 1009: // part_value_list
        value.template destroy< vector< PartValueItemsSPtr > > ();
        break;

      case 812: // opt_udf_expr_list
      case 813: // udf_expr_list
      case 1112: // opt_expr_list
      case 1114: // expr_list
      case 1158: // group_list
      case 1304: // insert_from_constructor
      case 1327: // fields
        value.template destroy< vector<Expression> > ();
        break;

      case 787: // from_tables
      case 788: // table_reference_list
        value.template destroy< vector<JoinStructurePointer> > ();
        break;

      case 1162: // order_list
        value.template destroy< vector<OrderItem> > ();
        break;

      case 1022: // execute_using
      case 1023: // execute_var_list
      case 1082: // key_list
      case 1084: // key_list_with_expression
      case 1151: // opt_derived_column_list
      case 1152: // simple_ident_list
        value.template destroy< vector<string> > ();
        break;

      case 1119: // when_list
        value.template destroy< vector<tuple<Expression, Expression>> > ();
        break;

      default:
        break;
    }

        Base::clear ();
      }

      /// Whether empty.
      bool empty () const YY_NOEXCEPT;

      /// Destructive move, \a s is emptied into this.
      void move (basic_symbol& s);

      /// The semantic value.
      semantic_type value;

      /// The location.
      location_type location;

    private:
#if YY_CPLUSPLUS < 201103L
      /// Assignment operator.
      basic_symbol& operator= (const basic_symbol& that);
#endif
    };

    /// Type access provider for token (enum) based symbols.
    struct by_type
    {
      /// Default constructor.
      by_type ();

#if 201103L <= YY_CPLUSPLUS
      /// Move constructor.
      by_type (by_type&& that);
#endif

      /// Copy constructor.
      by_type (const by_type& that);

      /// The symbol type as needed by the constructor.
      typedef token_type kind_type;

      /// Constructor from (external) token numbers.
      by_type (kind_type t);

      /// Record that this symbol is empty.
      void clear ();

      /// Steal the symbol type from \a that.
      void move (by_type& that);

      /// The (internal) type number (corresponding to \a type).
      /// \a empty when empty.
      symbol_number_type type_get () const YY_NOEXCEPT;

      /// The symbol type.
      /// \a empty_symbol when empty.
      /// An int, not token_number_type, to be able to store empty_symbol.
      int type;
    };

    /// "External" symbols: returned by the scanner.
    struct symbol_type : basic_symbol<by_type>
    {
      /// Superclass.
      typedef basic_symbol<by_type> super_type;

      /// Empty symbol.
      symbol_type () {}

      /// Constructor for valueless symbols, and symbols from each type.
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, location_type l)
        : super_type(token_type (tok), std::move (l))
      {
        YY_ASSERT (tok == token::END_OF_INPUT || tok == token::ABORT_SYM || tok == token::ACCESSIBLE_SYM || tok == token::ADD || tok == token::ALL || tok == token::ALTER || tok == token::OBSOLETE_TOKEN_271 || tok == token::ANALYZE_SYM || tok == token::AND_AND_SYM || tok == token::AND_SYM || tok == token::AS || tok == token::ASENSITIVE_SYM || tok == token::BEFORE_SYM || tok == token::BETWEEN_SYM || tok == token::BIGINT_SYM || tok == token::BINARY_SYM || tok == token::BIN_NUM || tok == token::BIT_AND || tok == token::BIT_OR || tok == token::BIT_XOR || tok == token::BLOB_SYM || tok == token::BOTH || tok == token::BY || tok == token::CALL_SYM || tok == token::CASCADE || tok == token::CASE_SYM || tok == token::CAST_SYM || tok == token::CHANGE || tok == token::CHAR_SYM || tok == token::CHECK_SYM || tok == token::COLLATE_SYM || tok == token::COLUMN_SYM || tok == token::CONDITION_SYM || tok == token::CONNECTION_ID_SYM || tok == token::CONSTRAINT || tok == token::CONTINUE_SYM || tok == token::CONVERT_SYM || tok == token::COUNT_SYM || tok == token::CROSS || tok == token::CUBE_SYM || tok == token::CURDATE || tok == token::CURRENT_USER || tok == token::CURSOR_SYM || tok == token::CURTIME || tok == token::DATABASE || tok == token::DATABASES || tok == token::DATE_ADD_INTERVAL || tok == token::DATE_SUB_INTERVAL || tok == token::DAY_HOUR_SYM || tok == token::DAY_MICROSECOND_SYM || tok == token::DAY_MINUTE_SYM || tok == token::DAY_SECOND_SYM || tok == token::REAL_NUM || tok == token::DECIMAL_SYM || tok == token::DECLARE_SYM || tok == token::DEFAULT_SYM || tok == token::DELAYED_SYM || tok == token::DELETE_SYM || tok == token::DESCRIBE || tok == token::OBSOLETE_TOKEN_388 || tok == token::DETERMINISTIC_SYM || tok == token::DICT_INDEX_SYM || tok == token::DISTINCT || tok == token::DIV_SYM || tok == token::DOUBLE_SYM || tok == token::DROP || tok == token::DUAL_SYM || tok == token::EACH_SYM || tok == token::ELSE || tok == token::ELSEIF_SYM || tok == token::ENCLOSED || tok == token::ENCODING || tok == token::EQ || tok == token::EQUAL_SYM || tok == token::ESCAPED || tok == token::EXISTS || tok == token::EXIT_SYM || tok == token::EXTRACT_SYM || tok == token::FALSE_SYM || tok == token::FETCH_SYM || tok == token::FLOAT_SYM || tok == token::FORCE_SYM || tok == token::FOREIGN || tok == token::FOR_SYM || tok == token::FROM || tok == token::FULLTEXT_SYM || tok == token::FUNCTION_SYM || tok == token::GE || tok == token::GENERATED || tok == token::GET_SYM || tok == token::GRANT || tok == token::GROUP_SYM || tok == token::GROUP_CONCAT_SYM || tok == token::GT_SYM || tok == token::HAVING || tok == token::HIGH_PRIORITY || tok == token::HOUR_MICROSECOND_SYM || tok == token::HOUR_MINUTE_SYM || tok == token::HOUR_SECOND_SYM || tok == token::IDENT_QUOTED || tok == token::IF || tok == token::IGNORE_SYM || tok == token::INDEX_SYM || tok == token::INFILE || tok == token::INNER_SYM || tok == token::INOUT_SYM || tok == token::INSENSITIVE_SYM || tok == token::INSERT_SYM || tok == token::INTO || tok == token::INT_SYM || tok == token::INTEGER_SYM || tok == token::IN_SYM || tok == token::IO_AFTER_GTIDS || tok == token::IO_BEFORE_GTIDS || tok == token::IS || tok == token::ITERATE_SYM || tok == token::JOIN_SYM || tok == token::JSON_SEPARATOR_SYM || tok == token::KEYS || tok == token::KEY_SYM || tok == token::KILL_SYM || tok == token::LE || tok == token::LEADING || tok == token::LEAVE_SYM || tok == token::LEFT || tok == token::LIKE || tok == token::LIMIT || tok == token::LINEAR_SYM || tok == token::LINES || tok == token::LOAD || tok == token::OBSOLETE_TOKEN_538 || tok == token::LOCK_SYM || tok == token::LONGBLOB_SYM || tok == token::LONGTEXT_SYM || tok == token::LONG_SYM || tok == token::LOOP_SYM || tok == token::LOW_PRIORITY || tok == token::LT || tok == token::MASTER_BIND_SYM || tok == token::MASTER_SSL_VERIFY_SERVER_CERT_SYM || tok == token::MATCH || tok == token::MAX_SYM || tok == token::MAX_VALUE_SYM || tok == token::MEDIUMBLOB_SYM || tok == token::MEDIUMINT_SYM || tok == token::MEDIUMTEXT_SYM || tok == token::MINUTE_MICROSECOND_SYM || tok == token::MINUTE_SECOND_SYM || tok == token::MIN_SYM || tok == token::MODIFIES_SYM || tok == token::MOD_SYM || tok == token::NATURAL || tok == token::NCHAR_STRING || tok == token::NE || tok == token::NEG || tok == token::NOT2_SYM || tok == token::NOT_SYM || tok == token::NOW_SYM || tok == token::NO_WRITE_TO_BINLOG || tok == token::NULL_SYM || tok == token::NUMERIC_SYM || tok == token::ON_SYM || tok == token::OPTIMIZE || tok == token::OPTIMIZER_COSTS_SYM || tok == token::OPTION || tok == token::OPTIONALLY || tok == token::OR2_SYM || tok == token::ORDER_SYM || tok == token::OR_OR_SYM || tok == token::OR_SYM || tok == token::OUTER || tok == token::OUTFILE || tok == token::OUT_SYM || tok == token::PARAM_MARKER || tok == token::OBSOLETE_TOKEN_654 || tok == token::PARTITION_SYM || tok == token::POSITION_SYM || tok == token::PRECISION || tok == token::PRIMARY_SYM || tok == token::PROCEDURE_SYM || tok == token::PURGE || tok == token::RANGE_SYM || tok == token::READS_SYM || tok == token::READ_SYM || tok == token::READ_WRITE_SYM || tok == token::REAL_SYM || tok == token::OBSOLETE_TOKEN_693 || tok == token::REFERENCES || tok == token::REGEXP || tok == token::RELEASE_SYM || tok == token::RENAME || tok == token::REPEAT_SYM || tok == token::REPLACE_SYM || tok == token::REQUIRE_SYM || tok == token::RESIGNAL_SYM || tok == token::RESTRICT || tok == token::RETURN_SYM || tok == token::REVOKE || tok == token::RIGHT || tok == token::ROWS_SYM || tok == token::ROW_SYM || tok == token::SCHEMA || tok == token::SECOND_MICROSECOND_SYM || tok == token::SELECT_SYM || tok == token::SENSITIVE_SYM || tok == token::SEPARATOR_SYM || tok == token::OBSOLETE_TOKEN_755 || tok == token::SET || tok == token::SET_VAR || tok == token::SHIFT_LEFT || tok == token::SHIFT_RIGHT || tok == token::SHOW || tok == token::SIGNAL_SYM || tok == token::SMALLINT_SYM || tok == token::SPATIAL_SYM || tok == token::SPECIFIC_SYM || tok == token::SQLEXCEPTION_SYM || tok == token::SQLSTATE_SYM || tok == token::SQLWARNING_SYM || tok == token::SQL_BIG_RESULT || tok == token::OBSOLETE_TOKEN_784 || tok == token::SQL_CALC_FOUND_ROWS || tok == token::SQL_SMALL_RESULT || tok == token::SQL_SYM || tok == token::SSL_SYM || tok == token::STARTING || tok == token::STDDEV_SAMP_SYM || tok == token::STD_SYM || tok == token::STORED_SYM || tok == token::STRAIGHT_JOIN || tok == token::SUBSTRING || tok == token::SUM_SYM || tok == token::SYSDATE || tok == token::OBSOLETE_TOKEN_820 || tok == token::TABLE_SYM || tok == token::TERMINATED || tok == token::THEN_SYM || tok == token::TINYBLOB_SYM || tok == token::TINYINT_SYM || tok == token::TINYTEXT_SYN || tok == token::TO_SYM || tok == token::TRAILING || tok == token::TRIGGER_SYM || tok == token::TRIM || tok == token::TRUE_SYM || tok == token::OBSOLETE_TOKEN_848 || tok == token::UNDERSCORE_CHARSET || tok == token::UNDO_SYM || tok == token::UNION_SYM || tok == token::UNIQUE_SYM || tok == token::UNLOCK_SYM || tok == token::UNSIGNED_SYM || tok == token::UPDATE_SYM || tok == token::USAGE || tok == token::USE_SYM || tok == token::USING || tok == token::UTC_DATE_SYM || tok == token::UTC_TIMESTAMP_SYM || tok == token::UTC_TIME_SYM || tok == token::VALUES || tok == token::VARCHAR_SYM || tok == token::VARIANCE_SYM || tok == token::VARYING || tok == token::VAR_SAMP_SYM || tok == token::VERSION_SYM || tok == token::VIRTUAL_SYM || tok == token::WHEN_SYM || tok == token::WHERE || tok == token::WHILE_SYM || tok == token::WITH || tok == token::OBSOLETE_TOKEN_893 || tok == token::WITH_ROLLUP_SYM || tok == token::WRITE_SYM || tok == token::XOR || tok == token::YEAR_MONTH_SYM || tok == token::ZEROFILL_SYM || tok == token::EXPLAIN_SYM || tok == token::TREE_SYM || tok == token::TRADITIONAL_SYM || tok == token::JSON_UNQUOTED_SEPARATOR_SYM || tok == token::EXCEPT_SYM || tok == token::RECURSIVE_SYM || tok == token::GRAMMAR_SELECTOR_EXPR || tok == token::GRAMMAR_SELECTOR_GCOL || tok == token::GRAMMAR_SELECTOR_PART || tok == token::GRAMMAR_SELECTOR_CTE || tok == token::JSON_OBJECTAGG || tok == token::JSON_ARRAYAGG || tok == token::OF_SYM || tok == token::GROUPING_SYM || tok == token::CUME_DIST_SYM || tok == token::DENSE_RANK_SYM || tok == token::FIRST_VALUE_SYM || tok == token::GROUPS_SYM || tok == token::LAG_SYM || tok == token::LAST_VALUE_SYM || tok == token::LEAD_SYM || tok == token::NTH_VALUE_SYM || tok == token::NTILE_SYM || tok == token::OVER_SYM || tok == token::PERCENT_RANK_SYM || tok == token::RANK_SYM || tok == token::ROW_NUMBER_SYM || tok == token::WINDOW_SYM || tok == token::EMPTY_SYM || tok == token::JSON_TABLE_SYM || tok == token::SYSTEM_SYM || tok == token::LATERAL_SYM || tok == token::ADD_SYM || tok == 43 || tok == token::MINUS_SYM || tok == 45 || tok == token::CONDITIONLESS_JOIN || tok == 124 || tok == 38 || tok == 42 || tok == 47 || tok == 37 || tok == 94 || tok == 126 || tok == token::SUBQUERY_AS_EXPR || tok == 40 || tok == 41 || tok == token::EMPTY_FROM_CLAUSE || tok == 59 || tok == 44 || tok == 33 || tok == 46 || tok == 64 || tok == 58);
      }
#else
      symbol_type (int tok, const location_type& l)
        : super_type(token_type (tok), l)
      {
        YY_ASSERT (tok == token::END_OF_INPUT || tok == token::ABORT_SYM || tok == token::ACCESSIBLE_SYM || tok == token::ADD || tok == token::ALL || tok == token::ALTER || tok == token::OBSOLETE_TOKEN_271 || tok == token::ANALYZE_SYM || tok == token::AND_AND_SYM || tok == token::AND_SYM || tok == token::AS || tok == token::ASENSITIVE_SYM || tok == token::BEFORE_SYM || tok == token::BETWEEN_SYM || tok == token::BIGINT_SYM || tok == token::BINARY_SYM || tok == token::BIN_NUM || tok == token::BIT_AND || tok == token::BIT_OR || tok == token::BIT_XOR || tok == token::BLOB_SYM || tok == token::BOTH || tok == token::BY || tok == token::CALL_SYM || tok == token::CASCADE || tok == token::CASE_SYM || tok == token::CAST_SYM || tok == token::CHANGE || tok == token::CHAR_SYM || tok == token::CHECK_SYM || tok == token::COLLATE_SYM || tok == token::COLUMN_SYM || tok == token::CONDITION_SYM || tok == token::CONNECTION_ID_SYM || tok == token::CONSTRAINT || tok == token::CONTINUE_SYM || tok == token::CONVERT_SYM || tok == token::COUNT_SYM || tok == token::CROSS || tok == token::CUBE_SYM || tok == token::CURDATE || tok == token::CURRENT_USER || tok == token::CURSOR_SYM || tok == token::CURTIME || tok == token::DATABASE || tok == token::DATABASES || tok == token::DATE_ADD_INTERVAL || tok == token::DATE_SUB_INTERVAL || tok == token::DAY_HOUR_SYM || tok == token::DAY_MICROSECOND_SYM || tok == token::DAY_MINUTE_SYM || tok == token::DAY_SECOND_SYM || tok == token::REAL_NUM || tok == token::DECIMAL_SYM || tok == token::DECLARE_SYM || tok == token::DEFAULT_SYM || tok == token::DELAYED_SYM || tok == token::DELETE_SYM || tok == token::DESCRIBE || tok == token::OBSOLETE_TOKEN_388 || tok == token::DETERMINISTIC_SYM || tok == token::DICT_INDEX_SYM || tok == token::DISTINCT || tok == token::DIV_SYM || tok == token::DOUBLE_SYM || tok == token::DROP || tok == token::DUAL_SYM || tok == token::EACH_SYM || tok == token::ELSE || tok == token::ELSEIF_SYM || tok == token::ENCLOSED || tok == token::ENCODING || tok == token::EQ || tok == token::EQUAL_SYM || tok == token::ESCAPED || tok == token::EXISTS || tok == token::EXIT_SYM || tok == token::EXTRACT_SYM || tok == token::FALSE_SYM || tok == token::FETCH_SYM || tok == token::FLOAT_SYM || tok == token::FORCE_SYM || tok == token::FOREIGN || tok == token::FOR_SYM || tok == token::FROM || tok == token::FULLTEXT_SYM || tok == token::FUNCTION_SYM || tok == token::GE || tok == token::GENERATED || tok == token::GET_SYM || tok == token::GRANT || tok == token::GROUP_SYM || tok == token::GROUP_CONCAT_SYM || tok == token::GT_SYM || tok == token::HAVING || tok == token::HIGH_PRIORITY || tok == token::HOUR_MICROSECOND_SYM || tok == token::HOUR_MINUTE_SYM || tok == token::HOUR_SECOND_SYM || tok == token::IDENT_QUOTED || tok == token::IF || tok == token::IGNORE_SYM || tok == token::INDEX_SYM || tok == token::INFILE || tok == token::INNER_SYM || tok == token::INOUT_SYM || tok == token::INSENSITIVE_SYM || tok == token::INSERT_SYM || tok == token::INTO || tok == token::INT_SYM || tok == token::INTEGER_SYM || tok == token::IN_SYM || tok == token::IO_AFTER_GTIDS || tok == token::IO_BEFORE_GTIDS || tok == token::IS || tok == token::ITERATE_SYM || tok == token::JOIN_SYM || tok == token::JSON_SEPARATOR_SYM || tok == token::KEYS || tok == token::KEY_SYM || tok == token::KILL_SYM || tok == token::LE || tok == token::LEADING || tok == token::LEAVE_SYM || tok == token::LEFT || tok == token::LIKE || tok == token::LIMIT || tok == token::LINEAR_SYM || tok == token::LINES || tok == token::LOAD || tok == token::OBSOLETE_TOKEN_538 || tok == token::LOCK_SYM || tok == token::LONGBLOB_SYM || tok == token::LONGTEXT_SYM || tok == token::LONG_SYM || tok == token::LOOP_SYM || tok == token::LOW_PRIORITY || tok == token::LT || tok == token::MASTER_BIND_SYM || tok == token::MASTER_SSL_VERIFY_SERVER_CERT_SYM || tok == token::MATCH || tok == token::MAX_SYM || tok == token::MAX_VALUE_SYM || tok == token::MEDIUMBLOB_SYM || tok == token::MEDIUMINT_SYM || tok == token::MEDIUMTEXT_SYM || tok == token::MINUTE_MICROSECOND_SYM || tok == token::MINUTE_SECOND_SYM || tok == token::MIN_SYM || tok == token::MODIFIES_SYM || tok == token::MOD_SYM || tok == token::NATURAL || tok == token::NCHAR_STRING || tok == token::NE || tok == token::NEG || tok == token::NOT2_SYM || tok == token::NOT_SYM || tok == token::NOW_SYM || tok == token::NO_WRITE_TO_BINLOG || tok == token::NULL_SYM || tok == token::NUMERIC_SYM || tok == token::ON_SYM || tok == token::OPTIMIZE || tok == token::OPTIMIZER_COSTS_SYM || tok == token::OPTION || tok == token::OPTIONALLY || tok == token::OR2_SYM || tok == token::ORDER_SYM || tok == token::OR_OR_SYM || tok == token::OR_SYM || tok == token::OUTER || tok == token::OUTFILE || tok == token::OUT_SYM || tok == token::PARAM_MARKER || tok == token::OBSOLETE_TOKEN_654 || tok == token::PARTITION_SYM || tok == token::POSITION_SYM || tok == token::PRECISION || tok == token::PRIMARY_SYM || tok == token::PROCEDURE_SYM || tok == token::PURGE || tok == token::RANGE_SYM || tok == token::READS_SYM || tok == token::READ_SYM || tok == token::READ_WRITE_SYM || tok == token::REAL_SYM || tok == token::OBSOLETE_TOKEN_693 || tok == token::REFERENCES || tok == token::REGEXP || tok == token::RELEASE_SYM || tok == token::RENAME || tok == token::REPEAT_SYM || tok == token::REPLACE_SYM || tok == token::REQUIRE_SYM || tok == token::RESIGNAL_SYM || tok == token::RESTRICT || tok == token::RETURN_SYM || tok == token::REVOKE || tok == token::RIGHT || tok == token::ROWS_SYM || tok == token::ROW_SYM || tok == token::SCHEMA || tok == token::SECOND_MICROSECOND_SYM || tok == token::SELECT_SYM || tok == token::SENSITIVE_SYM || tok == token::SEPARATOR_SYM || tok == token::OBSOLETE_TOKEN_755 || tok == token::SET || tok == token::SET_VAR || tok == token::SHIFT_LEFT || tok == token::SHIFT_RIGHT || tok == token::SHOW || tok == token::SIGNAL_SYM || tok == token::SMALLINT_SYM || tok == token::SPATIAL_SYM || tok == token::SPECIFIC_SYM || tok == token::SQLEXCEPTION_SYM || tok == token::SQLSTATE_SYM || tok == token::SQLWARNING_SYM || tok == token::SQL_BIG_RESULT || tok == token::OBSOLETE_TOKEN_784 || tok == token::SQL_CALC_FOUND_ROWS || tok == token::SQL_SMALL_RESULT || tok == token::SQL_SYM || tok == token::SSL_SYM || tok == token::STARTING || tok == token::STDDEV_SAMP_SYM || tok == token::STD_SYM || tok == token::STORED_SYM || tok == token::STRAIGHT_JOIN || tok == token::SUBSTRING || tok == token::SUM_SYM || tok == token::SYSDATE || tok == token::OBSOLETE_TOKEN_820 || tok == token::TABLE_SYM || tok == token::TERMINATED || tok == token::THEN_SYM || tok == token::TINYBLOB_SYM || tok == token::TINYINT_SYM || tok == token::TINYTEXT_SYN || tok == token::TO_SYM || tok == token::TRAILING || tok == token::TRIGGER_SYM || tok == token::TRIM || tok == token::TRUE_SYM || tok == token::OBSOLETE_TOKEN_848 || tok == token::UNDERSCORE_CHARSET || tok == token::UNDO_SYM || tok == token::UNION_SYM || tok == token::UNIQUE_SYM || tok == token::UNLOCK_SYM || tok == token::UNSIGNED_SYM || tok == token::UPDATE_SYM || tok == token::USAGE || tok == token::USE_SYM || tok == token::USING || tok == token::UTC_DATE_SYM || tok == token::UTC_TIMESTAMP_SYM || tok == token::UTC_TIME_SYM || tok == token::VALUES || tok == token::VARCHAR_SYM || tok == token::VARIANCE_SYM || tok == token::VARYING || tok == token::VAR_SAMP_SYM || tok == token::VERSION_SYM || tok == token::VIRTUAL_SYM || tok == token::WHEN_SYM || tok == token::WHERE || tok == token::WHILE_SYM || tok == token::WITH || tok == token::OBSOLETE_TOKEN_893 || tok == token::WITH_ROLLUP_SYM || tok == token::WRITE_SYM || tok == token::XOR || tok == token::YEAR_MONTH_SYM || tok == token::ZEROFILL_SYM || tok == token::EXPLAIN_SYM || tok == token::TREE_SYM || tok == token::TRADITIONAL_SYM || tok == token::JSON_UNQUOTED_SEPARATOR_SYM || tok == token::EXCEPT_SYM || tok == token::RECURSIVE_SYM || tok == token::GRAMMAR_SELECTOR_EXPR || tok == token::GRAMMAR_SELECTOR_GCOL || tok == token::GRAMMAR_SELECTOR_PART || tok == token::GRAMMAR_SELECTOR_CTE || tok == token::JSON_OBJECTAGG || tok == token::JSON_ARRAYAGG || tok == token::OF_SYM || tok == token::GROUPING_SYM || tok == token::CUME_DIST_SYM || tok == token::DENSE_RANK_SYM || tok == token::FIRST_VALUE_SYM || tok == token::GROUPS_SYM || tok == token::LAG_SYM || tok == token::LAST_VALUE_SYM || tok == token::LEAD_SYM || tok == token::NTH_VALUE_SYM || tok == token::NTILE_SYM || tok == token::OVER_SYM || tok == token::PERCENT_RANK_SYM || tok == token::RANK_SYM || tok == token::ROW_NUMBER_SYM || tok == token::WINDOW_SYM || tok == token::EMPTY_SYM || tok == token::JSON_TABLE_SYM || tok == token::SYSTEM_SYM || tok == token::LATERAL_SYM || tok == token::ADD_SYM || tok == 43 || tok == token::MINUS_SYM || tok == 45 || tok == token::CONDITIONLESS_JOIN || tok == 124 || tok == 38 || tok == 42 || tok == 47 || tok == 37 || tok == 94 || tok == 126 || tok == token::SUBQUERY_AS_EXPR || tok == 40 || tok == 41 || tok == token::EMPTY_FROM_CLAUSE || tok == 59 || tok == 44 || tok == 33 || tok == 46 || tok == 64 || tok == 58);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      symbol_type (int tok, string v, location_type l)
        : super_type(token_type (tok), std::move (v), std::move (l))
      {
        YY_ASSERT (tok == token::ACCOUNT_SYM || tok == token::ACTION || tok == token::ADDDATE_SYM || tok == token::AFTER_SYM || tok == token::AGAINST || tok == token::AGGREGATE_SYM || tok == token::ALGORITHM_SYM || tok == token::ALWAYS_SYM || tok == token::ANY_SYM || tok == token::ASC || tok == token::ASCII_SYM || tok == token::AT_SYM || tok == token::AUTOEXTEND_SIZE_SYM || tok == token::AUTO_INC || tok == token::AVG_ROW_LENGTH || tok == token::AVG_SYM || tok == token::BACKUP_SYM || tok == token::BEGIN_SYM || tok == token::BINLOG_SYM || tok == token::BIT_SYM || tok == token::BLOCK_SYM || tok == token::BOOLEAN_SYM || tok == token::BOOL_SYM || tok == token::BTREE_SYM || tok == token::BYTE_SYM || tok == token::CACHE_SYM || tok == token::CASCADED || tok == token::CATALOG_NAME_SYM || tok == token::CHAIN_SYM || tok == token::CHANGED || tok == token::CHANNEL_SYM || tok == token::CHARSET || tok == token::CHECKSUM_SYM || tok == token::CIPHER_SYM || tok == token::CLASS_ORIGIN_SYM || tok == token::CLIENT_SYM || tok == token::CLOSE_SYM || tok == token::COALESCE || tok == token::CODE_SYM || tok == token::COLLATION_SYM || tok == token::COLUMNS || tok == token::COLUMN_FORMAT_SYM || tok == token::COLUMN_NAME_SYM || tok == token::COMMENT_SYM || tok == token::COMMITTED_SYM || tok == token::COMMIT_SYM || tok == token::COMPACT_SYM || tok == token::COMPLETION_SYM || tok == token::COMPRESSED_SYM || tok == token::COMPRESSION_SYM || tok == token::ENCRYPTION_SYM || tok == token::CONCURRENT || tok == token::CONNECTION_SYM || tok == token::CONSISTENT_SYM || tok == token::CONSTRAINT_CATALOG_SYM || tok == token::CONSTRAINT_NAME_SYM || tok == token::CONSTRAINT_SCHEMA_SYM || tok == token::CONTAINS_SYM || tok == token::CONTEXT_SYM || tok == token::CPU_SYM || tok == token::CREATE || tok == token::CURRENT_SYM || tok == token::CURSOR_NAME_SYM || tok == token::DATAFILE_SYM || tok == token::DATA_SYM || tok == token::DATETIME_SYM || tok == token::DATE_SYM || tok == token::DAY_SYM || tok == token::DEALLOCATE_SYM || tok == token::DECIMAL_NUM || tok == token::DEFAULT_AUTH_SYM || tok == token::DEFINER_SYM || tok == token::DELAY_KEY_WRITE_SYM || tok == token::DESC || tok == token::DIAGNOSTICS_SYM || tok == token::BYTEDICT_SYM || tok == token::SHORTDICT_SYM || tok == token::INTDICT_SYM || tok == token::DIRECTORY_SYM || tok == token::DISABLE_SYM || tok == token::DISCARD_SYM || tok == token::DISK_SYM || tok == token::DO_SYM || tok == token::DUMPFILE || tok == token::DUPLICATE_SYM || tok == token::DYNAMIC_SYM || tok == token::ENABLE_SYM || tok == token::END || tok == token::ENDS_SYM || tok == token::ENGINES_SYM || tok == token::ENGINE_SYM || tok == token::ENUM_SYM || tok == token::ERROR_SYM || tok == token::ERRORS || tok == token::ESCAPE_SYM || tok == token::EVENTS_SYM || tok == token::EVENT_SYM || tok == token::EVERY_SYM || tok == token::EXCHANGE_SYM || tok == token::EXECUTE_SYM || tok == token::EXPANSION_SYM || tok == token::EXPIRE_SYM || tok == token::EXPORT_SYM || tok == token::EXTENDED_SYM || tok == token::EXTENT_SIZE_SYM || tok == token::FAST_SYM || tok == token::FAULTS_SYM || tok == token::FILE_SYM || tok == token::FILE_BLOCK_SIZE_SYM || tok == token::FILTER_SYM || tok == token::FIRST_SYM || tok == token::FIXED_SYM || tok == token::FLOAT_NUM || tok == token::FLUSH_SYM || tok == token::FOLLOWS_SYM || tok == token::FORMAT_SYM || tok == token::FOUND_SYM || tok == token::FULL || tok == token::GENERAL || tok == token::GROUP_REPLICATION || tok == token::GEOMETRYCOLLECTION_SYM || tok == token::GEOMETRY_SYM || tok == token::GET_FORMAT || tok == token::GLOBAL_SYM || tok == token::GRANTS || tok == token::HANDLER_SYM || tok == token::HASH_SYM || tok == token::HELP_SYM || tok == token::HEX_NUM || tok == token::HOST_SYM || tok == token::HOSTS_SYM || tok == token::HOUR_SYM || tok == token::IDENT || tok == token::IDENTIFIED_SYM || tok == token::IGNORE_SERVER_IDS_SYM || tok == token::IMPORT || tok == token::INDEXES || tok == token::INITIAL_SIZE_SYM || tok == token::INSERT_METHOD || tok == token::INSTANCE_SYM || tok == token::INSTALL_SYM || tok == token::INTERVAL_SYM || tok == token::INVOKER_SYM || tok == token::IO_SYM || tok == token::IPC_SYM || tok == token::ISOLATION || tok == token::ISSUER_SYM || tok == token::JSON_SYM || tok == token::KEY_BLOCK_SIZE || tok == token::LANGUAGE_SYM || tok == token::LAST_SYM || tok == token::LEAVES || tok == token::LESS_SYM || tok == token::LEVEL_SYM || tok == token::LEX_HOSTNAME || tok == token::LINESTRING_SYM || tok == token::LIST_SYM || tok == token::LOCAL_SYM || tok == token::LOCKS_SYM || tok == token::LOGFILE_SYM || tok == token::LOGS_SYM || tok == token::LONG_NUM || tok == token::MASTER_AUTO_POSITION_SYM || tok == token::MASTER_CONNECT_RETRY_SYM || tok == token::MASTER_DELAY_SYM || tok == token::MASTER_HOST_SYM || tok == token::MASTER_LOG_FILE_SYM || tok == token::MASTER_LOG_POS_SYM || tok == token::MASTER_PASSWORD_SYM || tok == token::MASTER_PORT_SYM || tok == token::MASTER_RETRY_COUNT_SYM || tok == token::MASTER_SERVER_ID_SYM || tok == token::MASTER_SSL_CAPATH_SYM || tok == token::MASTER_TLS_VERSION_SYM || tok == token::MASTER_SSL_CA_SYM || tok == token::MASTER_SSL_CERT_SYM || tok == token::MASTER_SSL_CIPHER_SYM || tok == token::MASTER_SSL_CRL_SYM || tok == token::MASTER_SSL_CRLPATH_SYM || tok == token::MASTER_SSL_KEY_SYM || tok == token::MASTER_SSL_SYM || tok == token::MASTER_SYM || tok == token::MASTER_USER_SYM || tok == token::MASTER_HEARTBEAT_PERIOD_SYM || tok == token::MAX_CONNECTIONS_PER_HOUR || tok == token::MAX_QUERIES_PER_HOUR || tok == token::MAX_ROWS || tok == token::MAX_SIZE_SYM || tok == token::MAX_UPDATES_PER_HOUR || tok == token::MAX_USER_CONNECTIONS_SYM || tok == token::MEDIUM_SYM || tok == token::MEMORY_SYM || tok == token::MERGE_SYM || tok == token::MESSAGE_TEXT_SYM || tok == token::MICROSECOND_SYM || tok == token::MIGRATE_SYM || tok == token::MINUTE_SYM || tok == token::MIN_ROWS || tok == token::MODE_SYM || tok == token::MODIFY_SYM || tok == token::MONTH_SYM || tok == token::MULTILINESTRING_SYM || tok == token::MULTIPOINT_SYM || tok == token::MULTIPOLYGON_SYM || tok == token::MUTEX_SYM || tok == token::MYSQL_ERRNO_SYM || tok == token::NAMES_SYM || tok == token::NAME_SYM || tok == token::NATIONAL_SYM || tok == token::NCHAR_SYM || tok == token::NDBCLUSTER_SYM || tok == token::NEVER_SYM || tok == token::NEW_SYM || tok == token::NEXT_SYM || tok == token::NODEGROUP_SYM || tok == token::NONE_SYM || tok == token::NO_SYM || tok == token::NO_WAIT_SYM || tok == token::NUM || tok == token::NUMBER_SYM || tok == token::NVARCHAR_SYM || tok == token::OFFSET_SYM || tok == token::ONE_SYM || tok == token::ONLY_SYM || tok == token::OPEN_SYM || tok == token::OPTIONS_SYM || tok == token::OWNER_SYM || tok == token::PACK_KEYS_SYM || tok == token::PAGE_SYM || tok == token::PARSER_SYM || tok == token::PARTIAL || tok == token::PARTITIONS_SYM || tok == token::PARTITIONING_SYM || tok == token::PASSWORD || tok == token::PHASE_SYM || tok == token::PLUGIN_DIR_SYM || tok == token::PLUGIN_SYM || tok == token::PLUGINS_SYM || tok == token::POINT_SYM || tok == token::POLYGON_SYM || tok == token::PORT_SYM || tok == token::PRECEDES_SYM || tok == token::PREPARE_SYM || tok == token::PRESERVE_SYM || tok == token::PREV_SYM || tok == token::PRIVILEGES || tok == token::PROCESS || tok == token::PROCESSLIST_SYM || tok == token::PROFILE_SYM || tok == token::PROFILES_SYM || tok == token::PROXY_SYM || tok == token::QUARTER_SYM || tok == token::QUERY_SYM || tok == token::QUICK || tok == token::READ_ONLY_SYM || tok == token::REBUILD_SYM || tok == token::RECOVER_SYM || tok == token::REDO_BUFFER_SIZE_SYM || tok == token::REDUNDANT_SYM || tok == token::RELAY || tok == token::RELAYLOG_SYM || tok == token::RELAY_LOG_FILE_SYM || tok == token::RELAY_LOG_POS_SYM || tok == token::RELAY_THREAD || tok == token::RELOAD || tok == token::REMOVE_SYM || tok == token::REORGANIZE_SYM || tok == token::REPAIR || tok == token::REPEATABLE_SYM || tok == token::REPLICATION || tok == token::REPLICATE_DO_DB || tok == token::REPLICATE_IGNORE_DB || tok == token::REPLICATE_DO_TABLE || tok == token::REPLICATE_IGNORE_TABLE || tok == token::REPLICATE_WILD_DO_TABLE || tok == token::REPLICATE_WILD_IGNORE_TABLE || tok == token::REPLICATE_REWRITE_DB || tok == token::RESET_SYM || tok == token::RESOURCES || tok == token::RESTORE_SYM || tok == token::RESUME_SYM || tok == token::RETURNED_SQLSTATE_SYM || tok == token::RETURNS_SYM || tok == token::REVERSE_SYM || tok == token::ROLLBACK_SYM || tok == token::ROLLUP_SYM || tok == token::ROTATE_SYM || tok == token::ROUTINE_SYM || tok == token::ROW_FORMAT_SYM || tok == token::ROW_COUNT_SYM || tok == token::RTREE_SYM || tok == token::SAVEPOINT_SYM || tok == token::SCHEDULE_SYM || tok == token::SCHEMA_NAME_SYM || tok == token::SECOND_SYM || tok == token::SECURITY_SYM || tok == token::SERIALIZABLE_SYM || tok == token::SERIAL_SYM || tok == token::SESSION_SYM || tok == token::SERVER_SYM || tok == token::SHARE_SYM || tok == token::SHARES_SYM || tok == token::SHUTDOWN || tok == token::SIGNED_SYM || tok == token::SIMPLE_SYM || tok == token::SLAVE || tok == token::SLOW || tok == token::SNAPSHOT_SYM || tok == token::SOCKET_SYM || tok == token::SONAME_SYM || tok == token::SOUNDS_SYM || tok == token::SOURCE_SYM || tok == token::SQL_AFTER_GTIDS || tok == token::SQL_AFTER_MTS_GAPS || tok == token::SQL_BEFORE_GTIDS || tok == token::SQL_BUFFER_RESULT || tok == token::SQL_NO_CACHE_SYM || tok == token::SQL_THREAD || tok == token::STACKED_SYM || tok == token::STARTS_SYM || tok == token::START_SYM || tok == token::STATS_AUTO_RECALC_SYM || tok == token::STATS_PERSISTENT_SYM || tok == token::STATS_SAMPLE_PAGES_SYM || tok == token::STATUS_SYM || tok == token::STOP_SYM || tok == token::STORAGE_SYM || tok == token::STRING_SYM || tok == token::SUBCLASS_ORIGIN_SYM || tok == token::SUBDATE_SYM || tok == token::SUBJECT_SYM || tok == token::SUBPARTITIONS_SYM || tok == token::SUBPARTITION_SYM || tok == token::SUPER_SYM || tok == token::SUSPEND_SYM || tok == token::SWAPS_SYM || tok == token::SWITCHES_SYM || tok == token::TABLES || tok == token::VIEWS || tok == token::TABLESPACE_SYM || tok == token::TABLE_CHECKSUM_SYM || tok == token::TABLE_NAME_SYM || tok == token::TEMPORARY || tok == token::TEMPTABLE_SYM || tok == token::TEXT_STRING || tok == token::TEXT_SYM || tok == token::THAN_SYM || tok == token::TIMESTAMP_SYM || tok == token::TIMESTAMP_ADD || tok == token::TIMESTAMP_DIFF || tok == token::TIME_SYM || tok == token::TRANSACTION_SYM || tok == token::TRIGGERS_SYM || tok == token::TRUNCATE_SYM || tok == token::TYPES_SYM || tok == token::TYPE_SYM || tok == token::ULONGLONG_NUM || tok == token::UNCOMMITTED_SYM || tok == token::UNDEFINED_SYM || tok == token::UNDOFILE_SYM || tok == token::UNDO_BUFFER_SIZE_SYM || tok == token::UNICODE_SYM || tok == token::UNINSTALL_SYM || tok == token::UNKNOWN_SYM || tok == token::UNTIL_SYM || tok == token::UPGRADE_SYM || tok == token::USER || tok == token::USE_FRM || tok == token::VALIDATION_SYM || tok == token::VALUE_SYM || tok == token::VARBINARY_SYM || tok == token::VARIABLES || tok == token::VIEW_SYM || tok == token::WAIT_SYM || tok == token::WARNINGS || tok == token::WEEK_SYM || tok == token::WEIGHT_STRING_SYM || tok == token::WITHOUT_SYM || tok == token::WORK_SYM || tok == token::WRAPPER_SYM || tok == token::X509_SYM || tok == token::XA_SYM || tok == token::XID_SYM || tok == token::XML_SYM || tok == token::YEAR_SYM || tok == token::PERSIST_SYM || tok == token::ROLE_SYM || tok == token::ADMIN_SYM || tok == token::INVISIBLE_SYM || tok == token::VISIBLE_SYM || tok == token::COMPONENT_SYM || tok == token::SKIP_SYM || tok == token::LOCKED_SYM || tok == token::NOWAIT_SYM || tok == token::PERSIST_ONLY_SYM || tok == token::HISTOGRAM_SYM || tok == token::BUCKETS_SYM || tok == token::OBSOLETE_TOKEN_930 || tok == token::CLONE_SYM || tok == token::EXCLUDE_SYM || tok == token::FOLLOWING_SYM || tok == token::NULLS_SYM || tok == token::OTHERS_SYM || tok == token::PRECEDING_SYM || tok == token::RESPECT_SYM || tok == token::TIES_SYM || tok == token::UNBOUNDED_SYM || tok == token::NESTED_SYM || tok == token::ORDINALITY_SYM || tok == token::PATH_SYM || tok == token::HISTORY_SYM || tok == token::REUSE_SYM || tok == token::SRID_SYM || tok == token::THREAD_PRIORITY_SYM || tok == token::RESOURCE_SYM || tok == token::VCPU_SYM || tok == token::MASTER_PUBLIC_KEY_PATH_SYM || tok == token::GET_MASTER_PUBLIC_KEY_SYM || tok == token::RESTART_SYM || tok == token::DEFINITION_SYM || tok == token::DESCRIPTION_SYM || tok == token::ORGANIZATION_SYM || tok == token::REFERENCE_SYM || tok == token::ACTIVE_SYM || tok == token::INACTIVE_SYM || tok == token::OPTIONAL_SYM || tok == token::SECONDARY_SYM || tok == token::SECONDARY_ENGINE_SYM || tok == token::SECONDARY_LOAD_SYM || tok == token::SECONDARY_UNLOAD_SYM || tok == token::RETAIN_SYM || tok == token::OLD_SYM || tok == token::ENFORCED_SYM || tok == token::OJ_SYM || tok == token::NETWORK_NAMESPACE_SYM);
      }
#else
      symbol_type (int tok, const string& v, const location_type& l)
        : super_type(token_type (tok), v, l)
      {
        YY_ASSERT (tok == token::ACCOUNT_SYM || tok == token::ACTION || tok == token::ADDDATE_SYM || tok == token::AFTER_SYM || tok == token::AGAINST || tok == token::AGGREGATE_SYM || tok == token::ALGORITHM_SYM || tok == token::ALWAYS_SYM || tok == token::ANY_SYM || tok == token::ASC || tok == token::ASCII_SYM || tok == token::AT_SYM || tok == token::AUTOEXTEND_SIZE_SYM || tok == token::AUTO_INC || tok == token::AVG_ROW_LENGTH || tok == token::AVG_SYM || tok == token::BACKUP_SYM || tok == token::BEGIN_SYM || tok == token::BINLOG_SYM || tok == token::BIT_SYM || tok == token::BLOCK_SYM || tok == token::BOOLEAN_SYM || tok == token::BOOL_SYM || tok == token::BTREE_SYM || tok == token::BYTE_SYM || tok == token::CACHE_SYM || tok == token::CASCADED || tok == token::CATALOG_NAME_SYM || tok == token::CHAIN_SYM || tok == token::CHANGED || tok == token::CHANNEL_SYM || tok == token::CHARSET || tok == token::CHECKSUM_SYM || tok == token::CIPHER_SYM || tok == token::CLASS_ORIGIN_SYM || tok == token::CLIENT_SYM || tok == token::CLOSE_SYM || tok == token::COALESCE || tok == token::CODE_SYM || tok == token::COLLATION_SYM || tok == token::COLUMNS || tok == token::COLUMN_FORMAT_SYM || tok == token::COLUMN_NAME_SYM || tok == token::COMMENT_SYM || tok == token::COMMITTED_SYM || tok == token::COMMIT_SYM || tok == token::COMPACT_SYM || tok == token::COMPLETION_SYM || tok == token::COMPRESSED_SYM || tok == token::COMPRESSION_SYM || tok == token::ENCRYPTION_SYM || tok == token::CONCURRENT || tok == token::CONNECTION_SYM || tok == token::CONSISTENT_SYM || tok == token::CONSTRAINT_CATALOG_SYM || tok == token::CONSTRAINT_NAME_SYM || tok == token::CONSTRAINT_SCHEMA_SYM || tok == token::CONTAINS_SYM || tok == token::CONTEXT_SYM || tok == token::CPU_SYM || tok == token::CREATE || tok == token::CURRENT_SYM || tok == token::CURSOR_NAME_SYM || tok == token::DATAFILE_SYM || tok == token::DATA_SYM || tok == token::DATETIME_SYM || tok == token::DATE_SYM || tok == token::DAY_SYM || tok == token::DEALLOCATE_SYM || tok == token::DECIMAL_NUM || tok == token::DEFAULT_AUTH_SYM || tok == token::DEFINER_SYM || tok == token::DELAY_KEY_WRITE_SYM || tok == token::DESC || tok == token::DIAGNOSTICS_SYM || tok == token::BYTEDICT_SYM || tok == token::SHORTDICT_SYM || tok == token::INTDICT_SYM || tok == token::DIRECTORY_SYM || tok == token::DISABLE_SYM || tok == token::DISCARD_SYM || tok == token::DISK_SYM || tok == token::DO_SYM || tok == token::DUMPFILE || tok == token::DUPLICATE_SYM || tok == token::DYNAMIC_SYM || tok == token::ENABLE_SYM || tok == token::END || tok == token::ENDS_SYM || tok == token::ENGINES_SYM || tok == token::ENGINE_SYM || tok == token::ENUM_SYM || tok == token::ERROR_SYM || tok == token::ERRORS || tok == token::ESCAPE_SYM || tok == token::EVENTS_SYM || tok == token::EVENT_SYM || tok == token::EVERY_SYM || tok == token::EXCHANGE_SYM || tok == token::EXECUTE_SYM || tok == token::EXPANSION_SYM || tok == token::EXPIRE_SYM || tok == token::EXPORT_SYM || tok == token::EXTENDED_SYM || tok == token::EXTENT_SIZE_SYM || tok == token::FAST_SYM || tok == token::FAULTS_SYM || tok == token::FILE_SYM || tok == token::FILE_BLOCK_SIZE_SYM || tok == token::FILTER_SYM || tok == token::FIRST_SYM || tok == token::FIXED_SYM || tok == token::FLOAT_NUM || tok == token::FLUSH_SYM || tok == token::FOLLOWS_SYM || tok == token::FORMAT_SYM || tok == token::FOUND_SYM || tok == token::FULL || tok == token::GENERAL || tok == token::GROUP_REPLICATION || tok == token::GEOMETRYCOLLECTION_SYM || tok == token::GEOMETRY_SYM || tok == token::GET_FORMAT || tok == token::GLOBAL_SYM || tok == token::GRANTS || tok == token::HANDLER_SYM || tok == token::HASH_SYM || tok == token::HELP_SYM || tok == token::HEX_NUM || tok == token::HOST_SYM || tok == token::HOSTS_SYM || tok == token::HOUR_SYM || tok == token::IDENT || tok == token::IDENTIFIED_SYM || tok == token::IGNORE_SERVER_IDS_SYM || tok == token::IMPORT || tok == token::INDEXES || tok == token::INITIAL_SIZE_SYM || tok == token::INSERT_METHOD || tok == token::INSTANCE_SYM || tok == token::INSTALL_SYM || tok == token::INTERVAL_SYM || tok == token::INVOKER_SYM || tok == token::IO_SYM || tok == token::IPC_SYM || tok == token::ISOLATION || tok == token::ISSUER_SYM || tok == token::JSON_SYM || tok == token::KEY_BLOCK_SIZE || tok == token::LANGUAGE_SYM || tok == token::LAST_SYM || tok == token::LEAVES || tok == token::LESS_SYM || tok == token::LEVEL_SYM || tok == token::LEX_HOSTNAME || tok == token::LINESTRING_SYM || tok == token::LIST_SYM || tok == token::LOCAL_SYM || tok == token::LOCKS_SYM || tok == token::LOGFILE_SYM || tok == token::LOGS_SYM || tok == token::LONG_NUM || tok == token::MASTER_AUTO_POSITION_SYM || tok == token::MASTER_CONNECT_RETRY_SYM || tok == token::MASTER_DELAY_SYM || tok == token::MASTER_HOST_SYM || tok == token::MASTER_LOG_FILE_SYM || tok == token::MASTER_LOG_POS_SYM || tok == token::MASTER_PASSWORD_SYM || tok == token::MASTER_PORT_SYM || tok == token::MASTER_RETRY_COUNT_SYM || tok == token::MASTER_SERVER_ID_SYM || tok == token::MASTER_SSL_CAPATH_SYM || tok == token::MASTER_TLS_VERSION_SYM || tok == token::MASTER_SSL_CA_SYM || tok == token::MASTER_SSL_CERT_SYM || tok == token::MASTER_SSL_CIPHER_SYM || tok == token::MASTER_SSL_CRL_SYM || tok == token::MASTER_SSL_CRLPATH_SYM || tok == token::MASTER_SSL_KEY_SYM || tok == token::MASTER_SSL_SYM || tok == token::MASTER_SYM || tok == token::MASTER_USER_SYM || tok == token::MASTER_HEARTBEAT_PERIOD_SYM || tok == token::MAX_CONNECTIONS_PER_HOUR || tok == token::MAX_QUERIES_PER_HOUR || tok == token::MAX_ROWS || tok == token::MAX_SIZE_SYM || tok == token::MAX_UPDATES_PER_HOUR || tok == token::MAX_USER_CONNECTIONS_SYM || tok == token::MEDIUM_SYM || tok == token::MEMORY_SYM || tok == token::MERGE_SYM || tok == token::MESSAGE_TEXT_SYM || tok == token::MICROSECOND_SYM || tok == token::MIGRATE_SYM || tok == token::MINUTE_SYM || tok == token::MIN_ROWS || tok == token::MODE_SYM || tok == token::MODIFY_SYM || tok == token::MONTH_SYM || tok == token::MULTILINESTRING_SYM || tok == token::MULTIPOINT_SYM || tok == token::MULTIPOLYGON_SYM || tok == token::MUTEX_SYM || tok == token::MYSQL_ERRNO_SYM || tok == token::NAMES_SYM || tok == token::NAME_SYM || tok == token::NATIONAL_SYM || tok == token::NCHAR_SYM || tok == token::NDBCLUSTER_SYM || tok == token::NEVER_SYM || tok == token::NEW_SYM || tok == token::NEXT_SYM || tok == token::NODEGROUP_SYM || tok == token::NONE_SYM || tok == token::NO_SYM || tok == token::NO_WAIT_SYM || tok == token::NUM || tok == token::NUMBER_SYM || tok == token::NVARCHAR_SYM || tok == token::OFFSET_SYM || tok == token::ONE_SYM || tok == token::ONLY_SYM || tok == token::OPEN_SYM || tok == token::OPTIONS_SYM || tok == token::OWNER_SYM || tok == token::PACK_KEYS_SYM || tok == token::PAGE_SYM || tok == token::PARSER_SYM || tok == token::PARTIAL || tok == token::PARTITIONS_SYM || tok == token::PARTITIONING_SYM || tok == token::PASSWORD || tok == token::PHASE_SYM || tok == token::PLUGIN_DIR_SYM || tok == token::PLUGIN_SYM || tok == token::PLUGINS_SYM || tok == token::POINT_SYM || tok == token::POLYGON_SYM || tok == token::PORT_SYM || tok == token::PRECEDES_SYM || tok == token::PREPARE_SYM || tok == token::PRESERVE_SYM || tok == token::PREV_SYM || tok == token::PRIVILEGES || tok == token::PROCESS || tok == token::PROCESSLIST_SYM || tok == token::PROFILE_SYM || tok == token::PROFILES_SYM || tok == token::PROXY_SYM || tok == token::QUARTER_SYM || tok == token::QUERY_SYM || tok == token::QUICK || tok == token::READ_ONLY_SYM || tok == token::REBUILD_SYM || tok == token::RECOVER_SYM || tok == token::REDO_BUFFER_SIZE_SYM || tok == token::REDUNDANT_SYM || tok == token::RELAY || tok == token::RELAYLOG_SYM || tok == token::RELAY_LOG_FILE_SYM || tok == token::RELAY_LOG_POS_SYM || tok == token::RELAY_THREAD || tok == token::RELOAD || tok == token::REMOVE_SYM || tok == token::REORGANIZE_SYM || tok == token::REPAIR || tok == token::REPEATABLE_SYM || tok == token::REPLICATION || tok == token::REPLICATE_DO_DB || tok == token::REPLICATE_IGNORE_DB || tok == token::REPLICATE_DO_TABLE || tok == token::REPLICATE_IGNORE_TABLE || tok == token::REPLICATE_WILD_DO_TABLE || tok == token::REPLICATE_WILD_IGNORE_TABLE || tok == token::REPLICATE_REWRITE_DB || tok == token::RESET_SYM || tok == token::RESOURCES || tok == token::RESTORE_SYM || tok == token::RESUME_SYM || tok == token::RETURNED_SQLSTATE_SYM || tok == token::RETURNS_SYM || tok == token::REVERSE_SYM || tok == token::ROLLBACK_SYM || tok == token::ROLLUP_SYM || tok == token::ROTATE_SYM || tok == token::ROUTINE_SYM || tok == token::ROW_FORMAT_SYM || tok == token::ROW_COUNT_SYM || tok == token::RTREE_SYM || tok == token::SAVEPOINT_SYM || tok == token::SCHEDULE_SYM || tok == token::SCHEMA_NAME_SYM || tok == token::SECOND_SYM || tok == token::SECURITY_SYM || tok == token::SERIALIZABLE_SYM || tok == token::SERIAL_SYM || tok == token::SESSION_SYM || tok == token::SERVER_SYM || tok == token::SHARE_SYM || tok == token::SHARES_SYM || tok == token::SHUTDOWN || tok == token::SIGNED_SYM || tok == token::SIMPLE_SYM || tok == token::SLAVE || tok == token::SLOW || tok == token::SNAPSHOT_SYM || tok == token::SOCKET_SYM || tok == token::SONAME_SYM || tok == token::SOUNDS_SYM || tok == token::SOURCE_SYM || tok == token::SQL_AFTER_GTIDS || tok == token::SQL_AFTER_MTS_GAPS || tok == token::SQL_BEFORE_GTIDS || tok == token::SQL_BUFFER_RESULT || tok == token::SQL_NO_CACHE_SYM || tok == token::SQL_THREAD || tok == token::STACKED_SYM || tok == token::STARTS_SYM || tok == token::START_SYM || tok == token::STATS_AUTO_RECALC_SYM || tok == token::STATS_PERSISTENT_SYM || tok == token::STATS_SAMPLE_PAGES_SYM || tok == token::STATUS_SYM || tok == token::STOP_SYM || tok == token::STORAGE_SYM || tok == token::STRING_SYM || tok == token::SUBCLASS_ORIGIN_SYM || tok == token::SUBDATE_SYM || tok == token::SUBJECT_SYM || tok == token::SUBPARTITIONS_SYM || tok == token::SUBPARTITION_SYM || tok == token::SUPER_SYM || tok == token::SUSPEND_SYM || tok == token::SWAPS_SYM || tok == token::SWITCHES_SYM || tok == token::TABLES || tok == token::VIEWS || tok == token::TABLESPACE_SYM || tok == token::TABLE_CHECKSUM_SYM || tok == token::TABLE_NAME_SYM || tok == token::TEMPORARY || tok == token::TEMPTABLE_SYM || tok == token::TEXT_STRING || tok == token::TEXT_SYM || tok == token::THAN_SYM || tok == token::TIMESTAMP_SYM || tok == token::TIMESTAMP_ADD || tok == token::TIMESTAMP_DIFF || tok == token::TIME_SYM || tok == token::TRANSACTION_SYM || tok == token::TRIGGERS_SYM || tok == token::TRUNCATE_SYM || tok == token::TYPES_SYM || tok == token::TYPE_SYM || tok == token::ULONGLONG_NUM || tok == token::UNCOMMITTED_SYM || tok == token::UNDEFINED_SYM || tok == token::UNDOFILE_SYM || tok == token::UNDO_BUFFER_SIZE_SYM || tok == token::UNICODE_SYM || tok == token::UNINSTALL_SYM || tok == token::UNKNOWN_SYM || tok == token::UNTIL_SYM || tok == token::UPGRADE_SYM || tok == token::USER || tok == token::USE_FRM || tok == token::VALIDATION_SYM || tok == token::VALUE_SYM || tok == token::VARBINARY_SYM || tok == token::VARIABLES || tok == token::VIEW_SYM || tok == token::WAIT_SYM || tok == token::WARNINGS || tok == token::WEEK_SYM || tok == token::WEIGHT_STRING_SYM || tok == token::WITHOUT_SYM || tok == token::WORK_SYM || tok == token::WRAPPER_SYM || tok == token::X509_SYM || tok == token::XA_SYM || tok == token::XID_SYM || tok == token::XML_SYM || tok == token::YEAR_SYM || tok == token::PERSIST_SYM || tok == token::ROLE_SYM || tok == token::ADMIN_SYM || tok == token::INVISIBLE_SYM || tok == token::VISIBLE_SYM || tok == token::COMPONENT_SYM || tok == token::SKIP_SYM || tok == token::LOCKED_SYM || tok == token::NOWAIT_SYM || tok == token::PERSIST_ONLY_SYM || tok == token::HISTOGRAM_SYM || tok == token::BUCKETS_SYM || tok == token::OBSOLETE_TOKEN_930 || tok == token::CLONE_SYM || tok == token::EXCLUDE_SYM || tok == token::FOLLOWING_SYM || tok == token::NULLS_SYM || tok == token::OTHERS_SYM || tok == token::PRECEDING_SYM || tok == token::RESPECT_SYM || tok == token::TIES_SYM || tok == token::UNBOUNDED_SYM || tok == token::NESTED_SYM || tok == token::ORDINALITY_SYM || tok == token::PATH_SYM || tok == token::HISTORY_SYM || tok == token::REUSE_SYM || tok == token::SRID_SYM || tok == token::THREAD_PRIORITY_SYM || tok == token::RESOURCE_SYM || tok == token::VCPU_SYM || tok == token::MASTER_PUBLIC_KEY_PATH_SYM || tok == token::GET_MASTER_PUBLIC_KEY_SYM || tok == token::RESTART_SYM || tok == token::DEFINITION_SYM || tok == token::DESCRIPTION_SYM || tok == token::ORGANIZATION_SYM || tok == token::REFERENCE_SYM || tok == token::ACTIVE_SYM || tok == token::INACTIVE_SYM || tok == token::OPTIONAL_SYM || tok == token::SECONDARY_SYM || tok == token::SECONDARY_ENGINE_SYM || tok == token::SECONDARY_LOAD_SYM || tok == token::SECONDARY_UNLOAD_SYM || tok == token::RETAIN_SYM || tok == token::OLD_SYM || tok == token::ENFORCED_SYM || tok == token::OJ_SYM || tok == token::NETWORK_NAMESPACE_SYM);
      }
#endif
    };

    /// Build a parser object.
    Parser (class Driver& driver_yyarg);
    virtual ~Parser ();

    /// Parse.  An alias for parse ().
    /// \returns  0 iff parsing succeeded.
    int operator() ();

    /// Parse.
    /// \returns  0 iff parsing succeeded.
    virtual int parse ();

#if ARIES_PARSERDEBUG
    /// The current debugging stream.
    std::ostream& debug_stream () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging stream.
    void set_debug_stream (std::ostream &);

    /// Type for debugging levels.
    typedef int debug_level_type;
    /// The current debugging level.
    debug_level_type debug_level () const YY_ATTRIBUTE_PURE;
    /// Set the current debugging level.
    void set_debug_level (debug_level_type l);
#endif

    /// Report a syntax error.
    /// \param loc    where the syntax error is found.
    /// \param msg    a description of the syntax error.
    virtual void error (const location_type& loc, const std::string& msg);

    /// Report a syntax error.
    void error (const syntax_error& err);

    // Implementation of make_symbol for each symbol type.
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_END_OF_INPUT (location_type l)
      {
        return symbol_type (token::END_OF_INPUT, std::move (l));
      }
#else
      static
      symbol_type
      make_END_OF_INPUT (const location_type& l)
      {
        return symbol_type (token::END_OF_INPUT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ABORT_SYM (location_type l)
      {
        return symbol_type (token::ABORT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ABORT_SYM (const location_type& l)
      {
        return symbol_type (token::ABORT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACCESSIBLE_SYM (location_type l)
      {
        return symbol_type (token::ACCESSIBLE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ACCESSIBLE_SYM (const location_type& l)
      {
        return symbol_type (token::ACCESSIBLE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACCOUNT_SYM (string v, location_type l)
      {
        return symbol_type (token::ACCOUNT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ACCOUNT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ACCOUNT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACTION (string v, location_type l)
      {
        return symbol_type (token::ACTION, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ACTION (const string& v, const location_type& l)
      {
        return symbol_type (token::ACTION, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ADD (location_type l)
      {
        return symbol_type (token::ADD, std::move (l));
      }
#else
      static
      symbol_type
      make_ADD (const location_type& l)
      {
        return symbol_type (token::ADD, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ADDDATE_SYM (string v, location_type l)
      {
        return symbol_type (token::ADDDATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ADDDATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ADDDATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AFTER_SYM (string v, location_type l)
      {
        return symbol_type (token::AFTER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AFTER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::AFTER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AGAINST (string v, location_type l)
      {
        return symbol_type (token::AGAINST, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AGAINST (const string& v, const location_type& l)
      {
        return symbol_type (token::AGAINST, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AGGREGATE_SYM (string v, location_type l)
      {
        return symbol_type (token::AGGREGATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AGGREGATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::AGGREGATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ALGORITHM_SYM (string v, location_type l)
      {
        return symbol_type (token::ALGORITHM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ALGORITHM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ALGORITHM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ALL (location_type l)
      {
        return symbol_type (token::ALL, std::move (l));
      }
#else
      static
      symbol_type
      make_ALL (const location_type& l)
      {
        return symbol_type (token::ALL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ALTER (location_type l)
      {
        return symbol_type (token::ALTER, std::move (l));
      }
#else
      static
      symbol_type
      make_ALTER (const location_type& l)
      {
        return symbol_type (token::ALTER, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ALWAYS_SYM (string v, location_type l)
      {
        return symbol_type (token::ALWAYS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ALWAYS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ALWAYS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_271 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_271, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_271 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_271, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ANALYZE_SYM (location_type l)
      {
        return symbol_type (token::ANALYZE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ANALYZE_SYM (const location_type& l)
      {
        return symbol_type (token::ANALYZE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AND_AND_SYM (location_type l)
      {
        return symbol_type (token::AND_AND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_AND_AND_SYM (const location_type& l)
      {
        return symbol_type (token::AND_AND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AND_SYM (location_type l)
      {
        return symbol_type (token::AND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_AND_SYM (const location_type& l)
      {
        return symbol_type (token::AND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ANY_SYM (string v, location_type l)
      {
        return symbol_type (token::ANY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ANY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ANY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AS (location_type l)
      {
        return symbol_type (token::AS, std::move (l));
      }
#else
      static
      symbol_type
      make_AS (const location_type& l)
      {
        return symbol_type (token::AS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ASC (string v, location_type l)
      {
        return symbol_type (token::ASC, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ASC (const string& v, const location_type& l)
      {
        return symbol_type (token::ASC, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ASCII_SYM (string v, location_type l)
      {
        return symbol_type (token::ASCII_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ASCII_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ASCII_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ASENSITIVE_SYM (location_type l)
      {
        return symbol_type (token::ASENSITIVE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ASENSITIVE_SYM (const location_type& l)
      {
        return symbol_type (token::ASENSITIVE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AT_SYM (string v, location_type l)
      {
        return symbol_type (token::AT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::AT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AUTOEXTEND_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::AUTOEXTEND_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AUTOEXTEND_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::AUTOEXTEND_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AUTO_INC (string v, location_type l)
      {
        return symbol_type (token::AUTO_INC, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AUTO_INC (const string& v, const location_type& l)
      {
        return symbol_type (token::AUTO_INC, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AVG_ROW_LENGTH (string v, location_type l)
      {
        return symbol_type (token::AVG_ROW_LENGTH, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AVG_ROW_LENGTH (const string& v, const location_type& l)
      {
        return symbol_type (token::AVG_ROW_LENGTH, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_AVG_SYM (string v, location_type l)
      {
        return symbol_type (token::AVG_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_AVG_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::AVG_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BACKUP_SYM (string v, location_type l)
      {
        return symbol_type (token::BACKUP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BACKUP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BACKUP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BEFORE_SYM (location_type l)
      {
        return symbol_type (token::BEFORE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_BEFORE_SYM (const location_type& l)
      {
        return symbol_type (token::BEFORE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BEGIN_SYM (string v, location_type l)
      {
        return symbol_type (token::BEGIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BEGIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BEGIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BETWEEN_SYM (location_type l)
      {
        return symbol_type (token::BETWEEN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_BETWEEN_SYM (const location_type& l)
      {
        return symbol_type (token::BETWEEN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIGINT_SYM (location_type l)
      {
        return symbol_type (token::BIGINT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_BIGINT_SYM (const location_type& l)
      {
        return symbol_type (token::BIGINT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BINARY_SYM (location_type l)
      {
        return symbol_type (token::BINARY_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_BINARY_SYM (const location_type& l)
      {
        return symbol_type (token::BINARY_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BINLOG_SYM (string v, location_type l)
      {
        return symbol_type (token::BINLOG_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BINLOG_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BINLOG_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIN_NUM (location_type l)
      {
        return symbol_type (token::BIN_NUM, std::move (l));
      }
#else
      static
      symbol_type
      make_BIN_NUM (const location_type& l)
      {
        return symbol_type (token::BIN_NUM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIT_AND (location_type l)
      {
        return symbol_type (token::BIT_AND, std::move (l));
      }
#else
      static
      symbol_type
      make_BIT_AND (const location_type& l)
      {
        return symbol_type (token::BIT_AND, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIT_OR (location_type l)
      {
        return symbol_type (token::BIT_OR, std::move (l));
      }
#else
      static
      symbol_type
      make_BIT_OR (const location_type& l)
      {
        return symbol_type (token::BIT_OR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIT_SYM (string v, location_type l)
      {
        return symbol_type (token::BIT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BIT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BIT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BIT_XOR (location_type l)
      {
        return symbol_type (token::BIT_XOR, std::move (l));
      }
#else
      static
      symbol_type
      make_BIT_XOR (const location_type& l)
      {
        return symbol_type (token::BIT_XOR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BLOB_SYM (location_type l)
      {
        return symbol_type (token::BLOB_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_BLOB_SYM (const location_type& l)
      {
        return symbol_type (token::BLOB_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BLOCK_SYM (string v, location_type l)
      {
        return symbol_type (token::BLOCK_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BLOCK_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BLOCK_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BOOLEAN_SYM (string v, location_type l)
      {
        return symbol_type (token::BOOLEAN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BOOLEAN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BOOLEAN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BOOL_SYM (string v, location_type l)
      {
        return symbol_type (token::BOOL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BOOL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BOOL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BOTH (location_type l)
      {
        return symbol_type (token::BOTH, std::move (l));
      }
#else
      static
      symbol_type
      make_BOTH (const location_type& l)
      {
        return symbol_type (token::BOTH, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BTREE_SYM (string v, location_type l)
      {
        return symbol_type (token::BTREE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BTREE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BTREE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BY (location_type l)
      {
        return symbol_type (token::BY, std::move (l));
      }
#else
      static
      symbol_type
      make_BY (const location_type& l)
      {
        return symbol_type (token::BY, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BYTE_SYM (string v, location_type l)
      {
        return symbol_type (token::BYTE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BYTE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BYTE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CACHE_SYM (string v, location_type l)
      {
        return symbol_type (token::CACHE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CACHE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CACHE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CALL_SYM (location_type l)
      {
        return symbol_type (token::CALL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CALL_SYM (const location_type& l)
      {
        return symbol_type (token::CALL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CASCADE (location_type l)
      {
        return symbol_type (token::CASCADE, std::move (l));
      }
#else
      static
      symbol_type
      make_CASCADE (const location_type& l)
      {
        return symbol_type (token::CASCADE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CASCADED (string v, location_type l)
      {
        return symbol_type (token::CASCADED, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CASCADED (const string& v, const location_type& l)
      {
        return symbol_type (token::CASCADED, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CASE_SYM (location_type l)
      {
        return symbol_type (token::CASE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CASE_SYM (const location_type& l)
      {
        return symbol_type (token::CASE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CAST_SYM (location_type l)
      {
        return symbol_type (token::CAST_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CAST_SYM (const location_type& l)
      {
        return symbol_type (token::CAST_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CATALOG_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::CATALOG_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CATALOG_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CATALOG_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHAIN_SYM (string v, location_type l)
      {
        return symbol_type (token::CHAIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHAIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CHAIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHANGE (location_type l)
      {
        return symbol_type (token::CHANGE, std::move (l));
      }
#else
      static
      symbol_type
      make_CHANGE (const location_type& l)
      {
        return symbol_type (token::CHANGE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHANGED (string v, location_type l)
      {
        return symbol_type (token::CHANGED, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHANGED (const string& v, const location_type& l)
      {
        return symbol_type (token::CHANGED, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHANNEL_SYM (string v, location_type l)
      {
        return symbol_type (token::CHANNEL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHANNEL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CHANNEL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHARSET (string v, location_type l)
      {
        return symbol_type (token::CHARSET, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHARSET (const string& v, const location_type& l)
      {
        return symbol_type (token::CHARSET, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHAR_SYM (location_type l)
      {
        return symbol_type (token::CHAR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CHAR_SYM (const location_type& l)
      {
        return symbol_type (token::CHAR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHECKSUM_SYM (string v, location_type l)
      {
        return symbol_type (token::CHECKSUM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CHECKSUM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CHECKSUM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CHECK_SYM (location_type l)
      {
        return symbol_type (token::CHECK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CHECK_SYM (const location_type& l)
      {
        return symbol_type (token::CHECK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CIPHER_SYM (string v, location_type l)
      {
        return symbol_type (token::CIPHER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CIPHER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CIPHER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CLASS_ORIGIN_SYM (string v, location_type l)
      {
        return symbol_type (token::CLASS_ORIGIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CLASS_ORIGIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CLASS_ORIGIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CLIENT_SYM (string v, location_type l)
      {
        return symbol_type (token::CLIENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CLIENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CLIENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CLOSE_SYM (string v, location_type l)
      {
        return symbol_type (token::CLOSE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CLOSE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CLOSE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COALESCE (string v, location_type l)
      {
        return symbol_type (token::COALESCE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COALESCE (const string& v, const location_type& l)
      {
        return symbol_type (token::COALESCE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CODE_SYM (string v, location_type l)
      {
        return symbol_type (token::CODE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CODE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CODE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLLATE_SYM (location_type l)
      {
        return symbol_type (token::COLLATE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_COLLATE_SYM (const location_type& l)
      {
        return symbol_type (token::COLLATE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLLATION_SYM (string v, location_type l)
      {
        return symbol_type (token::COLLATION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COLLATION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COLLATION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLUMNS (string v, location_type l)
      {
        return symbol_type (token::COLUMNS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COLUMNS (const string& v, const location_type& l)
      {
        return symbol_type (token::COLUMNS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLUMN_SYM (location_type l)
      {
        return symbol_type (token::COLUMN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_COLUMN_SYM (const location_type& l)
      {
        return symbol_type (token::COLUMN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLUMN_FORMAT_SYM (string v, location_type l)
      {
        return symbol_type (token::COLUMN_FORMAT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COLUMN_FORMAT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COLUMN_FORMAT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COLUMN_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::COLUMN_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COLUMN_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COLUMN_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMMENT_SYM (string v, location_type l)
      {
        return symbol_type (token::COMMENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMMENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMMENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMMITTED_SYM (string v, location_type l)
      {
        return symbol_type (token::COMMITTED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMMITTED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMMITTED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMMIT_SYM (string v, location_type l)
      {
        return symbol_type (token::COMMIT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMMIT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMMIT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPACT_SYM (string v, location_type l)
      {
        return symbol_type (token::COMPACT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPACT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMPACT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPLETION_SYM (string v, location_type l)
      {
        return symbol_type (token::COMPLETION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPLETION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMPLETION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPRESSED_SYM (string v, location_type l)
      {
        return symbol_type (token::COMPRESSED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPRESSED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMPRESSED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPRESSION_SYM (string v, location_type l)
      {
        return symbol_type (token::COMPRESSION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPRESSION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMPRESSION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENCRYPTION_SYM (string v, location_type l)
      {
        return symbol_type (token::ENCRYPTION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENCRYPTION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENCRYPTION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONCURRENT (string v, location_type l)
      {
        return symbol_type (token::CONCURRENT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONCURRENT (const string& v, const location_type& l)
      {
        return symbol_type (token::CONCURRENT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONDITION_SYM (location_type l)
      {
        return symbol_type (token::CONDITION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CONDITION_SYM (const location_type& l)
      {
        return symbol_type (token::CONDITION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONNECTION_ID_SYM (location_type l)
      {
        return symbol_type (token::CONNECTION_ID_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CONNECTION_ID_SYM (const location_type& l)
      {
        return symbol_type (token::CONNECTION_ID_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONNECTION_SYM (string v, location_type l)
      {
        return symbol_type (token::CONNECTION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONNECTION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONNECTION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONSISTENT_SYM (string v, location_type l)
      {
        return symbol_type (token::CONSISTENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONSISTENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONSISTENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONSTRAINT (location_type l)
      {
        return symbol_type (token::CONSTRAINT, std::move (l));
      }
#else
      static
      symbol_type
      make_CONSTRAINT (const location_type& l)
      {
        return symbol_type (token::CONSTRAINT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONSTRAINT_CATALOG_SYM (string v, location_type l)
      {
        return symbol_type (token::CONSTRAINT_CATALOG_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONSTRAINT_CATALOG_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONSTRAINT_CATALOG_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONSTRAINT_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::CONSTRAINT_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONSTRAINT_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONSTRAINT_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONSTRAINT_SCHEMA_SYM (string v, location_type l)
      {
        return symbol_type (token::CONSTRAINT_SCHEMA_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONSTRAINT_SCHEMA_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONSTRAINT_SCHEMA_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONTAINS_SYM (string v, location_type l)
      {
        return symbol_type (token::CONTAINS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONTAINS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONTAINS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONTEXT_SYM (string v, location_type l)
      {
        return symbol_type (token::CONTEXT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CONTEXT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CONTEXT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONTINUE_SYM (location_type l)
      {
        return symbol_type (token::CONTINUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CONTINUE_SYM (const location_type& l)
      {
        return symbol_type (token::CONTINUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONVERT_SYM (location_type l)
      {
        return symbol_type (token::CONVERT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CONVERT_SYM (const location_type& l)
      {
        return symbol_type (token::CONVERT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COUNT_SYM (location_type l)
      {
        return symbol_type (token::COUNT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_COUNT_SYM (const location_type& l)
      {
        return symbol_type (token::COUNT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CPU_SYM (string v, location_type l)
      {
        return symbol_type (token::CPU_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CPU_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CPU_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CREATE (string v, location_type l)
      {
        return symbol_type (token::CREATE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CREATE (const string& v, const location_type& l)
      {
        return symbol_type (token::CREATE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CROSS (location_type l)
      {
        return symbol_type (token::CROSS, std::move (l));
      }
#else
      static
      symbol_type
      make_CROSS (const location_type& l)
      {
        return symbol_type (token::CROSS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CUBE_SYM (location_type l)
      {
        return symbol_type (token::CUBE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CUBE_SYM (const location_type& l)
      {
        return symbol_type (token::CUBE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURDATE (location_type l)
      {
        return symbol_type (token::CURDATE, std::move (l));
      }
#else
      static
      symbol_type
      make_CURDATE (const location_type& l)
      {
        return symbol_type (token::CURDATE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURRENT_SYM (string v, location_type l)
      {
        return symbol_type (token::CURRENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CURRENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CURRENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURRENT_USER (location_type l)
      {
        return symbol_type (token::CURRENT_USER, std::move (l));
      }
#else
      static
      symbol_type
      make_CURRENT_USER (const location_type& l)
      {
        return symbol_type (token::CURRENT_USER, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURSOR_SYM (location_type l)
      {
        return symbol_type (token::CURSOR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CURSOR_SYM (const location_type& l)
      {
        return symbol_type (token::CURSOR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURSOR_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::CURSOR_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CURSOR_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CURSOR_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CURTIME (location_type l)
      {
        return symbol_type (token::CURTIME, std::move (l));
      }
#else
      static
      symbol_type
      make_CURTIME (const location_type& l)
      {
        return symbol_type (token::CURTIME, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATABASE (location_type l)
      {
        return symbol_type (token::DATABASE, std::move (l));
      }
#else
      static
      symbol_type
      make_DATABASE (const location_type& l)
      {
        return symbol_type (token::DATABASE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATABASES (location_type l)
      {
        return symbol_type (token::DATABASES, std::move (l));
      }
#else
      static
      symbol_type
      make_DATABASES (const location_type& l)
      {
        return symbol_type (token::DATABASES, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATAFILE_SYM (string v, location_type l)
      {
        return symbol_type (token::DATAFILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DATAFILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DATAFILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATA_SYM (string v, location_type l)
      {
        return symbol_type (token::DATA_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DATA_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DATA_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATETIME_SYM (string v, location_type l)
      {
        return symbol_type (token::DATETIME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DATETIME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DATETIME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATE_ADD_INTERVAL (location_type l)
      {
        return symbol_type (token::DATE_ADD_INTERVAL, std::move (l));
      }
#else
      static
      symbol_type
      make_DATE_ADD_INTERVAL (const location_type& l)
      {
        return symbol_type (token::DATE_ADD_INTERVAL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATE_SUB_INTERVAL (location_type l)
      {
        return symbol_type (token::DATE_SUB_INTERVAL, std::move (l));
      }
#else
      static
      symbol_type
      make_DATE_SUB_INTERVAL (const location_type& l)
      {
        return symbol_type (token::DATE_SUB_INTERVAL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DATE_SYM (string v, location_type l)
      {
        return symbol_type (token::DATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DAY_HOUR_SYM (location_type l)
      {
        return symbol_type (token::DAY_HOUR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DAY_HOUR_SYM (const location_type& l)
      {
        return symbol_type (token::DAY_HOUR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DAY_MICROSECOND_SYM (location_type l)
      {
        return symbol_type (token::DAY_MICROSECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DAY_MICROSECOND_SYM (const location_type& l)
      {
        return symbol_type (token::DAY_MICROSECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DAY_MINUTE_SYM (location_type l)
      {
        return symbol_type (token::DAY_MINUTE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DAY_MINUTE_SYM (const location_type& l)
      {
        return symbol_type (token::DAY_MINUTE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DAY_SECOND_SYM (location_type l)
      {
        return symbol_type (token::DAY_SECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DAY_SECOND_SYM (const location_type& l)
      {
        return symbol_type (token::DAY_SECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DAY_SYM (string v, location_type l)
      {
        return symbol_type (token::DAY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DAY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DAY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEALLOCATE_SYM (string v, location_type l)
      {
        return symbol_type (token::DEALLOCATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DEALLOCATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DEALLOCATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DECIMAL_NUM (string v, location_type l)
      {
        return symbol_type (token::DECIMAL_NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DECIMAL_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::DECIMAL_NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REAL_NUM (location_type l)
      {
        return symbol_type (token::REAL_NUM, std::move (l));
      }
#else
      static
      symbol_type
      make_REAL_NUM (const location_type& l)
      {
        return symbol_type (token::REAL_NUM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DECIMAL_SYM (location_type l)
      {
        return symbol_type (token::DECIMAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DECIMAL_SYM (const location_type& l)
      {
        return symbol_type (token::DECIMAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DECLARE_SYM (location_type l)
      {
        return symbol_type (token::DECLARE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DECLARE_SYM (const location_type& l)
      {
        return symbol_type (token::DECLARE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEFAULT_SYM (location_type l)
      {
        return symbol_type (token::DEFAULT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DEFAULT_SYM (const location_type& l)
      {
        return symbol_type (token::DEFAULT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEFAULT_AUTH_SYM (string v, location_type l)
      {
        return symbol_type (token::DEFAULT_AUTH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DEFAULT_AUTH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DEFAULT_AUTH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEFINER_SYM (string v, location_type l)
      {
        return symbol_type (token::DEFINER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DEFINER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DEFINER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DELAYED_SYM (location_type l)
      {
        return symbol_type (token::DELAYED_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DELAYED_SYM (const location_type& l)
      {
        return symbol_type (token::DELAYED_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DELAY_KEY_WRITE_SYM (string v, location_type l)
      {
        return symbol_type (token::DELAY_KEY_WRITE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DELAY_KEY_WRITE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DELAY_KEY_WRITE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DELETE_SYM (location_type l)
      {
        return symbol_type (token::DELETE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DELETE_SYM (const location_type& l)
      {
        return symbol_type (token::DELETE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DESC (string v, location_type l)
      {
        return symbol_type (token::DESC, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DESC (const string& v, const location_type& l)
      {
        return symbol_type (token::DESC, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DESCRIBE (location_type l)
      {
        return symbol_type (token::DESCRIBE, std::move (l));
      }
#else
      static
      symbol_type
      make_DESCRIBE (const location_type& l)
      {
        return symbol_type (token::DESCRIBE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_388 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_388, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_388 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_388, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DETERMINISTIC_SYM (location_type l)
      {
        return symbol_type (token::DETERMINISTIC_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DETERMINISTIC_SYM (const location_type& l)
      {
        return symbol_type (token::DETERMINISTIC_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DIAGNOSTICS_SYM (string v, location_type l)
      {
        return symbol_type (token::DIAGNOSTICS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DIAGNOSTICS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DIAGNOSTICS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BYTEDICT_SYM (string v, location_type l)
      {
        return symbol_type (token::BYTEDICT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BYTEDICT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BYTEDICT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHORTDICT_SYM (string v, location_type l)
      {
        return symbol_type (token::SHORTDICT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SHORTDICT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SHORTDICT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INTDICT_SYM (string v, location_type l)
      {
        return symbol_type (token::INTDICT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INTDICT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INTDICT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DICT_INDEX_SYM (location_type l)
      {
        return symbol_type (token::DICT_INDEX_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DICT_INDEX_SYM (const location_type& l)
      {
        return symbol_type (token::DICT_INDEX_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DIRECTORY_SYM (string v, location_type l)
      {
        return symbol_type (token::DIRECTORY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DIRECTORY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DIRECTORY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DISABLE_SYM (string v, location_type l)
      {
        return symbol_type (token::DISABLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DISABLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DISABLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DISCARD_SYM (string v, location_type l)
      {
        return symbol_type (token::DISCARD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DISCARD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DISCARD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DISK_SYM (string v, location_type l)
      {
        return symbol_type (token::DISK_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DISK_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DISK_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DISTINCT (location_type l)
      {
        return symbol_type (token::DISTINCT, std::move (l));
      }
#else
      static
      symbol_type
      make_DISTINCT (const location_type& l)
      {
        return symbol_type (token::DISTINCT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DIV_SYM (location_type l)
      {
        return symbol_type (token::DIV_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DIV_SYM (const location_type& l)
      {
        return symbol_type (token::DIV_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DOUBLE_SYM (location_type l)
      {
        return symbol_type (token::DOUBLE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DOUBLE_SYM (const location_type& l)
      {
        return symbol_type (token::DOUBLE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DO_SYM (string v, location_type l)
      {
        return symbol_type (token::DO_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DO_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DO_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DROP (location_type l)
      {
        return symbol_type (token::DROP, std::move (l));
      }
#else
      static
      symbol_type
      make_DROP (const location_type& l)
      {
        return symbol_type (token::DROP, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DUAL_SYM (location_type l)
      {
        return symbol_type (token::DUAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DUAL_SYM (const location_type& l)
      {
        return symbol_type (token::DUAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DUMPFILE (string v, location_type l)
      {
        return symbol_type (token::DUMPFILE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DUMPFILE (const string& v, const location_type& l)
      {
        return symbol_type (token::DUMPFILE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DUPLICATE_SYM (string v, location_type l)
      {
        return symbol_type (token::DUPLICATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DUPLICATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DUPLICATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DYNAMIC_SYM (string v, location_type l)
      {
        return symbol_type (token::DYNAMIC_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DYNAMIC_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DYNAMIC_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EACH_SYM (location_type l)
      {
        return symbol_type (token::EACH_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EACH_SYM (const location_type& l)
      {
        return symbol_type (token::EACH_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ELSE (location_type l)
      {
        return symbol_type (token::ELSE, std::move (l));
      }
#else
      static
      symbol_type
      make_ELSE (const location_type& l)
      {
        return symbol_type (token::ELSE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ELSEIF_SYM (location_type l)
      {
        return symbol_type (token::ELSEIF_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ELSEIF_SYM (const location_type& l)
      {
        return symbol_type (token::ELSEIF_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENABLE_SYM (string v, location_type l)
      {
        return symbol_type (token::ENABLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENABLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENABLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENCLOSED (location_type l)
      {
        return symbol_type (token::ENCLOSED, std::move (l));
      }
#else
      static
      symbol_type
      make_ENCLOSED (const location_type& l)
      {
        return symbol_type (token::ENCLOSED, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENCODING (location_type l)
      {
        return symbol_type (token::ENCODING, std::move (l));
      }
#else
      static
      symbol_type
      make_ENCODING (const location_type& l)
      {
        return symbol_type (token::ENCODING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_END (string v, location_type l)
      {
        return symbol_type (token::END, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_END (const string& v, const location_type& l)
      {
        return symbol_type (token::END, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENDS_SYM (string v, location_type l)
      {
        return symbol_type (token::ENDS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENDS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENDS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENGINES_SYM (string v, location_type l)
      {
        return symbol_type (token::ENGINES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENGINES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENGINES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENGINE_SYM (string v, location_type l)
      {
        return symbol_type (token::ENGINE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENGINE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENGINE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENUM_SYM (string v, location_type l)
      {
        return symbol_type (token::ENUM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENUM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENUM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EQ (location_type l)
      {
        return symbol_type (token::EQ, std::move (l));
      }
#else
      static
      symbol_type
      make_EQ (const location_type& l)
      {
        return symbol_type (token::EQ, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EQUAL_SYM (location_type l)
      {
        return symbol_type (token::EQUAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EQUAL_SYM (const location_type& l)
      {
        return symbol_type (token::EQUAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ERROR_SYM (string v, location_type l)
      {
        return symbol_type (token::ERROR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ERROR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ERROR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ERRORS (string v, location_type l)
      {
        return symbol_type (token::ERRORS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ERRORS (const string& v, const location_type& l)
      {
        return symbol_type (token::ERRORS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ESCAPED (location_type l)
      {
        return symbol_type (token::ESCAPED, std::move (l));
      }
#else
      static
      symbol_type
      make_ESCAPED (const location_type& l)
      {
        return symbol_type (token::ESCAPED, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ESCAPE_SYM (string v, location_type l)
      {
        return symbol_type (token::ESCAPE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ESCAPE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ESCAPE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EVENTS_SYM (string v, location_type l)
      {
        return symbol_type (token::EVENTS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EVENTS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EVENTS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EVENT_SYM (string v, location_type l)
      {
        return symbol_type (token::EVENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EVENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EVENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EVERY_SYM (string v, location_type l)
      {
        return symbol_type (token::EVERY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EVERY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EVERY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXCHANGE_SYM (string v, location_type l)
      {
        return symbol_type (token::EXCHANGE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXCHANGE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXCHANGE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXECUTE_SYM (string v, location_type l)
      {
        return symbol_type (token::EXECUTE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXECUTE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXECUTE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXISTS (location_type l)
      {
        return symbol_type (token::EXISTS, std::move (l));
      }
#else
      static
      symbol_type
      make_EXISTS (const location_type& l)
      {
        return symbol_type (token::EXISTS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXIT_SYM (location_type l)
      {
        return symbol_type (token::EXIT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EXIT_SYM (const location_type& l)
      {
        return symbol_type (token::EXIT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXPANSION_SYM (string v, location_type l)
      {
        return symbol_type (token::EXPANSION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXPANSION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXPANSION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXPIRE_SYM (string v, location_type l)
      {
        return symbol_type (token::EXPIRE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXPIRE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXPIRE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXPORT_SYM (string v, location_type l)
      {
        return symbol_type (token::EXPORT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXPORT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXPORT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXTENDED_SYM (string v, location_type l)
      {
        return symbol_type (token::EXTENDED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXTENDED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXTENDED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXTENT_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::EXTENT_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXTENT_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXTENT_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXTRACT_SYM (location_type l)
      {
        return symbol_type (token::EXTRACT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EXTRACT_SYM (const location_type& l)
      {
        return symbol_type (token::EXTRACT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FALSE_SYM (location_type l)
      {
        return symbol_type (token::FALSE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FALSE_SYM (const location_type& l)
      {
        return symbol_type (token::FALSE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FAST_SYM (string v, location_type l)
      {
        return symbol_type (token::FAST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FAST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FAST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FAULTS_SYM (string v, location_type l)
      {
        return symbol_type (token::FAULTS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FAULTS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FAULTS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FETCH_SYM (location_type l)
      {
        return symbol_type (token::FETCH_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FETCH_SYM (const location_type& l)
      {
        return symbol_type (token::FETCH_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FILE_SYM (string v, location_type l)
      {
        return symbol_type (token::FILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FILE_BLOCK_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::FILE_BLOCK_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FILE_BLOCK_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FILE_BLOCK_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FILTER_SYM (string v, location_type l)
      {
        return symbol_type (token::FILTER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FILTER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FILTER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FIRST_SYM (string v, location_type l)
      {
        return symbol_type (token::FIRST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FIRST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FIRST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FIXED_SYM (string v, location_type l)
      {
        return symbol_type (token::FIXED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FIXED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FIXED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FLOAT_NUM (string v, location_type l)
      {
        return symbol_type (token::FLOAT_NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FLOAT_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::FLOAT_NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FLOAT_SYM (location_type l)
      {
        return symbol_type (token::FLOAT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FLOAT_SYM (const location_type& l)
      {
        return symbol_type (token::FLOAT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FLUSH_SYM (string v, location_type l)
      {
        return symbol_type (token::FLUSH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FLUSH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FLUSH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FOLLOWS_SYM (string v, location_type l)
      {
        return symbol_type (token::FOLLOWS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FOLLOWS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FOLLOWS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FORCE_SYM (location_type l)
      {
        return symbol_type (token::FORCE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FORCE_SYM (const location_type& l)
      {
        return symbol_type (token::FORCE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FOREIGN (location_type l)
      {
        return symbol_type (token::FOREIGN, std::move (l));
      }
#else
      static
      symbol_type
      make_FOREIGN (const location_type& l)
      {
        return symbol_type (token::FOREIGN, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FOR_SYM (location_type l)
      {
        return symbol_type (token::FOR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FOR_SYM (const location_type& l)
      {
        return symbol_type (token::FOR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FORMAT_SYM (string v, location_type l)
      {
        return symbol_type (token::FORMAT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FORMAT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FORMAT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FOUND_SYM (string v, location_type l)
      {
        return symbol_type (token::FOUND_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FOUND_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FOUND_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FROM (location_type l)
      {
        return symbol_type (token::FROM, std::move (l));
      }
#else
      static
      symbol_type
      make_FROM (const location_type& l)
      {
        return symbol_type (token::FROM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FULL (string v, location_type l)
      {
        return symbol_type (token::FULL, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FULL (const string& v, const location_type& l)
      {
        return symbol_type (token::FULL, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FULLTEXT_SYM (location_type l)
      {
        return symbol_type (token::FULLTEXT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FULLTEXT_SYM (const location_type& l)
      {
        return symbol_type (token::FULLTEXT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FUNCTION_SYM (location_type l)
      {
        return symbol_type (token::FUNCTION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FUNCTION_SYM (const location_type& l)
      {
        return symbol_type (token::FUNCTION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GE (location_type l)
      {
        return symbol_type (token::GE, std::move (l));
      }
#else
      static
      symbol_type
      make_GE (const location_type& l)
      {
        return symbol_type (token::GE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GENERAL (string v, location_type l)
      {
        return symbol_type (token::GENERAL, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GENERAL (const string& v, const location_type& l)
      {
        return symbol_type (token::GENERAL, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GENERATED (location_type l)
      {
        return symbol_type (token::GENERATED, std::move (l));
      }
#else
      static
      symbol_type
      make_GENERATED (const location_type& l)
      {
        return symbol_type (token::GENERATED, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GROUP_REPLICATION (string v, location_type l)
      {
        return symbol_type (token::GROUP_REPLICATION, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GROUP_REPLICATION (const string& v, const location_type& l)
      {
        return symbol_type (token::GROUP_REPLICATION, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GEOMETRYCOLLECTION_SYM (string v, location_type l)
      {
        return symbol_type (token::GEOMETRYCOLLECTION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GEOMETRYCOLLECTION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::GEOMETRYCOLLECTION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GEOMETRY_SYM (string v, location_type l)
      {
        return symbol_type (token::GEOMETRY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GEOMETRY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::GEOMETRY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GET_FORMAT (string v, location_type l)
      {
        return symbol_type (token::GET_FORMAT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GET_FORMAT (const string& v, const location_type& l)
      {
        return symbol_type (token::GET_FORMAT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GET_SYM (location_type l)
      {
        return symbol_type (token::GET_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GET_SYM (const location_type& l)
      {
        return symbol_type (token::GET_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GLOBAL_SYM (string v, location_type l)
      {
        return symbol_type (token::GLOBAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GLOBAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::GLOBAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRANT (location_type l)
      {
        return symbol_type (token::GRANT, std::move (l));
      }
#else
      static
      symbol_type
      make_GRANT (const location_type& l)
      {
        return symbol_type (token::GRANT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRANTS (string v, location_type l)
      {
        return symbol_type (token::GRANTS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GRANTS (const string& v, const location_type& l)
      {
        return symbol_type (token::GRANTS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GROUP_SYM (location_type l)
      {
        return symbol_type (token::GROUP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GROUP_SYM (const location_type& l)
      {
        return symbol_type (token::GROUP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GROUP_CONCAT_SYM (location_type l)
      {
        return symbol_type (token::GROUP_CONCAT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GROUP_CONCAT_SYM (const location_type& l)
      {
        return symbol_type (token::GROUP_CONCAT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GT_SYM (location_type l)
      {
        return symbol_type (token::GT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GT_SYM (const location_type& l)
      {
        return symbol_type (token::GT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HANDLER_SYM (string v, location_type l)
      {
        return symbol_type (token::HANDLER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HANDLER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HANDLER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HASH_SYM (string v, location_type l)
      {
        return symbol_type (token::HASH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HASH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HASH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HAVING (location_type l)
      {
        return symbol_type (token::HAVING, std::move (l));
      }
#else
      static
      symbol_type
      make_HAVING (const location_type& l)
      {
        return symbol_type (token::HAVING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HELP_SYM (string v, location_type l)
      {
        return symbol_type (token::HELP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HELP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HELP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HEX_NUM (string v, location_type l)
      {
        return symbol_type (token::HEX_NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HEX_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::HEX_NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HIGH_PRIORITY (location_type l)
      {
        return symbol_type (token::HIGH_PRIORITY, std::move (l));
      }
#else
      static
      symbol_type
      make_HIGH_PRIORITY (const location_type& l)
      {
        return symbol_type (token::HIGH_PRIORITY, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOST_SYM (string v, location_type l)
      {
        return symbol_type (token::HOST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HOST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HOST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOSTS_SYM (string v, location_type l)
      {
        return symbol_type (token::HOSTS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HOSTS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HOSTS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOUR_MICROSECOND_SYM (location_type l)
      {
        return symbol_type (token::HOUR_MICROSECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_HOUR_MICROSECOND_SYM (const location_type& l)
      {
        return symbol_type (token::HOUR_MICROSECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOUR_MINUTE_SYM (location_type l)
      {
        return symbol_type (token::HOUR_MINUTE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_HOUR_MINUTE_SYM (const location_type& l)
      {
        return symbol_type (token::HOUR_MINUTE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOUR_SECOND_SYM (location_type l)
      {
        return symbol_type (token::HOUR_SECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_HOUR_SECOND_SYM (const location_type& l)
      {
        return symbol_type (token::HOUR_SECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HOUR_SYM (string v, location_type l)
      {
        return symbol_type (token::HOUR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HOUR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HOUR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IDENT (string v, location_type l)
      {
        return symbol_type (token::IDENT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IDENT (const string& v, const location_type& l)
      {
        return symbol_type (token::IDENT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IDENTIFIED_SYM (string v, location_type l)
      {
        return symbol_type (token::IDENTIFIED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IDENTIFIED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::IDENTIFIED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IDENT_QUOTED (location_type l)
      {
        return symbol_type (token::IDENT_QUOTED, std::move (l));
      }
#else
      static
      symbol_type
      make_IDENT_QUOTED (const location_type& l)
      {
        return symbol_type (token::IDENT_QUOTED, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IF (location_type l)
      {
        return symbol_type (token::IF, std::move (l));
      }
#else
      static
      symbol_type
      make_IF (const location_type& l)
      {
        return symbol_type (token::IF, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IGNORE_SYM (location_type l)
      {
        return symbol_type (token::IGNORE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_IGNORE_SYM (const location_type& l)
      {
        return symbol_type (token::IGNORE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IGNORE_SERVER_IDS_SYM (string v, location_type l)
      {
        return symbol_type (token::IGNORE_SERVER_IDS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IGNORE_SERVER_IDS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::IGNORE_SERVER_IDS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IMPORT (string v, location_type l)
      {
        return symbol_type (token::IMPORT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IMPORT (const string& v, const location_type& l)
      {
        return symbol_type (token::IMPORT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INDEXES (string v, location_type l)
      {
        return symbol_type (token::INDEXES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INDEXES (const string& v, const location_type& l)
      {
        return symbol_type (token::INDEXES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INDEX_SYM (location_type l)
      {
        return symbol_type (token::INDEX_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INDEX_SYM (const location_type& l)
      {
        return symbol_type (token::INDEX_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INFILE (location_type l)
      {
        return symbol_type (token::INFILE, std::move (l));
      }
#else
      static
      symbol_type
      make_INFILE (const location_type& l)
      {
        return symbol_type (token::INFILE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INITIAL_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::INITIAL_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INITIAL_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INITIAL_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INNER_SYM (location_type l)
      {
        return symbol_type (token::INNER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INNER_SYM (const location_type& l)
      {
        return symbol_type (token::INNER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INOUT_SYM (location_type l)
      {
        return symbol_type (token::INOUT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INOUT_SYM (const location_type& l)
      {
        return symbol_type (token::INOUT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INSENSITIVE_SYM (location_type l)
      {
        return symbol_type (token::INSENSITIVE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INSENSITIVE_SYM (const location_type& l)
      {
        return symbol_type (token::INSENSITIVE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INSERT_SYM (location_type l)
      {
        return symbol_type (token::INSERT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INSERT_SYM (const location_type& l)
      {
        return symbol_type (token::INSERT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INSERT_METHOD (string v, location_type l)
      {
        return symbol_type (token::INSERT_METHOD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INSERT_METHOD (const string& v, const location_type& l)
      {
        return symbol_type (token::INSERT_METHOD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INSTANCE_SYM (string v, location_type l)
      {
        return symbol_type (token::INSTANCE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INSTANCE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INSTANCE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INSTALL_SYM (string v, location_type l)
      {
        return symbol_type (token::INSTALL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INSTALL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INSTALL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INTERVAL_SYM (string v, location_type l)
      {
        return symbol_type (token::INTERVAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INTERVAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INTERVAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INTO (location_type l)
      {
        return symbol_type (token::INTO, std::move (l));
      }
#else
      static
      symbol_type
      make_INTO (const location_type& l)
      {
        return symbol_type (token::INTO, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INT_SYM (location_type l)
      {
        return symbol_type (token::INT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INT_SYM (const location_type& l)
      {
        return symbol_type (token::INT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INTEGER_SYM (location_type l)
      {
        return symbol_type (token::INTEGER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_INTEGER_SYM (const location_type& l)
      {
        return symbol_type (token::INTEGER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INVOKER_SYM (string v, location_type l)
      {
        return symbol_type (token::INVOKER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INVOKER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INVOKER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IN_SYM (location_type l)
      {
        return symbol_type (token::IN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_IN_SYM (const location_type& l)
      {
        return symbol_type (token::IN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IO_AFTER_GTIDS (location_type l)
      {
        return symbol_type (token::IO_AFTER_GTIDS, std::move (l));
      }
#else
      static
      symbol_type
      make_IO_AFTER_GTIDS (const location_type& l)
      {
        return symbol_type (token::IO_AFTER_GTIDS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IO_BEFORE_GTIDS (location_type l)
      {
        return symbol_type (token::IO_BEFORE_GTIDS, std::move (l));
      }
#else
      static
      symbol_type
      make_IO_BEFORE_GTIDS (const location_type& l)
      {
        return symbol_type (token::IO_BEFORE_GTIDS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IO_SYM (string v, location_type l)
      {
        return symbol_type (token::IO_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IO_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::IO_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IPC_SYM (string v, location_type l)
      {
        return symbol_type (token::IPC_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_IPC_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::IPC_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_IS (location_type l)
      {
        return symbol_type (token::IS, std::move (l));
      }
#else
      static
      symbol_type
      make_IS (const location_type& l)
      {
        return symbol_type (token::IS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ISOLATION (string v, location_type l)
      {
        return symbol_type (token::ISOLATION, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ISOLATION (const string& v, const location_type& l)
      {
        return symbol_type (token::ISOLATION, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ISSUER_SYM (string v, location_type l)
      {
        return symbol_type (token::ISSUER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ISSUER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ISSUER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ITERATE_SYM (location_type l)
      {
        return symbol_type (token::ITERATE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ITERATE_SYM (const location_type& l)
      {
        return symbol_type (token::ITERATE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JOIN_SYM (location_type l)
      {
        return symbol_type (token::JOIN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_JOIN_SYM (const location_type& l)
      {
        return symbol_type (token::JOIN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_SEPARATOR_SYM (location_type l)
      {
        return symbol_type (token::JSON_SEPARATOR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_SEPARATOR_SYM (const location_type& l)
      {
        return symbol_type (token::JSON_SEPARATOR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_SYM (string v, location_type l)
      {
        return symbol_type (token::JSON_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::JSON_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_KEYS (location_type l)
      {
        return symbol_type (token::KEYS, std::move (l));
      }
#else
      static
      symbol_type
      make_KEYS (const location_type& l)
      {
        return symbol_type (token::KEYS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_KEY_BLOCK_SIZE (string v, location_type l)
      {
        return symbol_type (token::KEY_BLOCK_SIZE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_KEY_BLOCK_SIZE (const string& v, const location_type& l)
      {
        return symbol_type (token::KEY_BLOCK_SIZE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_KEY_SYM (location_type l)
      {
        return symbol_type (token::KEY_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_KEY_SYM (const location_type& l)
      {
        return symbol_type (token::KEY_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_KILL_SYM (location_type l)
      {
        return symbol_type (token::KILL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_KILL_SYM (const location_type& l)
      {
        return symbol_type (token::KILL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LANGUAGE_SYM (string v, location_type l)
      {
        return symbol_type (token::LANGUAGE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LANGUAGE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LANGUAGE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LAST_SYM (string v, location_type l)
      {
        return symbol_type (token::LAST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LAST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LAST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LE (location_type l)
      {
        return symbol_type (token::LE, std::move (l));
      }
#else
      static
      symbol_type
      make_LE (const location_type& l)
      {
        return symbol_type (token::LE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEADING (location_type l)
      {
        return symbol_type (token::LEADING, std::move (l));
      }
#else
      static
      symbol_type
      make_LEADING (const location_type& l)
      {
        return symbol_type (token::LEADING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEAVES (string v, location_type l)
      {
        return symbol_type (token::LEAVES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LEAVES (const string& v, const location_type& l)
      {
        return symbol_type (token::LEAVES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEAVE_SYM (location_type l)
      {
        return symbol_type (token::LEAVE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LEAVE_SYM (const location_type& l)
      {
        return symbol_type (token::LEAVE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEFT (location_type l)
      {
        return symbol_type (token::LEFT, std::move (l));
      }
#else
      static
      symbol_type
      make_LEFT (const location_type& l)
      {
        return symbol_type (token::LEFT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LESS_SYM (string v, location_type l)
      {
        return symbol_type (token::LESS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LESS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LESS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEVEL_SYM (string v, location_type l)
      {
        return symbol_type (token::LEVEL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LEVEL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LEVEL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEX_HOSTNAME (string v, location_type l)
      {
        return symbol_type (token::LEX_HOSTNAME, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LEX_HOSTNAME (const string& v, const location_type& l)
      {
        return symbol_type (token::LEX_HOSTNAME, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LIKE (location_type l)
      {
        return symbol_type (token::LIKE, std::move (l));
      }
#else
      static
      symbol_type
      make_LIKE (const location_type& l)
      {
        return symbol_type (token::LIKE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LIMIT (location_type l)
      {
        return symbol_type (token::LIMIT, std::move (l));
      }
#else
      static
      symbol_type
      make_LIMIT (const location_type& l)
      {
        return symbol_type (token::LIMIT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LINEAR_SYM (location_type l)
      {
        return symbol_type (token::LINEAR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LINEAR_SYM (const location_type& l)
      {
        return symbol_type (token::LINEAR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LINES (location_type l)
      {
        return symbol_type (token::LINES, std::move (l));
      }
#else
      static
      symbol_type
      make_LINES (const location_type& l)
      {
        return symbol_type (token::LINES, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LINESTRING_SYM (string v, location_type l)
      {
        return symbol_type (token::LINESTRING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LINESTRING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LINESTRING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LIST_SYM (string v, location_type l)
      {
        return symbol_type (token::LIST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LIST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LIST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOAD (location_type l)
      {
        return symbol_type (token::LOAD, std::move (l));
      }
#else
      static
      symbol_type
      make_LOAD (const location_type& l)
      {
        return symbol_type (token::LOAD, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOCAL_SYM (string v, location_type l)
      {
        return symbol_type (token::LOCAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LOCAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LOCAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_538 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_538, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_538 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_538, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOCKS_SYM (string v, location_type l)
      {
        return symbol_type (token::LOCKS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LOCKS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LOCKS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOCK_SYM (location_type l)
      {
        return symbol_type (token::LOCK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LOCK_SYM (const location_type& l)
      {
        return symbol_type (token::LOCK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOGFILE_SYM (string v, location_type l)
      {
        return symbol_type (token::LOGFILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LOGFILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LOGFILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOGS_SYM (string v, location_type l)
      {
        return symbol_type (token::LOGS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LOGS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LOGS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LONGBLOB_SYM (location_type l)
      {
        return symbol_type (token::LONGBLOB_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LONGBLOB_SYM (const location_type& l)
      {
        return symbol_type (token::LONGBLOB_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LONGTEXT_SYM (location_type l)
      {
        return symbol_type (token::LONGTEXT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LONGTEXT_SYM (const location_type& l)
      {
        return symbol_type (token::LONGTEXT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LONG_NUM (string v, location_type l)
      {
        return symbol_type (token::LONG_NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LONG_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::LONG_NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LONG_SYM (location_type l)
      {
        return symbol_type (token::LONG_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LONG_SYM (const location_type& l)
      {
        return symbol_type (token::LONG_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOOP_SYM (location_type l)
      {
        return symbol_type (token::LOOP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LOOP_SYM (const location_type& l)
      {
        return symbol_type (token::LOOP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOW_PRIORITY (location_type l)
      {
        return symbol_type (token::LOW_PRIORITY, std::move (l));
      }
#else
      static
      symbol_type
      make_LOW_PRIORITY (const location_type& l)
      {
        return symbol_type (token::LOW_PRIORITY, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LT (location_type l)
      {
        return symbol_type (token::LT, std::move (l));
      }
#else
      static
      symbol_type
      make_LT (const location_type& l)
      {
        return symbol_type (token::LT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_AUTO_POSITION_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_AUTO_POSITION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_AUTO_POSITION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_AUTO_POSITION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_BIND_SYM (location_type l)
      {
        return symbol_type (token::MASTER_BIND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_BIND_SYM (const location_type& l)
      {
        return symbol_type (token::MASTER_BIND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_CONNECT_RETRY_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_CONNECT_RETRY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_CONNECT_RETRY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_CONNECT_RETRY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_DELAY_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_DELAY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_DELAY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_DELAY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_HOST_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_HOST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_HOST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_HOST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_LOG_FILE_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_LOG_FILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_LOG_FILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_LOG_FILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_LOG_POS_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_LOG_POS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_LOG_POS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_LOG_POS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_PASSWORD_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_PASSWORD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_PASSWORD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_PASSWORD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_PORT_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_PORT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_PORT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_PORT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_RETRY_COUNT_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_RETRY_COUNT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_RETRY_COUNT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_RETRY_COUNT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SERVER_ID_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SERVER_ID_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SERVER_ID_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SERVER_ID_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CAPATH_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CAPATH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CAPATH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CAPATH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_TLS_VERSION_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_TLS_VERSION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_TLS_VERSION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_TLS_VERSION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CA_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CA_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CA_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CA_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CERT_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CERT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CERT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CERT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CIPHER_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CIPHER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CIPHER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CIPHER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CRL_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CRL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CRL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CRL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_CRLPATH_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_CRLPATH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_CRLPATH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_CRLPATH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_KEY_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_KEY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_KEY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_KEY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SSL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SSL_VERIFY_SERVER_CERT_SYM (location_type l)
      {
        return symbol_type (token::MASTER_SSL_VERIFY_SERVER_CERT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SSL_VERIFY_SERVER_CERT_SYM (const location_type& l)
      {
        return symbol_type (token::MASTER_SSL_VERIFY_SERVER_CERT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_USER_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_USER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_USER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_USER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_HEARTBEAT_PERIOD_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_HEARTBEAT_PERIOD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_HEARTBEAT_PERIOD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_HEARTBEAT_PERIOD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MATCH (location_type l)
      {
        return symbol_type (token::MATCH, std::move (l));
      }
#else
      static
      symbol_type
      make_MATCH (const location_type& l)
      {
        return symbol_type (token::MATCH, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_CONNECTIONS_PER_HOUR (string v, location_type l)
      {
        return symbol_type (token::MAX_CONNECTIONS_PER_HOUR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_CONNECTIONS_PER_HOUR (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_CONNECTIONS_PER_HOUR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_QUERIES_PER_HOUR (string v, location_type l)
      {
        return symbol_type (token::MAX_QUERIES_PER_HOUR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_QUERIES_PER_HOUR (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_QUERIES_PER_HOUR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_ROWS (string v, location_type l)
      {
        return symbol_type (token::MAX_ROWS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_ROWS (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_ROWS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::MAX_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_SYM (location_type l)
      {
        return symbol_type (token::MAX_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_SYM (const location_type& l)
      {
        return symbol_type (token::MAX_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_UPDATES_PER_HOUR (string v, location_type l)
      {
        return symbol_type (token::MAX_UPDATES_PER_HOUR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_UPDATES_PER_HOUR (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_UPDATES_PER_HOUR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_USER_CONNECTIONS_SYM (string v, location_type l)
      {
        return symbol_type (token::MAX_USER_CONNECTIONS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_USER_CONNECTIONS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MAX_USER_CONNECTIONS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MAX_VALUE_SYM (location_type l)
      {
        return symbol_type (token::MAX_VALUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MAX_VALUE_SYM (const location_type& l)
      {
        return symbol_type (token::MAX_VALUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MEDIUMBLOB_SYM (location_type l)
      {
        return symbol_type (token::MEDIUMBLOB_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MEDIUMBLOB_SYM (const location_type& l)
      {
        return symbol_type (token::MEDIUMBLOB_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MEDIUMINT_SYM (location_type l)
      {
        return symbol_type (token::MEDIUMINT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MEDIUMINT_SYM (const location_type& l)
      {
        return symbol_type (token::MEDIUMINT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MEDIUMTEXT_SYM (location_type l)
      {
        return symbol_type (token::MEDIUMTEXT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MEDIUMTEXT_SYM (const location_type& l)
      {
        return symbol_type (token::MEDIUMTEXT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MEDIUM_SYM (string v, location_type l)
      {
        return symbol_type (token::MEDIUM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MEDIUM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MEDIUM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MEMORY_SYM (string v, location_type l)
      {
        return symbol_type (token::MEMORY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MEMORY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MEMORY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MERGE_SYM (string v, location_type l)
      {
        return symbol_type (token::MERGE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MERGE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MERGE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MESSAGE_TEXT_SYM (string v, location_type l)
      {
        return symbol_type (token::MESSAGE_TEXT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MESSAGE_TEXT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MESSAGE_TEXT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MICROSECOND_SYM (string v, location_type l)
      {
        return symbol_type (token::MICROSECOND_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MICROSECOND_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MICROSECOND_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MIGRATE_SYM (string v, location_type l)
      {
        return symbol_type (token::MIGRATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MIGRATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MIGRATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MINUTE_MICROSECOND_SYM (location_type l)
      {
        return symbol_type (token::MINUTE_MICROSECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MINUTE_MICROSECOND_SYM (const location_type& l)
      {
        return symbol_type (token::MINUTE_MICROSECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MINUTE_SECOND_SYM (location_type l)
      {
        return symbol_type (token::MINUTE_SECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MINUTE_SECOND_SYM (const location_type& l)
      {
        return symbol_type (token::MINUTE_SECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MINUTE_SYM (string v, location_type l)
      {
        return symbol_type (token::MINUTE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MINUTE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MINUTE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MIN_ROWS (string v, location_type l)
      {
        return symbol_type (token::MIN_ROWS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MIN_ROWS (const string& v, const location_type& l)
      {
        return symbol_type (token::MIN_ROWS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MIN_SYM (location_type l)
      {
        return symbol_type (token::MIN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MIN_SYM (const location_type& l)
      {
        return symbol_type (token::MIN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MODE_SYM (string v, location_type l)
      {
        return symbol_type (token::MODE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MODE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MODE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MODIFIES_SYM (location_type l)
      {
        return symbol_type (token::MODIFIES_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MODIFIES_SYM (const location_type& l)
      {
        return symbol_type (token::MODIFIES_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MODIFY_SYM (string v, location_type l)
      {
        return symbol_type (token::MODIFY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MODIFY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MODIFY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MOD_SYM (location_type l)
      {
        return symbol_type (token::MOD_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MOD_SYM (const location_type& l)
      {
        return symbol_type (token::MOD_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MONTH_SYM (string v, location_type l)
      {
        return symbol_type (token::MONTH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MONTH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MONTH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MULTILINESTRING_SYM (string v, location_type l)
      {
        return symbol_type (token::MULTILINESTRING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MULTILINESTRING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MULTILINESTRING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MULTIPOINT_SYM (string v, location_type l)
      {
        return symbol_type (token::MULTIPOINT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MULTIPOINT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MULTIPOINT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MULTIPOLYGON_SYM (string v, location_type l)
      {
        return symbol_type (token::MULTIPOLYGON_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MULTIPOLYGON_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MULTIPOLYGON_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MUTEX_SYM (string v, location_type l)
      {
        return symbol_type (token::MUTEX_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MUTEX_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MUTEX_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MYSQL_ERRNO_SYM (string v, location_type l)
      {
        return symbol_type (token::MYSQL_ERRNO_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MYSQL_ERRNO_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MYSQL_ERRNO_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NAMES_SYM (string v, location_type l)
      {
        return symbol_type (token::NAMES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NAMES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NAMES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NATIONAL_SYM (string v, location_type l)
      {
        return symbol_type (token::NATIONAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NATIONAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NATIONAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NATURAL (location_type l)
      {
        return symbol_type (token::NATURAL, std::move (l));
      }
#else
      static
      symbol_type
      make_NATURAL (const location_type& l)
      {
        return symbol_type (token::NATURAL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NCHAR_STRING (location_type l)
      {
        return symbol_type (token::NCHAR_STRING, std::move (l));
      }
#else
      static
      symbol_type
      make_NCHAR_STRING (const location_type& l)
      {
        return symbol_type (token::NCHAR_STRING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NCHAR_SYM (string v, location_type l)
      {
        return symbol_type (token::NCHAR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NCHAR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NCHAR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NDBCLUSTER_SYM (string v, location_type l)
      {
        return symbol_type (token::NDBCLUSTER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NDBCLUSTER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NDBCLUSTER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NE (location_type l)
      {
        return symbol_type (token::NE, std::move (l));
      }
#else
      static
      symbol_type
      make_NE (const location_type& l)
      {
        return symbol_type (token::NE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEG (location_type l)
      {
        return symbol_type (token::NEG, std::move (l));
      }
#else
      static
      symbol_type
      make_NEG (const location_type& l)
      {
        return symbol_type (token::NEG, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEVER_SYM (string v, location_type l)
      {
        return symbol_type (token::NEVER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NEVER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NEVER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEW_SYM (string v, location_type l)
      {
        return symbol_type (token::NEW_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NEW_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NEW_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NEXT_SYM (string v, location_type l)
      {
        return symbol_type (token::NEXT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NEXT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NEXT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NODEGROUP_SYM (string v, location_type l)
      {
        return symbol_type (token::NODEGROUP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NODEGROUP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NODEGROUP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NONE_SYM (string v, location_type l)
      {
        return symbol_type (token::NONE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NONE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NONE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NOT2_SYM (location_type l)
      {
        return symbol_type (token::NOT2_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NOT2_SYM (const location_type& l)
      {
        return symbol_type (token::NOT2_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NOT_SYM (location_type l)
      {
        return symbol_type (token::NOT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NOT_SYM (const location_type& l)
      {
        return symbol_type (token::NOT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NOW_SYM (location_type l)
      {
        return symbol_type (token::NOW_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NOW_SYM (const location_type& l)
      {
        return symbol_type (token::NOW_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NO_SYM (string v, location_type l)
      {
        return symbol_type (token::NO_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NO_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NO_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NO_WAIT_SYM (string v, location_type l)
      {
        return symbol_type (token::NO_WAIT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NO_WAIT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NO_WAIT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NO_WRITE_TO_BINLOG (location_type l)
      {
        return symbol_type (token::NO_WRITE_TO_BINLOG, std::move (l));
      }
#else
      static
      symbol_type
      make_NO_WRITE_TO_BINLOG (const location_type& l)
      {
        return symbol_type (token::NO_WRITE_TO_BINLOG, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NULL_SYM (location_type l)
      {
        return symbol_type (token::NULL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NULL_SYM (const location_type& l)
      {
        return symbol_type (token::NULL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NUM (string v, location_type l)
      {
        return symbol_type (token::NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NUMBER_SYM (string v, location_type l)
      {
        return symbol_type (token::NUMBER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NUMBER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NUMBER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NUMERIC_SYM (location_type l)
      {
        return symbol_type (token::NUMERIC_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NUMERIC_SYM (const location_type& l)
      {
        return symbol_type (token::NUMERIC_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NVARCHAR_SYM (string v, location_type l)
      {
        return symbol_type (token::NVARCHAR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NVARCHAR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NVARCHAR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OFFSET_SYM (string v, location_type l)
      {
        return symbol_type (token::OFFSET_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OFFSET_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OFFSET_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ON_SYM (location_type l)
      {
        return symbol_type (token::ON_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ON_SYM (const location_type& l)
      {
        return symbol_type (token::ON_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ONE_SYM (string v, location_type l)
      {
        return symbol_type (token::ONE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ONE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ONE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ONLY_SYM (string v, location_type l)
      {
        return symbol_type (token::ONLY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ONLY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ONLY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPEN_SYM (string v, location_type l)
      {
        return symbol_type (token::OPEN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OPEN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OPEN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTIMIZE (location_type l)
      {
        return symbol_type (token::OPTIMIZE, std::move (l));
      }
#else
      static
      symbol_type
      make_OPTIMIZE (const location_type& l)
      {
        return symbol_type (token::OPTIMIZE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTIMIZER_COSTS_SYM (location_type l)
      {
        return symbol_type (token::OPTIMIZER_COSTS_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OPTIMIZER_COSTS_SYM (const location_type& l)
      {
        return symbol_type (token::OPTIMIZER_COSTS_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTIONS_SYM (string v, location_type l)
      {
        return symbol_type (token::OPTIONS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OPTIONS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OPTIONS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTION (location_type l)
      {
        return symbol_type (token::OPTION, std::move (l));
      }
#else
      static
      symbol_type
      make_OPTION (const location_type& l)
      {
        return symbol_type (token::OPTION, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTIONALLY (location_type l)
      {
        return symbol_type (token::OPTIONALLY, std::move (l));
      }
#else
      static
      symbol_type
      make_OPTIONALLY (const location_type& l)
      {
        return symbol_type (token::OPTIONALLY, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OR2_SYM (location_type l)
      {
        return symbol_type (token::OR2_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OR2_SYM (const location_type& l)
      {
        return symbol_type (token::OR2_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ORDER_SYM (location_type l)
      {
        return symbol_type (token::ORDER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ORDER_SYM (const location_type& l)
      {
        return symbol_type (token::ORDER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OR_OR_SYM (location_type l)
      {
        return symbol_type (token::OR_OR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OR_OR_SYM (const location_type& l)
      {
        return symbol_type (token::OR_OR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OR_SYM (location_type l)
      {
        return symbol_type (token::OR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OR_SYM (const location_type& l)
      {
        return symbol_type (token::OR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OUTER (location_type l)
      {
        return symbol_type (token::OUTER, std::move (l));
      }
#else
      static
      symbol_type
      make_OUTER (const location_type& l)
      {
        return symbol_type (token::OUTER, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OUTFILE (location_type l)
      {
        return symbol_type (token::OUTFILE, std::move (l));
      }
#else
      static
      symbol_type
      make_OUTFILE (const location_type& l)
      {
        return symbol_type (token::OUTFILE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OUT_SYM (location_type l)
      {
        return symbol_type (token::OUT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OUT_SYM (const location_type& l)
      {
        return symbol_type (token::OUT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OWNER_SYM (string v, location_type l)
      {
        return symbol_type (token::OWNER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OWNER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OWNER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PACK_KEYS_SYM (string v, location_type l)
      {
        return symbol_type (token::PACK_KEYS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PACK_KEYS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PACK_KEYS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PAGE_SYM (string v, location_type l)
      {
        return symbol_type (token::PAGE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PAGE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PAGE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARAM_MARKER (location_type l)
      {
        return symbol_type (token::PARAM_MARKER, std::move (l));
      }
#else
      static
      symbol_type
      make_PARAM_MARKER (const location_type& l)
      {
        return symbol_type (token::PARAM_MARKER, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARSER_SYM (string v, location_type l)
      {
        return symbol_type (token::PARSER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PARSER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PARSER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_654 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_654, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_654 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_654, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARTIAL (string v, location_type l)
      {
        return symbol_type (token::PARTIAL, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PARTIAL (const string& v, const location_type& l)
      {
        return symbol_type (token::PARTIAL, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARTITION_SYM (location_type l)
      {
        return symbol_type (token::PARTITION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_PARTITION_SYM (const location_type& l)
      {
        return symbol_type (token::PARTITION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARTITIONS_SYM (string v, location_type l)
      {
        return symbol_type (token::PARTITIONS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PARTITIONS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PARTITIONS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PARTITIONING_SYM (string v, location_type l)
      {
        return symbol_type (token::PARTITIONING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PARTITIONING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PARTITIONING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PASSWORD (string v, location_type l)
      {
        return symbol_type (token::PASSWORD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PASSWORD (const string& v, const location_type& l)
      {
        return symbol_type (token::PASSWORD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PHASE_SYM (string v, location_type l)
      {
        return symbol_type (token::PHASE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PHASE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PHASE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PLUGIN_DIR_SYM (string v, location_type l)
      {
        return symbol_type (token::PLUGIN_DIR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PLUGIN_DIR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PLUGIN_DIR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PLUGIN_SYM (string v, location_type l)
      {
        return symbol_type (token::PLUGIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PLUGIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PLUGIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PLUGINS_SYM (string v, location_type l)
      {
        return symbol_type (token::PLUGINS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PLUGINS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PLUGINS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POINT_SYM (string v, location_type l)
      {
        return symbol_type (token::POINT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_POINT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::POINT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POLYGON_SYM (string v, location_type l)
      {
        return symbol_type (token::POLYGON_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_POLYGON_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::POLYGON_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PORT_SYM (string v, location_type l)
      {
        return symbol_type (token::PORT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PORT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PORT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_POSITION_SYM (location_type l)
      {
        return symbol_type (token::POSITION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_POSITION_SYM (const location_type& l)
      {
        return symbol_type (token::POSITION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRECEDES_SYM (string v, location_type l)
      {
        return symbol_type (token::PRECEDES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PRECEDES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PRECEDES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRECISION (location_type l)
      {
        return symbol_type (token::PRECISION, std::move (l));
      }
#else
      static
      symbol_type
      make_PRECISION (const location_type& l)
      {
        return symbol_type (token::PRECISION, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PREPARE_SYM (string v, location_type l)
      {
        return symbol_type (token::PREPARE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PREPARE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PREPARE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRESERVE_SYM (string v, location_type l)
      {
        return symbol_type (token::PRESERVE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PRESERVE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PRESERVE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PREV_SYM (string v, location_type l)
      {
        return symbol_type (token::PREV_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PREV_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PREV_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRIMARY_SYM (location_type l)
      {
        return symbol_type (token::PRIMARY_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_PRIMARY_SYM (const location_type& l)
      {
        return symbol_type (token::PRIMARY_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRIVILEGES (string v, location_type l)
      {
        return symbol_type (token::PRIVILEGES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PRIVILEGES (const string& v, const location_type& l)
      {
        return symbol_type (token::PRIVILEGES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROCEDURE_SYM (location_type l)
      {
        return symbol_type (token::PROCEDURE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_PROCEDURE_SYM (const location_type& l)
      {
        return symbol_type (token::PROCEDURE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROCESS (string v, location_type l)
      {
        return symbol_type (token::PROCESS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROCESS (const string& v, const location_type& l)
      {
        return symbol_type (token::PROCESS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROCESSLIST_SYM (string v, location_type l)
      {
        return symbol_type (token::PROCESSLIST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROCESSLIST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PROCESSLIST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROFILE_SYM (string v, location_type l)
      {
        return symbol_type (token::PROFILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROFILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PROFILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROFILES_SYM (string v, location_type l)
      {
        return symbol_type (token::PROFILES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROFILES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PROFILES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PROXY_SYM (string v, location_type l)
      {
        return symbol_type (token::PROXY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PROXY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PROXY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PURGE (location_type l)
      {
        return symbol_type (token::PURGE, std::move (l));
      }
#else
      static
      symbol_type
      make_PURGE (const location_type& l)
      {
        return symbol_type (token::PURGE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_QUARTER_SYM (string v, location_type l)
      {
        return symbol_type (token::QUARTER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_QUARTER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::QUARTER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_QUERY_SYM (string v, location_type l)
      {
        return symbol_type (token::QUERY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_QUERY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::QUERY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_QUICK (string v, location_type l)
      {
        return symbol_type (token::QUICK, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_QUICK (const string& v, const location_type& l)
      {
        return symbol_type (token::QUICK, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RANGE_SYM (location_type l)
      {
        return symbol_type (token::RANGE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RANGE_SYM (const location_type& l)
      {
        return symbol_type (token::RANGE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_READS_SYM (location_type l)
      {
        return symbol_type (token::READS_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_READS_SYM (const location_type& l)
      {
        return symbol_type (token::READS_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_READ_ONLY_SYM (string v, location_type l)
      {
        return symbol_type (token::READ_ONLY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_READ_ONLY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::READ_ONLY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_READ_SYM (location_type l)
      {
        return symbol_type (token::READ_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_READ_SYM (const location_type& l)
      {
        return symbol_type (token::READ_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_READ_WRITE_SYM (location_type l)
      {
        return symbol_type (token::READ_WRITE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_READ_WRITE_SYM (const location_type& l)
      {
        return symbol_type (token::READ_WRITE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REAL_SYM (location_type l)
      {
        return symbol_type (token::REAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_REAL_SYM (const location_type& l)
      {
        return symbol_type (token::REAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REBUILD_SYM (string v, location_type l)
      {
        return symbol_type (token::REBUILD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REBUILD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REBUILD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RECOVER_SYM (string v, location_type l)
      {
        return symbol_type (token::RECOVER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RECOVER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RECOVER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_693 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_693, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_693 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_693, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REDO_BUFFER_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::REDO_BUFFER_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REDO_BUFFER_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REDO_BUFFER_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REDUNDANT_SYM (string v, location_type l)
      {
        return symbol_type (token::REDUNDANT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REDUNDANT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REDUNDANT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REFERENCES (location_type l)
      {
        return symbol_type (token::REFERENCES, std::move (l));
      }
#else
      static
      symbol_type
      make_REFERENCES (const location_type& l)
      {
        return symbol_type (token::REFERENCES, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REGEXP (location_type l)
      {
        return symbol_type (token::REGEXP, std::move (l));
      }
#else
      static
      symbol_type
      make_REGEXP (const location_type& l)
      {
        return symbol_type (token::REGEXP, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELAY (string v, location_type l)
      {
        return symbol_type (token::RELAY, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELAY (const string& v, const location_type& l)
      {
        return symbol_type (token::RELAY, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELAYLOG_SYM (string v, location_type l)
      {
        return symbol_type (token::RELAYLOG_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELAYLOG_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RELAYLOG_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELAY_LOG_FILE_SYM (string v, location_type l)
      {
        return symbol_type (token::RELAY_LOG_FILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELAY_LOG_FILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RELAY_LOG_FILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELAY_LOG_POS_SYM (string v, location_type l)
      {
        return symbol_type (token::RELAY_LOG_POS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELAY_LOG_POS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RELAY_LOG_POS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELAY_THREAD (string v, location_type l)
      {
        return symbol_type (token::RELAY_THREAD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELAY_THREAD (const string& v, const location_type& l)
      {
        return symbol_type (token::RELAY_THREAD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELEASE_SYM (location_type l)
      {
        return symbol_type (token::RELEASE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RELEASE_SYM (const location_type& l)
      {
        return symbol_type (token::RELEASE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RELOAD (string v, location_type l)
      {
        return symbol_type (token::RELOAD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RELOAD (const string& v, const location_type& l)
      {
        return symbol_type (token::RELOAD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REMOVE_SYM (string v, location_type l)
      {
        return symbol_type (token::REMOVE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REMOVE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REMOVE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RENAME (location_type l)
      {
        return symbol_type (token::RENAME, std::move (l));
      }
#else
      static
      symbol_type
      make_RENAME (const location_type& l)
      {
        return symbol_type (token::RENAME, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REORGANIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::REORGANIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REORGANIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REORGANIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPAIR (string v, location_type l)
      {
        return symbol_type (token::REPAIR, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPAIR (const string& v, const location_type& l)
      {
        return symbol_type (token::REPAIR, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPEATABLE_SYM (string v, location_type l)
      {
        return symbol_type (token::REPEATABLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPEATABLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REPEATABLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPEAT_SYM (location_type l)
      {
        return symbol_type (token::REPEAT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_REPEAT_SYM (const location_type& l)
      {
        return symbol_type (token::REPEAT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLACE_SYM (location_type l)
      {
        return symbol_type (token::REPLACE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_REPLACE_SYM (const location_type& l)
      {
        return symbol_type (token::REPLACE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATION (string v, location_type l)
      {
        return symbol_type (token::REPLICATION, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATION (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATION, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_DO_DB (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_DO_DB, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_DO_DB (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_DO_DB, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_IGNORE_DB (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_IGNORE_DB, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_IGNORE_DB (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_IGNORE_DB, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_DO_TABLE (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_DO_TABLE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_DO_TABLE (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_DO_TABLE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_IGNORE_TABLE (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_IGNORE_TABLE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_IGNORE_TABLE (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_IGNORE_TABLE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_WILD_DO_TABLE (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_WILD_DO_TABLE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_WILD_DO_TABLE (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_WILD_DO_TABLE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_WILD_IGNORE_TABLE (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_WILD_IGNORE_TABLE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_WILD_IGNORE_TABLE (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_WILD_IGNORE_TABLE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REPLICATE_REWRITE_DB (string v, location_type l)
      {
        return symbol_type (token::REPLICATE_REWRITE_DB, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REPLICATE_REWRITE_DB (const string& v, const location_type& l)
      {
        return symbol_type (token::REPLICATE_REWRITE_DB, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REQUIRE_SYM (location_type l)
      {
        return symbol_type (token::REQUIRE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_REQUIRE_SYM (const location_type& l)
      {
        return symbol_type (token::REQUIRE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESET_SYM (string v, location_type l)
      {
        return symbol_type (token::RESET_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESET_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESET_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESIGNAL_SYM (location_type l)
      {
        return symbol_type (token::RESIGNAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RESIGNAL_SYM (const location_type& l)
      {
        return symbol_type (token::RESIGNAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESOURCES (string v, location_type l)
      {
        return symbol_type (token::RESOURCES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESOURCES (const string& v, const location_type& l)
      {
        return symbol_type (token::RESOURCES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESTORE_SYM (string v, location_type l)
      {
        return symbol_type (token::RESTORE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESTORE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESTORE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESTRICT (location_type l)
      {
        return symbol_type (token::RESTRICT, std::move (l));
      }
#else
      static
      symbol_type
      make_RESTRICT (const location_type& l)
      {
        return symbol_type (token::RESTRICT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESUME_SYM (string v, location_type l)
      {
        return symbol_type (token::RESUME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESUME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESUME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RETURNED_SQLSTATE_SYM (string v, location_type l)
      {
        return symbol_type (token::RETURNED_SQLSTATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RETURNED_SQLSTATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RETURNED_SQLSTATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RETURNS_SYM (string v, location_type l)
      {
        return symbol_type (token::RETURNS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RETURNS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RETURNS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RETURN_SYM (location_type l)
      {
        return symbol_type (token::RETURN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RETURN_SYM (const location_type& l)
      {
        return symbol_type (token::RETURN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REVERSE_SYM (string v, location_type l)
      {
        return symbol_type (token::REVERSE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REVERSE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REVERSE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REVOKE (location_type l)
      {
        return symbol_type (token::REVOKE, std::move (l));
      }
#else
      static
      symbol_type
      make_REVOKE (const location_type& l)
      {
        return symbol_type (token::REVOKE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RIGHT (location_type l)
      {
        return symbol_type (token::RIGHT, std::move (l));
      }
#else
      static
      symbol_type
      make_RIGHT (const location_type& l)
      {
        return symbol_type (token::RIGHT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROLLBACK_SYM (string v, location_type l)
      {
        return symbol_type (token::ROLLBACK_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROLLBACK_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROLLBACK_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROLLUP_SYM (string v, location_type l)
      {
        return symbol_type (token::ROLLUP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROLLUP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROLLUP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROTATE_SYM (string v, location_type l)
      {
        return symbol_type (token::ROTATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROTATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROTATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROUTINE_SYM (string v, location_type l)
      {
        return symbol_type (token::ROUTINE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROUTINE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROUTINE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROWS_SYM (location_type l)
      {
        return symbol_type (token::ROWS_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ROWS_SYM (const location_type& l)
      {
        return symbol_type (token::ROWS_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROW_FORMAT_SYM (string v, location_type l)
      {
        return symbol_type (token::ROW_FORMAT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROW_FORMAT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROW_FORMAT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROW_SYM (location_type l)
      {
        return symbol_type (token::ROW_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ROW_SYM (const location_type& l)
      {
        return symbol_type (token::ROW_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROW_COUNT_SYM (string v, location_type l)
      {
        return symbol_type (token::ROW_COUNT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROW_COUNT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROW_COUNT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RTREE_SYM (string v, location_type l)
      {
        return symbol_type (token::RTREE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RTREE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RTREE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SAVEPOINT_SYM (string v, location_type l)
      {
        return symbol_type (token::SAVEPOINT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SAVEPOINT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SAVEPOINT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SCHEDULE_SYM (string v, location_type l)
      {
        return symbol_type (token::SCHEDULE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SCHEDULE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SCHEDULE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SCHEMA_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::SCHEMA_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SCHEMA_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SCHEMA_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SCHEMA (location_type l)
      {
        return symbol_type (token::SCHEMA, std::move (l));
      }
#else
      static
      symbol_type
      make_SCHEMA (const location_type& l)
      {
        return symbol_type (token::SCHEMA, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECOND_MICROSECOND_SYM (location_type l)
      {
        return symbol_type (token::SECOND_MICROSECOND_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SECOND_MICROSECOND_SYM (const location_type& l)
      {
        return symbol_type (token::SECOND_MICROSECOND_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECOND_SYM (string v, location_type l)
      {
        return symbol_type (token::SECOND_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECOND_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECOND_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECURITY_SYM (string v, location_type l)
      {
        return symbol_type (token::SECURITY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECURITY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECURITY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SELECT_SYM (location_type l)
      {
        return symbol_type (token::SELECT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SELECT_SYM (const location_type& l)
      {
        return symbol_type (token::SELECT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SENSITIVE_SYM (location_type l)
      {
        return symbol_type (token::SENSITIVE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SENSITIVE_SYM (const location_type& l)
      {
        return symbol_type (token::SENSITIVE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SEPARATOR_SYM (location_type l)
      {
        return symbol_type (token::SEPARATOR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SEPARATOR_SYM (const location_type& l)
      {
        return symbol_type (token::SEPARATOR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SERIALIZABLE_SYM (string v, location_type l)
      {
        return symbol_type (token::SERIALIZABLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SERIALIZABLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SERIALIZABLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SERIAL_SYM (string v, location_type l)
      {
        return symbol_type (token::SERIAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SERIAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SERIAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SESSION_SYM (string v, location_type l)
      {
        return symbol_type (token::SESSION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SESSION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SESSION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SERVER_SYM (string v, location_type l)
      {
        return symbol_type (token::SERVER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SERVER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SERVER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_755 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_755, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_755 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_755, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SET (location_type l)
      {
        return symbol_type (token::SET, std::move (l));
      }
#else
      static
      symbol_type
      make_SET (const location_type& l)
      {
        return symbol_type (token::SET, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SET_VAR (location_type l)
      {
        return symbol_type (token::SET_VAR, std::move (l));
      }
#else
      static
      symbol_type
      make_SET_VAR (const location_type& l)
      {
        return symbol_type (token::SET_VAR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHARE_SYM (string v, location_type l)
      {
        return symbol_type (token::SHARE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SHARE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SHARE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHARES_SYM (string v, location_type l)
      {
        return symbol_type (token::SHARES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SHARES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SHARES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHIFT_LEFT (location_type l)
      {
        return symbol_type (token::SHIFT_LEFT, std::move (l));
      }
#else
      static
      symbol_type
      make_SHIFT_LEFT (const location_type& l)
      {
        return symbol_type (token::SHIFT_LEFT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHIFT_RIGHT (location_type l)
      {
        return symbol_type (token::SHIFT_RIGHT, std::move (l));
      }
#else
      static
      symbol_type
      make_SHIFT_RIGHT (const location_type& l)
      {
        return symbol_type (token::SHIFT_RIGHT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHOW (location_type l)
      {
        return symbol_type (token::SHOW, std::move (l));
      }
#else
      static
      symbol_type
      make_SHOW (const location_type& l)
      {
        return symbol_type (token::SHOW, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SHUTDOWN (string v, location_type l)
      {
        return symbol_type (token::SHUTDOWN, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SHUTDOWN (const string& v, const location_type& l)
      {
        return symbol_type (token::SHUTDOWN, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SIGNAL_SYM (location_type l)
      {
        return symbol_type (token::SIGNAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SIGNAL_SYM (const location_type& l)
      {
        return symbol_type (token::SIGNAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SIGNED_SYM (string v, location_type l)
      {
        return symbol_type (token::SIGNED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SIGNED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SIGNED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SIMPLE_SYM (string v, location_type l)
      {
        return symbol_type (token::SIMPLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SIMPLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SIMPLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SLAVE (string v, location_type l)
      {
        return symbol_type (token::SLAVE, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SLAVE (const string& v, const location_type& l)
      {
        return symbol_type (token::SLAVE, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SLOW (string v, location_type l)
      {
        return symbol_type (token::SLOW, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SLOW (const string& v, const location_type& l)
      {
        return symbol_type (token::SLOW, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SMALLINT_SYM (location_type l)
      {
        return symbol_type (token::SMALLINT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SMALLINT_SYM (const location_type& l)
      {
        return symbol_type (token::SMALLINT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SNAPSHOT_SYM (string v, location_type l)
      {
        return symbol_type (token::SNAPSHOT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SNAPSHOT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SNAPSHOT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SOCKET_SYM (string v, location_type l)
      {
        return symbol_type (token::SOCKET_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SOCKET_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SOCKET_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SONAME_SYM (string v, location_type l)
      {
        return symbol_type (token::SONAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SONAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SONAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SOUNDS_SYM (string v, location_type l)
      {
        return symbol_type (token::SOUNDS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SOUNDS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SOUNDS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SOURCE_SYM (string v, location_type l)
      {
        return symbol_type (token::SOURCE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SOURCE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SOURCE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SPATIAL_SYM (location_type l)
      {
        return symbol_type (token::SPATIAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SPATIAL_SYM (const location_type& l)
      {
        return symbol_type (token::SPATIAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SPECIFIC_SYM (location_type l)
      {
        return symbol_type (token::SPECIFIC_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SPECIFIC_SYM (const location_type& l)
      {
        return symbol_type (token::SPECIFIC_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQLEXCEPTION_SYM (location_type l)
      {
        return symbol_type (token::SQLEXCEPTION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SQLEXCEPTION_SYM (const location_type& l)
      {
        return symbol_type (token::SQLEXCEPTION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQLSTATE_SYM (location_type l)
      {
        return symbol_type (token::SQLSTATE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SQLSTATE_SYM (const location_type& l)
      {
        return symbol_type (token::SQLSTATE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQLWARNING_SYM (location_type l)
      {
        return symbol_type (token::SQLWARNING_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SQLWARNING_SYM (const location_type& l)
      {
        return symbol_type (token::SQLWARNING_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_AFTER_GTIDS (string v, location_type l)
      {
        return symbol_type (token::SQL_AFTER_GTIDS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_AFTER_GTIDS (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_AFTER_GTIDS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_AFTER_MTS_GAPS (string v, location_type l)
      {
        return symbol_type (token::SQL_AFTER_MTS_GAPS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_AFTER_MTS_GAPS (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_AFTER_MTS_GAPS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_BEFORE_GTIDS (string v, location_type l)
      {
        return symbol_type (token::SQL_BEFORE_GTIDS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_BEFORE_GTIDS (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_BEFORE_GTIDS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_BIG_RESULT (location_type l)
      {
        return symbol_type (token::SQL_BIG_RESULT, std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_BIG_RESULT (const location_type& l)
      {
        return symbol_type (token::SQL_BIG_RESULT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_BUFFER_RESULT (string v, location_type l)
      {
        return symbol_type (token::SQL_BUFFER_RESULT, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_BUFFER_RESULT (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_BUFFER_RESULT, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_784 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_784, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_784 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_784, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_CALC_FOUND_ROWS (location_type l)
      {
        return symbol_type (token::SQL_CALC_FOUND_ROWS, std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_CALC_FOUND_ROWS (const location_type& l)
      {
        return symbol_type (token::SQL_CALC_FOUND_ROWS, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_NO_CACHE_SYM (string v, location_type l)
      {
        return symbol_type (token::SQL_NO_CACHE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_NO_CACHE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_NO_CACHE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_SMALL_RESULT (location_type l)
      {
        return symbol_type (token::SQL_SMALL_RESULT, std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_SMALL_RESULT (const location_type& l)
      {
        return symbol_type (token::SQL_SMALL_RESULT, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_SYM (location_type l)
      {
        return symbol_type (token::SQL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_SYM (const location_type& l)
      {
        return symbol_type (token::SQL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SQL_THREAD (string v, location_type l)
      {
        return symbol_type (token::SQL_THREAD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SQL_THREAD (const string& v, const location_type& l)
      {
        return symbol_type (token::SQL_THREAD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SSL_SYM (location_type l)
      {
        return symbol_type (token::SSL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SSL_SYM (const location_type& l)
      {
        return symbol_type (token::SSL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STACKED_SYM (string v, location_type l)
      {
        return symbol_type (token::STACKED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STACKED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STACKED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STARTING (location_type l)
      {
        return symbol_type (token::STARTING, std::move (l));
      }
#else
      static
      symbol_type
      make_STARTING (const location_type& l)
      {
        return symbol_type (token::STARTING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STARTS_SYM (string v, location_type l)
      {
        return symbol_type (token::STARTS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STARTS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STARTS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_START_SYM (string v, location_type l)
      {
        return symbol_type (token::START_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_START_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::START_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STATS_AUTO_RECALC_SYM (string v, location_type l)
      {
        return symbol_type (token::STATS_AUTO_RECALC_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STATS_AUTO_RECALC_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STATS_AUTO_RECALC_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STATS_PERSISTENT_SYM (string v, location_type l)
      {
        return symbol_type (token::STATS_PERSISTENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STATS_PERSISTENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STATS_PERSISTENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STATS_SAMPLE_PAGES_SYM (string v, location_type l)
      {
        return symbol_type (token::STATS_SAMPLE_PAGES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STATS_SAMPLE_PAGES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STATS_SAMPLE_PAGES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STATUS_SYM (string v, location_type l)
      {
        return symbol_type (token::STATUS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STATUS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STATUS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STDDEV_SAMP_SYM (location_type l)
      {
        return symbol_type (token::STDDEV_SAMP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_STDDEV_SAMP_SYM (const location_type& l)
      {
        return symbol_type (token::STDDEV_SAMP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STD_SYM (location_type l)
      {
        return symbol_type (token::STD_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_STD_SYM (const location_type& l)
      {
        return symbol_type (token::STD_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STOP_SYM (string v, location_type l)
      {
        return symbol_type (token::STOP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STOP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STOP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STORAGE_SYM (string v, location_type l)
      {
        return symbol_type (token::STORAGE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STORAGE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STORAGE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STORED_SYM (location_type l)
      {
        return symbol_type (token::STORED_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_STORED_SYM (const location_type& l)
      {
        return symbol_type (token::STORED_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STRAIGHT_JOIN (location_type l)
      {
        return symbol_type (token::STRAIGHT_JOIN, std::move (l));
      }
#else
      static
      symbol_type
      make_STRAIGHT_JOIN (const location_type& l)
      {
        return symbol_type (token::STRAIGHT_JOIN, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_STRING_SYM (string v, location_type l)
      {
        return symbol_type (token::STRING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_STRING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::STRING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBCLASS_ORIGIN_SYM (string v, location_type l)
      {
        return symbol_type (token::SUBCLASS_ORIGIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUBCLASS_ORIGIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUBCLASS_ORIGIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBDATE_SYM (string v, location_type l)
      {
        return symbol_type (token::SUBDATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUBDATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUBDATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBJECT_SYM (string v, location_type l)
      {
        return symbol_type (token::SUBJECT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUBJECT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUBJECT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBPARTITIONS_SYM (string v, location_type l)
      {
        return symbol_type (token::SUBPARTITIONS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUBPARTITIONS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUBPARTITIONS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBPARTITION_SYM (string v, location_type l)
      {
        return symbol_type (token::SUBPARTITION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUBPARTITION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUBPARTITION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBSTRING (location_type l)
      {
        return symbol_type (token::SUBSTRING, std::move (l));
      }
#else
      static
      symbol_type
      make_SUBSTRING (const location_type& l)
      {
        return symbol_type (token::SUBSTRING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUM_SYM (location_type l)
      {
        return symbol_type (token::SUM_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SUM_SYM (const location_type& l)
      {
        return symbol_type (token::SUM_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUPER_SYM (string v, location_type l)
      {
        return symbol_type (token::SUPER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUPER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUPER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUSPEND_SYM (string v, location_type l)
      {
        return symbol_type (token::SUSPEND_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SUSPEND_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SUSPEND_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SWAPS_SYM (string v, location_type l)
      {
        return symbol_type (token::SWAPS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SWAPS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SWAPS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SWITCHES_SYM (string v, location_type l)
      {
        return symbol_type (token::SWITCHES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SWITCHES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SWITCHES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SYSDATE (location_type l)
      {
        return symbol_type (token::SYSDATE, std::move (l));
      }
#else
      static
      symbol_type
      make_SYSDATE (const location_type& l)
      {
        return symbol_type (token::SYSDATE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TABLES (string v, location_type l)
      {
        return symbol_type (token::TABLES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TABLES (const string& v, const location_type& l)
      {
        return symbol_type (token::TABLES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VIEWS (string v, location_type l)
      {
        return symbol_type (token::VIEWS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VIEWS (const string& v, const location_type& l)
      {
        return symbol_type (token::VIEWS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TABLESPACE_SYM (string v, location_type l)
      {
        return symbol_type (token::TABLESPACE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TABLESPACE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TABLESPACE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_820 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_820, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_820 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_820, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TABLE_SYM (location_type l)
      {
        return symbol_type (token::TABLE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TABLE_SYM (const location_type& l)
      {
        return symbol_type (token::TABLE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TABLE_CHECKSUM_SYM (string v, location_type l)
      {
        return symbol_type (token::TABLE_CHECKSUM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TABLE_CHECKSUM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TABLE_CHECKSUM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TABLE_NAME_SYM (string v, location_type l)
      {
        return symbol_type (token::TABLE_NAME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TABLE_NAME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TABLE_NAME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TEMPORARY (string v, location_type l)
      {
        return symbol_type (token::TEMPORARY, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TEMPORARY (const string& v, const location_type& l)
      {
        return symbol_type (token::TEMPORARY, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TEMPTABLE_SYM (string v, location_type l)
      {
        return symbol_type (token::TEMPTABLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TEMPTABLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TEMPTABLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TERMINATED (location_type l)
      {
        return symbol_type (token::TERMINATED, std::move (l));
      }
#else
      static
      symbol_type
      make_TERMINATED (const location_type& l)
      {
        return symbol_type (token::TERMINATED, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TEXT_STRING (string v, location_type l)
      {
        return symbol_type (token::TEXT_STRING, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TEXT_STRING (const string& v, const location_type& l)
      {
        return symbol_type (token::TEXT_STRING, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TEXT_SYM (string v, location_type l)
      {
        return symbol_type (token::TEXT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TEXT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TEXT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_THAN_SYM (string v, location_type l)
      {
        return symbol_type (token::THAN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_THAN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::THAN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_THEN_SYM (location_type l)
      {
        return symbol_type (token::THEN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_THEN_SYM (const location_type& l)
      {
        return symbol_type (token::THEN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TIMESTAMP_SYM (string v, location_type l)
      {
        return symbol_type (token::TIMESTAMP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TIMESTAMP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TIMESTAMP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TIMESTAMP_ADD (string v, location_type l)
      {
        return symbol_type (token::TIMESTAMP_ADD, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TIMESTAMP_ADD (const string& v, const location_type& l)
      {
        return symbol_type (token::TIMESTAMP_ADD, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TIMESTAMP_DIFF (string v, location_type l)
      {
        return symbol_type (token::TIMESTAMP_DIFF, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TIMESTAMP_DIFF (const string& v, const location_type& l)
      {
        return symbol_type (token::TIMESTAMP_DIFF, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TIME_SYM (string v, location_type l)
      {
        return symbol_type (token::TIME_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TIME_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TIME_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TINYBLOB_SYM (location_type l)
      {
        return symbol_type (token::TINYBLOB_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TINYBLOB_SYM (const location_type& l)
      {
        return symbol_type (token::TINYBLOB_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TINYINT_SYM (location_type l)
      {
        return symbol_type (token::TINYINT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TINYINT_SYM (const location_type& l)
      {
        return symbol_type (token::TINYINT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TINYTEXT_SYN (location_type l)
      {
        return symbol_type (token::TINYTEXT_SYN, std::move (l));
      }
#else
      static
      symbol_type
      make_TINYTEXT_SYN (const location_type& l)
      {
        return symbol_type (token::TINYTEXT_SYN, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TO_SYM (location_type l)
      {
        return symbol_type (token::TO_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TO_SYM (const location_type& l)
      {
        return symbol_type (token::TO_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRAILING (location_type l)
      {
        return symbol_type (token::TRAILING, std::move (l));
      }
#else
      static
      symbol_type
      make_TRAILING (const location_type& l)
      {
        return symbol_type (token::TRAILING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRANSACTION_SYM (string v, location_type l)
      {
        return symbol_type (token::TRANSACTION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TRANSACTION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TRANSACTION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRIGGERS_SYM (string v, location_type l)
      {
        return symbol_type (token::TRIGGERS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TRIGGERS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TRIGGERS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRIGGER_SYM (location_type l)
      {
        return symbol_type (token::TRIGGER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TRIGGER_SYM (const location_type& l)
      {
        return symbol_type (token::TRIGGER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRIM (location_type l)
      {
        return symbol_type (token::TRIM, std::move (l));
      }
#else
      static
      symbol_type
      make_TRIM (const location_type& l)
      {
        return symbol_type (token::TRIM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRUE_SYM (location_type l)
      {
        return symbol_type (token::TRUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TRUE_SYM (const location_type& l)
      {
        return symbol_type (token::TRUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRUNCATE_SYM (string v, location_type l)
      {
        return symbol_type (token::TRUNCATE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TRUNCATE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TRUNCATE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TYPES_SYM (string v, location_type l)
      {
        return symbol_type (token::TYPES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TYPES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TYPES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TYPE_SYM (string v, location_type l)
      {
        return symbol_type (token::TYPE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TYPE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TYPE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_848 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_848, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_848 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_848, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ULONGLONG_NUM (string v, location_type l)
      {
        return symbol_type (token::ULONGLONG_NUM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ULONGLONG_NUM (const string& v, const location_type& l)
      {
        return symbol_type (token::ULONGLONG_NUM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNCOMMITTED_SYM (string v, location_type l)
      {
        return symbol_type (token::UNCOMMITTED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNCOMMITTED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNCOMMITTED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNDEFINED_SYM (string v, location_type l)
      {
        return symbol_type (token::UNDEFINED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNDEFINED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNDEFINED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNDERSCORE_CHARSET (location_type l)
      {
        return symbol_type (token::UNDERSCORE_CHARSET, std::move (l));
      }
#else
      static
      symbol_type
      make_UNDERSCORE_CHARSET (const location_type& l)
      {
        return symbol_type (token::UNDERSCORE_CHARSET, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNDOFILE_SYM (string v, location_type l)
      {
        return symbol_type (token::UNDOFILE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNDOFILE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNDOFILE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNDO_BUFFER_SIZE_SYM (string v, location_type l)
      {
        return symbol_type (token::UNDO_BUFFER_SIZE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNDO_BUFFER_SIZE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNDO_BUFFER_SIZE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNDO_SYM (location_type l)
      {
        return symbol_type (token::UNDO_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UNDO_SYM (const location_type& l)
      {
        return symbol_type (token::UNDO_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNICODE_SYM (string v, location_type l)
      {
        return symbol_type (token::UNICODE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNICODE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNICODE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNINSTALL_SYM (string v, location_type l)
      {
        return symbol_type (token::UNINSTALL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNINSTALL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNINSTALL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNION_SYM (location_type l)
      {
        return symbol_type (token::UNION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UNION_SYM (const location_type& l)
      {
        return symbol_type (token::UNION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNIQUE_SYM (location_type l)
      {
        return symbol_type (token::UNIQUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UNIQUE_SYM (const location_type& l)
      {
        return symbol_type (token::UNIQUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNKNOWN_SYM (string v, location_type l)
      {
        return symbol_type (token::UNKNOWN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNKNOWN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNKNOWN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNLOCK_SYM (location_type l)
      {
        return symbol_type (token::UNLOCK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UNLOCK_SYM (const location_type& l)
      {
        return symbol_type (token::UNLOCK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNSIGNED_SYM (location_type l)
      {
        return symbol_type (token::UNSIGNED_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UNSIGNED_SYM (const location_type& l)
      {
        return symbol_type (token::UNSIGNED_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNTIL_SYM (string v, location_type l)
      {
        return symbol_type (token::UNTIL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNTIL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNTIL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UPDATE_SYM (location_type l)
      {
        return symbol_type (token::UPDATE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UPDATE_SYM (const location_type& l)
      {
        return symbol_type (token::UPDATE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UPGRADE_SYM (string v, location_type l)
      {
        return symbol_type (token::UPGRADE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UPGRADE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UPGRADE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_USAGE (location_type l)
      {
        return symbol_type (token::USAGE, std::move (l));
      }
#else
      static
      symbol_type
      make_USAGE (const location_type& l)
      {
        return symbol_type (token::USAGE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_USER (string v, location_type l)
      {
        return symbol_type (token::USER, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_USER (const string& v, const location_type& l)
      {
        return symbol_type (token::USER, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_USE_FRM (string v, location_type l)
      {
        return symbol_type (token::USE_FRM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_USE_FRM (const string& v, const location_type& l)
      {
        return symbol_type (token::USE_FRM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_USE_SYM (location_type l)
      {
        return symbol_type (token::USE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_USE_SYM (const location_type& l)
      {
        return symbol_type (token::USE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_USING (location_type l)
      {
        return symbol_type (token::USING, std::move (l));
      }
#else
      static
      symbol_type
      make_USING (const location_type& l)
      {
        return symbol_type (token::USING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UTC_DATE_SYM (location_type l)
      {
        return symbol_type (token::UTC_DATE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UTC_DATE_SYM (const location_type& l)
      {
        return symbol_type (token::UTC_DATE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UTC_TIMESTAMP_SYM (location_type l)
      {
        return symbol_type (token::UTC_TIMESTAMP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UTC_TIMESTAMP_SYM (const location_type& l)
      {
        return symbol_type (token::UTC_TIMESTAMP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UTC_TIME_SYM (location_type l)
      {
        return symbol_type (token::UTC_TIME_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_UTC_TIME_SYM (const location_type& l)
      {
        return symbol_type (token::UTC_TIME_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VALIDATION_SYM (string v, location_type l)
      {
        return symbol_type (token::VALIDATION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VALIDATION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VALIDATION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VALUES (location_type l)
      {
        return symbol_type (token::VALUES, std::move (l));
      }
#else
      static
      symbol_type
      make_VALUES (const location_type& l)
      {
        return symbol_type (token::VALUES, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VALUE_SYM (string v, location_type l)
      {
        return symbol_type (token::VALUE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VALUE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VALUE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARBINARY_SYM (string v, location_type l)
      {
        return symbol_type (token::VARBINARY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VARBINARY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VARBINARY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARCHAR_SYM (location_type l)
      {
        return symbol_type (token::VARCHAR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_VARCHAR_SYM (const location_type& l)
      {
        return symbol_type (token::VARCHAR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARIABLES (string v, location_type l)
      {
        return symbol_type (token::VARIABLES, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VARIABLES (const string& v, const location_type& l)
      {
        return symbol_type (token::VARIABLES, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARIANCE_SYM (location_type l)
      {
        return symbol_type (token::VARIANCE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_VARIANCE_SYM (const location_type& l)
      {
        return symbol_type (token::VARIANCE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VARYING (location_type l)
      {
        return symbol_type (token::VARYING, std::move (l));
      }
#else
      static
      symbol_type
      make_VARYING (const location_type& l)
      {
        return symbol_type (token::VARYING, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VAR_SAMP_SYM (location_type l)
      {
        return symbol_type (token::VAR_SAMP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_VAR_SAMP_SYM (const location_type& l)
      {
        return symbol_type (token::VAR_SAMP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VERSION_SYM (location_type l)
      {
        return symbol_type (token::VERSION_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_VERSION_SYM (const location_type& l)
      {
        return symbol_type (token::VERSION_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VIEW_SYM (string v, location_type l)
      {
        return symbol_type (token::VIEW_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VIEW_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VIEW_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VIRTUAL_SYM (location_type l)
      {
        return symbol_type (token::VIRTUAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_VIRTUAL_SYM (const location_type& l)
      {
        return symbol_type (token::VIRTUAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WAIT_SYM (string v, location_type l)
      {
        return symbol_type (token::WAIT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WAIT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WAIT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WARNINGS (string v, location_type l)
      {
        return symbol_type (token::WARNINGS, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WARNINGS (const string& v, const location_type& l)
      {
        return symbol_type (token::WARNINGS, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WEEK_SYM (string v, location_type l)
      {
        return symbol_type (token::WEEK_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WEEK_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WEEK_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WEIGHT_STRING_SYM (string v, location_type l)
      {
        return symbol_type (token::WEIGHT_STRING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WEIGHT_STRING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WEIGHT_STRING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WHEN_SYM (location_type l)
      {
        return symbol_type (token::WHEN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_WHEN_SYM (const location_type& l)
      {
        return symbol_type (token::WHEN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WHERE (location_type l)
      {
        return symbol_type (token::WHERE, std::move (l));
      }
#else
      static
      symbol_type
      make_WHERE (const location_type& l)
      {
        return symbol_type (token::WHERE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WHILE_SYM (location_type l)
      {
        return symbol_type (token::WHILE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_WHILE_SYM (const location_type& l)
      {
        return symbol_type (token::WHILE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WITH (location_type l)
      {
        return symbol_type (token::WITH, std::move (l));
      }
#else
      static
      symbol_type
      make_WITH (const location_type& l)
      {
        return symbol_type (token::WITH, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_893 (location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_893, std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_893 (const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_893, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WITH_ROLLUP_SYM (location_type l)
      {
        return symbol_type (token::WITH_ROLLUP_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_WITH_ROLLUP_SYM (const location_type& l)
      {
        return symbol_type (token::WITH_ROLLUP_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WITHOUT_SYM (string v, location_type l)
      {
        return symbol_type (token::WITHOUT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WITHOUT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WITHOUT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WORK_SYM (string v, location_type l)
      {
        return symbol_type (token::WORK_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WORK_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WORK_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WRAPPER_SYM (string v, location_type l)
      {
        return symbol_type (token::WRAPPER_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_WRAPPER_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::WRAPPER_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WRITE_SYM (location_type l)
      {
        return symbol_type (token::WRITE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_WRITE_SYM (const location_type& l)
      {
        return symbol_type (token::WRITE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_X509_SYM (string v, location_type l)
      {
        return symbol_type (token::X509_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_X509_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::X509_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_XA_SYM (string v, location_type l)
      {
        return symbol_type (token::XA_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_XA_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::XA_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_XID_SYM (string v, location_type l)
      {
        return symbol_type (token::XID_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_XID_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::XID_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_XML_SYM (string v, location_type l)
      {
        return symbol_type (token::XML_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_XML_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::XML_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_XOR (location_type l)
      {
        return symbol_type (token::XOR, std::move (l));
      }
#else
      static
      symbol_type
      make_XOR (const location_type& l)
      {
        return symbol_type (token::XOR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YEAR_MONTH_SYM (location_type l)
      {
        return symbol_type (token::YEAR_MONTH_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_YEAR_MONTH_SYM (const location_type& l)
      {
        return symbol_type (token::YEAR_MONTH_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_YEAR_SYM (string v, location_type l)
      {
        return symbol_type (token::YEAR_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_YEAR_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::YEAR_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ZEROFILL_SYM (location_type l)
      {
        return symbol_type (token::ZEROFILL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ZEROFILL_SYM (const location_type& l)
      {
        return symbol_type (token::ZEROFILL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXPLAIN_SYM (location_type l)
      {
        return symbol_type (token::EXPLAIN_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EXPLAIN_SYM (const location_type& l)
      {
        return symbol_type (token::EXPLAIN_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TREE_SYM (location_type l)
      {
        return symbol_type (token::TREE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TREE_SYM (const location_type& l)
      {
        return symbol_type (token::TREE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TRADITIONAL_SYM (location_type l)
      {
        return symbol_type (token::TRADITIONAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_TRADITIONAL_SYM (const location_type& l)
      {
        return symbol_type (token::TRADITIONAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_UNQUOTED_SEPARATOR_SYM (location_type l)
      {
        return symbol_type (token::JSON_UNQUOTED_SEPARATOR_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_UNQUOTED_SEPARATOR_SYM (const location_type& l)
      {
        return symbol_type (token::JSON_UNQUOTED_SEPARATOR_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PERSIST_SYM (string v, location_type l)
      {
        return symbol_type (token::PERSIST_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PERSIST_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PERSIST_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROLE_SYM (string v, location_type l)
      {
        return symbol_type (token::ROLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ROLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ROLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ADMIN_SYM (string v, location_type l)
      {
        return symbol_type (token::ADMIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ADMIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ADMIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INVISIBLE_SYM (string v, location_type l)
      {
        return symbol_type (token::INVISIBLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INVISIBLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INVISIBLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VISIBLE_SYM (string v, location_type l)
      {
        return symbol_type (token::VISIBLE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VISIBLE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VISIBLE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXCEPT_SYM (location_type l)
      {
        return symbol_type (token::EXCEPT_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EXCEPT_SYM (const location_type& l)
      {
        return symbol_type (token::EXCEPT_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_COMPONENT_SYM (string v, location_type l)
      {
        return symbol_type (token::COMPONENT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_COMPONENT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::COMPONENT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RECURSIVE_SYM (location_type l)
      {
        return symbol_type (token::RECURSIVE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RECURSIVE_SYM (const location_type& l)
      {
        return symbol_type (token::RECURSIVE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRAMMAR_SELECTOR_EXPR (location_type l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_EXPR, std::move (l));
      }
#else
      static
      symbol_type
      make_GRAMMAR_SELECTOR_EXPR (const location_type& l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_EXPR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRAMMAR_SELECTOR_GCOL (location_type l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_GCOL, std::move (l));
      }
#else
      static
      symbol_type
      make_GRAMMAR_SELECTOR_GCOL (const location_type& l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_GCOL, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRAMMAR_SELECTOR_PART (location_type l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_PART, std::move (l));
      }
#else
      static
      symbol_type
      make_GRAMMAR_SELECTOR_PART (const location_type& l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_PART, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GRAMMAR_SELECTOR_CTE (location_type l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_CTE, std::move (l));
      }
#else
      static
      symbol_type
      make_GRAMMAR_SELECTOR_CTE (const location_type& l)
      {
        return symbol_type (token::GRAMMAR_SELECTOR_CTE, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_OBJECTAGG (location_type l)
      {
        return symbol_type (token::JSON_OBJECTAGG, std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_OBJECTAGG (const location_type& l)
      {
        return symbol_type (token::JSON_OBJECTAGG, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_ARRAYAGG (location_type l)
      {
        return symbol_type (token::JSON_ARRAYAGG, std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_ARRAYAGG (const location_type& l)
      {
        return symbol_type (token::JSON_ARRAYAGG, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OF_SYM (location_type l)
      {
        return symbol_type (token::OF_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OF_SYM (const location_type& l)
      {
        return symbol_type (token::OF_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SKIP_SYM (string v, location_type l)
      {
        return symbol_type (token::SKIP_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SKIP_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SKIP_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LOCKED_SYM (string v, location_type l)
      {
        return symbol_type (token::LOCKED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_LOCKED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::LOCKED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NOWAIT_SYM (string v, location_type l)
      {
        return symbol_type (token::NOWAIT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NOWAIT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NOWAIT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GROUPING_SYM (location_type l)
      {
        return symbol_type (token::GROUPING_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GROUPING_SYM (const location_type& l)
      {
        return symbol_type (token::GROUPING_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PERSIST_ONLY_SYM (string v, location_type l)
      {
        return symbol_type (token::PERSIST_ONLY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PERSIST_ONLY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PERSIST_ONLY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HISTOGRAM_SYM (string v, location_type l)
      {
        return symbol_type (token::HISTOGRAM_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HISTOGRAM_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HISTOGRAM_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_BUCKETS_SYM (string v, location_type l)
      {
        return symbol_type (token::BUCKETS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_BUCKETS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::BUCKETS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OBSOLETE_TOKEN_930 (string v, location_type l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_930, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OBSOLETE_TOKEN_930 (const string& v, const location_type& l)
      {
        return symbol_type (token::OBSOLETE_TOKEN_930, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CLONE_SYM (string v, location_type l)
      {
        return symbol_type (token::CLONE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_CLONE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::CLONE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CUME_DIST_SYM (location_type l)
      {
        return symbol_type (token::CUME_DIST_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_CUME_DIST_SYM (const location_type& l)
      {
        return symbol_type (token::CUME_DIST_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DENSE_RANK_SYM (location_type l)
      {
        return symbol_type (token::DENSE_RANK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_DENSE_RANK_SYM (const location_type& l)
      {
        return symbol_type (token::DENSE_RANK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EXCLUDE_SYM (string v, location_type l)
      {
        return symbol_type (token::EXCLUDE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_EXCLUDE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::EXCLUDE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FIRST_VALUE_SYM (location_type l)
      {
        return symbol_type (token::FIRST_VALUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_FIRST_VALUE_SYM (const location_type& l)
      {
        return symbol_type (token::FIRST_VALUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_FOLLOWING_SYM (string v, location_type l)
      {
        return symbol_type (token::FOLLOWING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_FOLLOWING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::FOLLOWING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GROUPS_SYM (location_type l)
      {
        return symbol_type (token::GROUPS_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_GROUPS_SYM (const location_type& l)
      {
        return symbol_type (token::GROUPS_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LAG_SYM (location_type l)
      {
        return symbol_type (token::LAG_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LAG_SYM (const location_type& l)
      {
        return symbol_type (token::LAG_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LAST_VALUE_SYM (location_type l)
      {
        return symbol_type (token::LAST_VALUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LAST_VALUE_SYM (const location_type& l)
      {
        return symbol_type (token::LAST_VALUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LEAD_SYM (location_type l)
      {
        return symbol_type (token::LEAD_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LEAD_SYM (const location_type& l)
      {
        return symbol_type (token::LEAD_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NTH_VALUE_SYM (location_type l)
      {
        return symbol_type (token::NTH_VALUE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NTH_VALUE_SYM (const location_type& l)
      {
        return symbol_type (token::NTH_VALUE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NTILE_SYM (location_type l)
      {
        return symbol_type (token::NTILE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_NTILE_SYM (const location_type& l)
      {
        return symbol_type (token::NTILE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NULLS_SYM (string v, location_type l)
      {
        return symbol_type (token::NULLS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NULLS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NULLS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OTHERS_SYM (string v, location_type l)
      {
        return symbol_type (token::OTHERS_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OTHERS_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OTHERS_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OVER_SYM (location_type l)
      {
        return symbol_type (token::OVER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_OVER_SYM (const location_type& l)
      {
        return symbol_type (token::OVER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PERCENT_RANK_SYM (location_type l)
      {
        return symbol_type (token::PERCENT_RANK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_PERCENT_RANK_SYM (const location_type& l)
      {
        return symbol_type (token::PERCENT_RANK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PRECEDING_SYM (string v, location_type l)
      {
        return symbol_type (token::PRECEDING_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PRECEDING_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PRECEDING_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RANK_SYM (location_type l)
      {
        return symbol_type (token::RANK_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_RANK_SYM (const location_type& l)
      {
        return symbol_type (token::RANK_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESPECT_SYM (string v, location_type l)
      {
        return symbol_type (token::RESPECT_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESPECT_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESPECT_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ROW_NUMBER_SYM (location_type l)
      {
        return symbol_type (token::ROW_NUMBER_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ROW_NUMBER_SYM (const location_type& l)
      {
        return symbol_type (token::ROW_NUMBER_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_TIES_SYM (string v, location_type l)
      {
        return symbol_type (token::TIES_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_TIES_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::TIES_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_UNBOUNDED_SYM (string v, location_type l)
      {
        return symbol_type (token::UNBOUNDED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_UNBOUNDED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::UNBOUNDED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_WINDOW_SYM (location_type l)
      {
        return symbol_type (token::WINDOW_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_WINDOW_SYM (const location_type& l)
      {
        return symbol_type (token::WINDOW_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EMPTY_SYM (location_type l)
      {
        return symbol_type (token::EMPTY_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_EMPTY_SYM (const location_type& l)
      {
        return symbol_type (token::EMPTY_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_JSON_TABLE_SYM (location_type l)
      {
        return symbol_type (token::JSON_TABLE_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_JSON_TABLE_SYM (const location_type& l)
      {
        return symbol_type (token::JSON_TABLE_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NESTED_SYM (string v, location_type l)
      {
        return symbol_type (token::NESTED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NESTED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NESTED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ORDINALITY_SYM (string v, location_type l)
      {
        return symbol_type (token::ORDINALITY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ORDINALITY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ORDINALITY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_PATH_SYM (string v, location_type l)
      {
        return symbol_type (token::PATH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_PATH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::PATH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_HISTORY_SYM (string v, location_type l)
      {
        return symbol_type (token::HISTORY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_HISTORY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::HISTORY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REUSE_SYM (string v, location_type l)
      {
        return symbol_type (token::REUSE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REUSE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REUSE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SRID_SYM (string v, location_type l)
      {
        return symbol_type (token::SRID_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SRID_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SRID_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_THREAD_PRIORITY_SYM (string v, location_type l)
      {
        return symbol_type (token::THREAD_PRIORITY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_THREAD_PRIORITY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::THREAD_PRIORITY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESOURCE_SYM (string v, location_type l)
      {
        return symbol_type (token::RESOURCE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESOURCE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESOURCE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SYSTEM_SYM (location_type l)
      {
        return symbol_type (token::SYSTEM_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_SYSTEM_SYM (const location_type& l)
      {
        return symbol_type (token::SYSTEM_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_VCPU_SYM (string v, location_type l)
      {
        return symbol_type (token::VCPU_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_VCPU_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::VCPU_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MASTER_PUBLIC_KEY_PATH_SYM (string v, location_type l)
      {
        return symbol_type (token::MASTER_PUBLIC_KEY_PATH_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_MASTER_PUBLIC_KEY_PATH_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::MASTER_PUBLIC_KEY_PATH_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_GET_MASTER_PUBLIC_KEY_SYM (string v, location_type l)
      {
        return symbol_type (token::GET_MASTER_PUBLIC_KEY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_GET_MASTER_PUBLIC_KEY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::GET_MASTER_PUBLIC_KEY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RESTART_SYM (string v, location_type l)
      {
        return symbol_type (token::RESTART_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RESTART_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RESTART_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DEFINITION_SYM (string v, location_type l)
      {
        return symbol_type (token::DEFINITION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DEFINITION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DEFINITION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_DESCRIPTION_SYM (string v, location_type l)
      {
        return symbol_type (token::DESCRIPTION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_DESCRIPTION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::DESCRIPTION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ORGANIZATION_SYM (string v, location_type l)
      {
        return symbol_type (token::ORGANIZATION_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ORGANIZATION_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ORGANIZATION_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_REFERENCE_SYM (string v, location_type l)
      {
        return symbol_type (token::REFERENCE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_REFERENCE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::REFERENCE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ACTIVE_SYM (string v, location_type l)
      {
        return symbol_type (token::ACTIVE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ACTIVE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ACTIVE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_INACTIVE_SYM (string v, location_type l)
      {
        return symbol_type (token::INACTIVE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_INACTIVE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::INACTIVE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_LATERAL_SYM (location_type l)
      {
        return symbol_type (token::LATERAL_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_LATERAL_SYM (const location_type& l)
      {
        return symbol_type (token::LATERAL_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OPTIONAL_SYM (string v, location_type l)
      {
        return symbol_type (token::OPTIONAL_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OPTIONAL_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OPTIONAL_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECONDARY_SYM (string v, location_type l)
      {
        return symbol_type (token::SECONDARY_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECONDARY_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECONDARY_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECONDARY_ENGINE_SYM (string v, location_type l)
      {
        return symbol_type (token::SECONDARY_ENGINE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECONDARY_ENGINE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECONDARY_ENGINE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECONDARY_LOAD_SYM (string v, location_type l)
      {
        return symbol_type (token::SECONDARY_LOAD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECONDARY_LOAD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECONDARY_LOAD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SECONDARY_UNLOAD_SYM (string v, location_type l)
      {
        return symbol_type (token::SECONDARY_UNLOAD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_SECONDARY_UNLOAD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::SECONDARY_UNLOAD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_RETAIN_SYM (string v, location_type l)
      {
        return symbol_type (token::RETAIN_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_RETAIN_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::RETAIN_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OLD_SYM (string v, location_type l)
      {
        return symbol_type (token::OLD_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OLD_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OLD_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ENFORCED_SYM (string v, location_type l)
      {
        return symbol_type (token::ENFORCED_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_ENFORCED_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::ENFORCED_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_OJ_SYM (string v, location_type l)
      {
        return symbol_type (token::OJ_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_OJ_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::OJ_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_NETWORK_NAMESPACE_SYM (string v, location_type l)
      {
        return symbol_type (token::NETWORK_NAMESPACE_SYM, std::move (v), std::move (l));
      }
#else
      static
      symbol_type
      make_NETWORK_NAMESPACE_SYM (const string& v, const location_type& l)
      {
        return symbol_type (token::NETWORK_NAMESPACE_SYM, v, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_ADD_SYM (location_type l)
      {
        return symbol_type (token::ADD_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_ADD_SYM (const location_type& l)
      {
        return symbol_type (token::ADD_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_MINUS_SYM (location_type l)
      {
        return symbol_type (token::MINUS_SYM, std::move (l));
      }
#else
      static
      symbol_type
      make_MINUS_SYM (const location_type& l)
      {
        return symbol_type (token::MINUS_SYM, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_CONDITIONLESS_JOIN (location_type l)
      {
        return symbol_type (token::CONDITIONLESS_JOIN, std::move (l));
      }
#else
      static
      symbol_type
      make_CONDITIONLESS_JOIN (const location_type& l)
      {
        return symbol_type (token::CONDITIONLESS_JOIN, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_SUBQUERY_AS_EXPR (location_type l)
      {
        return symbol_type (token::SUBQUERY_AS_EXPR, std::move (l));
      }
#else
      static
      symbol_type
      make_SUBQUERY_AS_EXPR (const location_type& l)
      {
        return symbol_type (token::SUBQUERY_AS_EXPR, l);
      }
#endif
#if 201103L <= YY_CPLUSPLUS
      static
      symbol_type
      make_EMPTY_FROM_CLAUSE (location_type l)
      {
        return symbol_type (token::EMPTY_FROM_CLAUSE, std::move (l));
      }
#else
      static
      symbol_type
      make_EMPTY_FROM_CLAUSE (const location_type& l)
      {
        return symbol_type (token::EMPTY_FROM_CLAUSE, l);
      }
#endif


  private:
    /// This class is not copyable.
    Parser (const Parser&);
    Parser& operator= (const Parser&);

    /// Stored state numbers (used for stacks).
    typedef short state_type;

    /// Generate an error message.
    /// \param yystate   the state where the error occurred.
    /// \param yyla      the lookahead token.
    virtual std::string yysyntax_error_ (state_type yystate,
                                         const symbol_type& yyla) const;

    /// Compute post-reduction state.
    /// \param yystate   the current state
    /// \param yysym     the nonterminal to push on the stack
    static state_type yy_lr_goto_state_ (state_type yystate, int yysym);

    /// Whether the given \c yypact_ value indicates a defaulted state.
    /// \param yyvalue   the value to check
    static bool yy_pact_value_is_default_ (int yyvalue);

    /// Whether the given \c yytable_ value indicates a syntax error.
    /// \param yyvalue   the value to check
    static bool yy_table_value_is_error_ (int yyvalue);

    static const short yypact_ninf_;
    static const short yytable_ninf_;

    /// Convert a scanner token number \a t to a symbol number.
    /// In theory \a t should be a token_type, but character literals
    /// are valid, yet not members of the token_type enum.
    static token_number_type yytranslate_ (int t);

    // Tables.
    // YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
    // STATE-NUM.
    static const int yypact_[];

    // YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
    // Performed when YYTABLE does not specify something else to do.  Zero
    // means the default is an error.
    static const short yydefact_[];

    // YYPGOTO[NTERM-NUM].
    static const short yypgoto_[];

    // YYDEFGOTO[NTERM-NUM].
    static const short yydefgoto_[];

    // YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
    // positive, shift that token.  If negative, reduce the rule whose
    // number is the opposite.  If YYTABLE_NINF, syntax error.
    static const short yytable_[];

    static const short yycheck_[];

    // YYSTOS[STATE-NUM] -- The (internal number of the) accessing
    // symbol of state STATE-NUM.
    static const short yystos_[];

    // YYR1[YYN] -- Symbol number of symbol that rule YYN derives.
    static const short yyr1_[];

    // YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.
    static const signed char yyr2_[];


#if ARIES_PARSERDEBUG
    /// For a symbol, its name in clear.
    static const char* const yytname_[];

    // YYRLINE[YYN] -- Source line where rule number YYN was defined.
    static const short yyrline_[];
    /// Report on the debug stream that the rule \a r is going to be reduced.
    virtual void yy_reduce_print_ (int r);
    /// Print the state stack on the debug stream.
    virtual void yystack_print_ ();

    /// Debugging level.
    int yydebug_;
    /// Debug stream.
    std::ostream* yycdebug_;

    /// \brief Display a symbol type, value and location.
    /// \param yyo    The output stream.
    /// \param yysym  The symbol.
    template <typename Base>
    void yy_print_ (std::ostream& yyo, const basic_symbol<Base>& yysym) const;
#endif

    /// \brief Reclaim the memory associated to a symbol.
    /// \param yymsg     Why this token is reclaimed.
    ///                  If null, print nothing.
    /// \param yysym     The symbol.
    template <typename Base>
    void yy_destroy_ (const char* yymsg, basic_symbol<Base>& yysym) const;

  private:
    /// Type access provider for state based symbols.
    struct by_state
    {
      /// Default constructor.
      by_state () YY_NOEXCEPT;

      /// The symbol type as needed by the constructor.
      typedef state_type kind_type;

      /// Constructor.
      by_state (kind_type s) YY_NOEXCEPT;

      /// Copy constructor.
      by_state (const by_state& that) YY_NOEXCEPT;

      /// Record that this symbol is empty.
      void clear () YY_NOEXCEPT;

      /// Steal the symbol type from \a that.
      void move (by_state& that);

      /// The (internal) type number (corresponding to \a state).
      /// \a empty_symbol when empty.
      symbol_number_type type_get () const YY_NOEXCEPT;

      /// The state number used to denote an empty symbol.
      /// We use the initial state, as it does not have a value.
      enum { empty_state = 0 };

      /// The state.
      /// \a empty when empty.
      state_type state;
    };

    /// "Internal" symbol: element of the stack.
    struct stack_symbol_type : basic_symbol<by_state>
    {
      /// Superclass.
      typedef basic_symbol<by_state> super_type;
      /// Construct an empty symbol.
      stack_symbol_type ();
      /// Move or copy construction.
      stack_symbol_type (YY_RVREF (stack_symbol_type) that);
      /// Steal the contents from \a sym to build this.
      stack_symbol_type (state_type s, YY_MOVE_REF (symbol_type) sym);
#if YY_CPLUSPLUS < 201103L
      /// Assignment, needed by push_back by some old implementations.
      /// Moves the contents of that.
      stack_symbol_type& operator= (stack_symbol_type& that);

      /// Assignment, needed by push_back by other implementations.
      /// Needed by some other old implementations.
      stack_symbol_type& operator= (const stack_symbol_type& that);
#endif
    };

    /// A stack with random access from its top.
    template <typename T, typename S = std::vector<T> >
    class stack
    {
    public:
      // Hide our reversed order.
      typedef typename S::reverse_iterator iterator;
      typedef typename S::const_reverse_iterator const_iterator;
      typedef typename S::size_type size_type;
      typedef typename std::ptrdiff_t index_type;

      stack (size_type n = 200)
        : seq_ (n)
      {}

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      const T&
      operator[] (index_type i) const
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Random access.
      ///
      /// Index 0 returns the topmost element.
      T&
      operator[] (index_type i)
      {
        return seq_[size_type (size () - 1 - i)];
      }

      /// Steal the contents of \a t.
      ///
      /// Close to move-semantics.
      void
      push (YY_MOVE_REF (T) t)
      {
        seq_.push_back (T ());
        operator[] (0).move (t);
      }

      /// Pop elements from the stack.
      void
      pop (std::ptrdiff_t n = 1) YY_NOEXCEPT
      {
        for (; 0 < n; --n)
          seq_.pop_back ();
      }

      /// Pop all elements from the stack.
      void
      clear () YY_NOEXCEPT
      {
        seq_.clear ();
      }

      /// Number of elements on the stack.
      index_type
      size () const YY_NOEXCEPT
      {
        return index_type (seq_.size ());
      }

      std::ptrdiff_t
      ssize () const YY_NOEXCEPT
      {
        return std::ptrdiff_t (size ());
      }

      /// Iterator on top of the stack (going downwards).
      const_iterator
      begin () const YY_NOEXCEPT
      {
        return seq_.rbegin ();
      }

      /// Bottom of the stack.
      const_iterator
      end () const YY_NOEXCEPT
      {
        return seq_.rend ();
      }

      /// Present a slice of the top of a stack.
      class slice
      {
      public:
        slice (const stack& stack, index_type range)
          : stack_ (stack)
          , range_ (range)
        {}

        const T&
        operator[] (index_type i) const
        {
          return stack_[range_ - i];
        }

      private:
        const stack& stack_;
        index_type range_;
      };

    private:
      stack (const stack&);
      stack& operator= (const stack&);
      /// The wrapped container.
      S seq_;
    };


    /// Stack type.
    typedef stack<stack_symbol_type> stack_type;

    /// The stack.
    stack_type yystack_;

    /// Push a new state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param sym  the symbol
    /// \warning the contents of \a s.value is stolen.
    void yypush_ (const char* m, YY_MOVE_REF (stack_symbol_type) sym);

    /// Push a new look ahead token on the state on the stack.
    /// \param m    a debug message to display
    ///             if null, no trace is output.
    /// \param s    the state
    /// \param sym  the symbol (for its value and location).
    /// \warning the contents of \a sym.value is stolen.
    void yypush_ (const char* m, state_type s, YY_MOVE_REF (symbol_type) sym);

    /// Pop \a n symbols from the stack.
    void yypop_ (int n = 1);

    /// Some specific tokens.
    static const token_number_type yy_error_token_ = 1;
    static const token_number_type yy_undef_token_ = 2;

    /// Constants.
    enum
    {
      yyeof_ = 0,
      yylast_ = 56428,     ///< Last index in yytable_.
      yynnts_ = 602,  ///< Number of nonterminal symbols.
      yyfinal_ = 616, ///< Termination state number.
      yyntokens_ = 767  ///< Number of tokens.
    };


    // User arguments.
    class Driver& driver;
  };


} // aries_parser
#line 15699 "parser.hh"





#endif // !YY_ARIES_PARSER_PARSER_HH_INCLUDED
