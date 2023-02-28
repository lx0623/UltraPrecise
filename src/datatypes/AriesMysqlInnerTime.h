//
// Created by david shen on 2019-07-18.
//

#pragma once

#include <string>
#include "AriesDefinition.h"
#include "AriesIntervalTime.hxx"
#include "AriesTime.hxx"
#include "AriesDate.hxx"

#define set_if_bigger(a,b)  do { if ((a) < (b)) (a)=(b); } while(0)
#define test_all_bits(a,b) (((a) & (b)) == (b))

/*
Deprecated workaround for false-positive uninitialized variables
warnings. Those should be silenced using tool-specific heuristics.

Enabled by default for g++ due to the bug referenced below.
*/
#if defined(_lint) || defined(FORCE_INIT_OF_VARS) || \
    (defined(__GNUC__) && defined(__cplusplus))
#define LINT_INIT(var) var= 0
#else
#define LINT_INIT(var)
#endif

/*
  Portable time_t replacement.
  Should be signed and hold seconds for 1902 -- 2038-01-19 range
  i.e at least a 32bit variable

  Using the system built in time_t is not an option as
  we rely on the above requirements in the time functions
*/
typedef int32_t my_time_t;

#define INT_MAX32       0x7FFFFFFFL
#define UINT_MAX32      0xFFFFFFFFL

#ifndef UINT_MAX
#define UINT_MAX  UINT_MAX32
#endif

/* Time handling defaults */
#define TIMESTAMP_MAX_YEAR 2038
#define TIMESTAMP_MIN_YEAR (1900 + YY_PART_YEAR - 1)
#define TIMESTAMP_MAX_VALUE INT_MAX32
#define TIMESTAMP_MIN_VALUE 1

/* two-digit years < this are 20..; >= this are 19.. */
#define YY_PART_YEAR	   70

/*
  check for valid times only if the range of time_t is greater than
  the range of my_time_t
*/
#if SIZEOF_TIME_T > 4 || defined(TIME_T_UNSIGNED)
# define IS_TIME_T_VALID_FOR_TIMESTAMP(x) \
    ((x) <= TIMESTAMP_MAX_VALUE && \
     (x) >= TIMESTAMP_MIN_VALUE)
#else
# define IS_TIME_T_VALID_FOR_TIMESTAMP(x) \
    ((x) >= TIMESTAMP_MIN_VALUE)
#endif


/*
   Suppress uninitialized variable warning without generating code.

   The _cplusplus is a temporary workaround for C++ code pending a fix
   for a g++ bug (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=34772).
*/
#if defined(_lint) || defined(FORCE_INIT_OF_VARS) || \
    defined(__cplusplus) || !defined(__GNUC__)
#define UNINIT_VAR(x) x= 0
#else
/* GCC specific self-initialization which inhibits the warning. */
#define UNINIT_VAR(x) x= x
#endif

#ifndef LL
#ifdef HAVE_LONG_LONG
#define LL(A) A ## LL
#else
#define LL(A) A ## L
#endif
#endif

#ifndef ULL
#ifdef HAVE_LONG_LONG
#define ULL(A) A ## ULL
#else
#define ULL(A) A ## UL
#endif
#endif

/* Flags to str_to_datetime */
#define TIME_FUZZY_DATE		1
#define TIME_DATETIME_ONLY	2
/* Must be same as MODE_NO_ZERO_IN_DATE */
#define TIME_NO_ZERO_IN_DATE    (65536L*2*2*2*2*2*2*2)
/* Must be same as MODE_NO_ZERO_DATE */
#define TIME_NO_ZERO_DATE	(TIME_NO_ZERO_IN_DATE*2)
#define TIME_INVALID_DATES	(TIME_NO_ZERO_DATE*2)


#define MYSQL_TIME_WARN_TRUNCATED    1
#define MYSQL_TIME_WARN_OUT_OF_RANGE 2

#define TIME_MAX_VALUE (TIME_MAX_HOUR*10000 + TIME_MAX_MINUTE*100 + \
                        TIME_MAX_SECOND)
#define TIME_MAX_VALUE_SECONDS (TIME_MAX_HOUR * 3600L + \
                                TIME_MAX_MINUTE * 60L + TIME_MAX_SECOND)

typedef char ad_bool; /* Small bool */

//typedef uint8_t uchar;
//typedef uint32_t ulong;
//typedef uint64_t ulonglong;
//typedef int64_t longlong;

BEGIN_ARIES_ACC_NAMESPACE

    extern uint8_t days_in_month[];
    extern uint64_t log_10_int[];
    /*
      Time declarations shared between the server and client API:
      you should not add anything to this header unless it's used
      (and hence should be visible) in mysql.h.
      If you're looking for a place to add new time-related declaration,
      it's most likely my_time.h. See also "C API Handling of AriesDate
      and Time Values" chapter in documentation.
    */

    enum enum_mysql_timestamp_type {
        MYSQL_TIMESTAMP_NONE = -2, MYSQL_TIMESTAMP_ERROR = -1,
        MYSQL_TIMESTAMP_DATE = 0, MYSQL_TIMESTAMP_DATETIME = 1, MYSQL_TIMESTAMP_TIME = 2
    };

    /*
      Structure which is used to represent datetime values inside MySQL.

      We assume that values in this structure are normalized, i.e. year <= 9999,
      month <= 12, day <= 31, hour <= 23, hour <= 59, hour <= 59. Many functions
      in server such as my_system_gmt_sec() or make_time() family of functions
      rely on this (actually now usage of make_*() family relies on a bit weaker
      restriction). Also functions that produce MYSQL_TIME as result ensure this.
      There is one exception to this rule though if this structure holds time
      value (time_type == MYSQL_TIMESTAMP_TIME) days and hour member can hold
      bigger values.
    */
    struct ARIES_PACKED AriesMysqlInnerTime {
        uint16_t year;
        uint8_t month;
        uint8_t day;
        uint16_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t second_part;
        ad_bool neg;
        enum enum_mysql_timestamp_type time_type;
    };

    typedef AriesMysqlInnerTime MYSQL_TIME;

    //transfer string to datetime
    enum enum_mysql_timestamp_type
    str_to_datetime(const char *str, uint32_t length, MYSQL_TIME *l_time,
                    uint64_t flags, int *was_cut);

    my_time_t my_system_gmt_sec(const MYSQL_TIME *t_src, int32_t *my_timezone, ad_bool *in_dst_time_gap);

    int check_time_range(MYSQL_TIME *my_time, int *warning);
    ad_bool validate_timestamp_range(const MYSQL_TIME *t);

//    int32_t calc_daynr(uint32_t year, uint32_t month, uint32_t day);
//    uint32_t calc_days_in_year(uint32_t year);
    uint32_t year_2000_handling(uint32_t year);

    ad_bool check_date(const MYSQL_TIME *ltime, ad_bool not_zero_date,
                       uint64_t flags, int *was_cut);
    int64_t number_to_datetime(int64_t nr, MYSQL_TIME *time_res,
                                uint64_t flags, int *was_cut);
    uint64_t TIME_to_ulonglong_datetime(const MYSQL_TIME *);
    uint64_t TIME_to_ulonglong_date(const MYSQL_TIME *);
    uint64_t TIME_to_ulonglong_time(const MYSQL_TIME *);
    uint64_t TIME_to_ulonglong(const MYSQL_TIME *);

    ad_bool str_to_time(const char *str,uint32_t length, MYSQL_TIME *l_time,
                        int *warning);

    int check_time_range(struct st_mysql_time *, int *warning);

    my_time_t
    my_system_gmt_sec(const MYSQL_TIME *t, int32_t *my_timezone,
                      ad_bool *in_dst_time_gap);

    void set_zero_time(MYSQL_TIME *tm, enum enum_mysql_timestamp_type time_type);

/*
  Required buffer length for my_time_to_str, my_date_to_str,
  my_datetime_to_str and TIME_to_string functions. Note, that the
  caller is still responsible to check that given TIME structure
  has values in valid ranges, otherwise size of the buffer could
  be not enough. We also rely on the fact that even wrong values
  sent using binary protocol fit in this buffer.
*/
#define MAX_DATE_STRING_REP_LENGTH 30

    int my_time_to_str(const MYSQL_TIME *l_time, char *to);
    int my_date_to_str(const MYSQL_TIME *l_time, char *to);
    int my_date_to_str(const AriesDate *l_time, char *to);
    int my_datetime_to_str(const MYSQL_TIME *l_time, char *to);
    int my_TIME_to_str(const MYSQL_TIME *l_time, char *to);

END_ARIES_ACC_NAMESPACE