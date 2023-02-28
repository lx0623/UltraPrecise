#include <regex>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <server/mysql/include/derror.h>
#include <server/mysql/include/mysqld_error.h>
#include <server/mysql/include/my_byteorder.h>
#include <server/mysql/include/sql_class.h>
#include <datatypes/AriesDatetimeTrans.h>

#include "CommonBiaodashi.h"
#include "SelectStructure.h"
#include "AriesAssert.h"
#include "utils/string_util.h"
#include "datatypes/AriesMysqlInnerTime.h"
#include "datatypes/AriesTimeCalc.hxx"
#include "BiaodashiAuxProcessor.h"

namespace aries
{

int CommonBiaodashi::NumericTypeOffsetInt8 = 1 << 0;
int CommonBiaodashi::NumericTypeOffsetInt16 = 1 << 2;
int CommonBiaodashi::NumericTypeOffsetInt32 = 1 << 3;
int CommonBiaodashi::NumericTypeOffsetInt64 = 1 << 4;
int CommonBiaodashi::NumericTypeOffsetDecimal = 1 << 5;
int CommonBiaodashi::NumericTypeOffsetFloat = 1 << 6;
int CommonBiaodashi::NumericTypeOffsetDouble = 1 << 7;

std::string get_name_of_value_type(BiaodashiValueType type) {
    switch (type) {
        case BiaodashiValueType::TINY_INT: return "TINY_INT";
        case BiaodashiValueType::SMALL_INT: return "SMALL_INT";
        case BiaodashiValueType::INT: return "INT";
        case BiaodashiValueType::LONG_INT: return "LONG_INT";
        case BiaodashiValueType::DECIMAL: return "DECIMAL";
        case BiaodashiValueType::FLOAT: return "FLOAT";
        case BiaodashiValueType::DOUBLE: return "DOUBLE";
        case BiaodashiValueType::TEXT: return "TEXT";
        case BiaodashiValueType::BOOL: return "BOOL";
        case BiaodashiValueType::DATE: return "DATE";
        case BiaodashiValueType::TIME: return "TIME";
        case BiaodashiValueType::DATE_TIME: return "DATE_TIME";
        case BiaodashiValueType::TIMESTAMP: return "TIMESTAMP";
        case BiaodashiValueType::UNKNOWN: return "UNKNOWN";
        case BiaodashiValueType::YEAR: return "YEAR";
        case BiaodashiValueType::LIST: return "LIST";
        case BiaodashiValueType::BINARY: return "BINARY";
        case BiaodashiValueType::VARBINARY: return "VARBINARY";
        default: ARIES_ASSERT(0, "unknown value type: " + std::to_string(static_cast<int>(type)));
    }
}

/**
 * This is from mysql's source file: field.cc
 * field_types_merge_rules
 */
static BiaodashiValueType column_types_merge_rules[17][17] = {
    // TINY_INT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TINY_INT,   BiaodashiValueType::SMALL_INT,
        // INT                          // LONG_INT
        BiaodashiValueType::INT,        BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::FLOAT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TINY_INT,   BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TINY_INT,   BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // SMALL_INT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::SMALL_INT,  BiaodashiValueType::SMALL_INT,
        // INT                          // LONG_INT
        BiaodashiValueType::INT,        BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::FLOAT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::SMALL_INT,  BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::SMALL_INT,  BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// INT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::INT,        BiaodashiValueType::INT,
        // INT                          // LONG_INT
        BiaodashiValueType::INT,        BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::DOUBLE,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::INT,        BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::INT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // LONG_INT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::LONG_INT,   BiaodashiValueType::LONG_INT,
        // INT                          // LONG_INT
        BiaodashiValueType::LONG_INT,   BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::DOUBLE,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::LONG_INT,   BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::LONG_INT,   BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // DECIMAL
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::DECIMAL,
        // INT                          // LONG_INT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::DECIMAL,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::DOUBLE,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// FLOAT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::FLOAT,      BiaodashiValueType::FLOAT,
        // INT                          // LONG_INT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::DOUBLE,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::DOUBLE,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::FLOAT,      BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::FLOAT,      BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // DOUBLE
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::DOUBLE,
        // INT                          // LONG_INT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::DOUBLE,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::DOUBLE,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// TEXT
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// BOOL
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TINY_INT,   BiaodashiValueType::SMALL_INT,
        // INT                          // LONG_INT
        BiaodashiValueType::INT,        BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::FLOAT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::BOOL,       BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::BOOL,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// DATE
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::DATE,
        // TIME                         // DATETIME
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::DATE_TIME,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // TIME
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::DATE_TIME,
        // TIME                         // DATETIME
        BiaodashiValueType::TIME,       BiaodashiValueType::DATE_TIME,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // DATE_TIME
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::DATE_TIME,
        // TIME                         // DATETIME
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::DATE_TIME,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // TIMESTAMP
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::DATE_TIME,
        // TIME                         // DATETIME
        BiaodashiValueType::DATE_TIME,  BiaodashiValueType::DATE_TIME,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TIMESTAMP,  BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
	// UNKNOWN
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // INT                          // LONG_INT
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // DOUBLE                       // TEXT
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // BOOL                         // DATE
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // TIME                         // DATETIME
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::UNKNOWN,    BiaodashiValueType::UNKNOWN,
        // BINARY
        BiaodashiValueType::UNKNOWN
    },
    // YEAR
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::INT,        BiaodashiValueType::INT,
        // INT                          // LONG_INT
        BiaodashiValueType::INT,        BiaodashiValueType::LONG_INT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::DECIMAL,    BiaodashiValueType::FLOAT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::DOUBLE,     BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::YEAR,       BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::YEAR,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // VARBINARY
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
    // BINARY
    {
        // TINY_INT                     // SMALL_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // INT                          // LONG_INT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DECIMAL                      // FLOAT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // DOUBLE                       // TEXT
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BOOL                         // DATE
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIME                         // DATETIME
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // TIMESTAMP                    // UNKNOWN
        BiaodashiValueType::TEXT,       BiaodashiValueType::UNKNOWN,
        // YEAR                         // VARBINARY
        BiaodashiValueType::TEXT,       BiaodashiValueType::TEXT,
        // BINARY
        BiaodashiValueType::TEXT
    },
};

std::string get_name_of_expr_type(BiaodashiType type) {
    unordered_map<int, string> map = {
            {BiaodashiType::Zhengshu,           "integer"},
            {BiaodashiType::Fudianshu,          "float"},
            {BiaodashiType::Zifuchuan,          "string"},
            {BiaodashiType::Decimal,            "decimal"},

            {BiaodashiType::Biaoshifu,          "identifier"},
            {BiaodashiType::Lie,                "column"},

            {BiaodashiType::Star,               "star"},

            {BiaodashiType::Hanshu,             "function"},
            {BiaodashiType::SQLFunc,            "sqlfunc"},

            {BiaodashiType::Shuzu,              "array"},
            {BiaodashiType::Yunsuan,            "calc"},
            {BiaodashiType::Qiufan,             "not"},
            {BiaodashiType::Bijiao,             "compare"},
            {BiaodashiType::Likeop,             "like"},
            {BiaodashiType::Inop,               "in"},
            {BiaodashiType::NotIn,              "not in"},
            {BiaodashiType::Between,            "between"},
            {BiaodashiType::IsNull,             "is null"},
            {BiaodashiType::IsNotNull,          "is not null"},

            {BiaodashiType::Cunzai,             "exist"},
            {BiaodashiType::Case,               "case"},
            {BiaodashiType::Andor,              "and / or"},

            {BiaodashiType::Kuohao,             "parenthesis"},

            {BiaodashiType::Zhenjia,            "bool"},
            {BiaodashiType::Distinct,           "distinct"},

            {BiaodashiType::Query,              "sub query"},

            {BiaodashiType::IfCondition,        "if"},
            {BiaodashiType::Null,               "null"},
            {BiaodashiType::IntervalExpression, "interval"},

            {BiaodashiType::QuestionMark,       "?"},
            {BiaodashiType::Buffer,             "buffer"},
            {BiaodashiType::ExprList,           "exprlist"}
    };
    auto it = map.find(type);
    if (map.end() == it) {
        ARIES_ASSERT(0, "unknown value type: " + std::to_string(static_cast<int>(type)));
    } else {
        return it->second;
    }
}

CommonBiaodashi::CommonBiaodashi(BiaodashiType arg_type, BiaodashiContent arg_content)
{
    this->type = arg_type;
    this->content = arg_content;
}

bool CommonBiaodashi::IsLiteral() const
{
    switch ( type )
    {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Decimal:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Null:
            return true;
        default:
            return false;
    }
}

void CommonBiaodashi::SetOrigName(std::string origExpString, std::string alias) {
    if (!alias.empty()) {
        origName = alias;
    } else {
        origName = origExpString;
    }
}

int CommonBiaodashi::GetColumnStartingPosition()
{
    return this->column_starting_position;
}

void CommonBiaodashi::SetColumnStartingPosition(int arg)
{
    this->column_starting_position = arg;
}

int CommonBiaodashi::GetColumnEndingPosition() { return this->column_ending_position; }

void CommonBiaodashi::SetColumnEndingPosition(int arg)
{
    this->column_ending_position = arg;
}

BiaodashiValueType CommonBiaodashi::GetValueType() { return this->value_type; }

void CommonBiaodashi::SetValueType(BiaodashiValueType arg_value)
{
    this->value_type = arg_value;
}

int CommonBiaodashi::GetNumericTypeValue() { return numeric_type_value; }

BiaodashiType CommonBiaodashi::GetType() { return this->type; }

BiaodashiContent CommonBiaodashi::GetContent() { return this->content; }

void CommonBiaodashi::SetPreparedStmtLongData(const char *arg_str, ulong arg_length) {
    is_param_set = true;
    is_long_data = true;
    long_data.append(arg_str, arg_length);
}
void CommonBiaodashi::SetPreparedStmtNullParam() {
    is_param_set = true;
    SetType(aries::BiaodashiType::Null);
    SetContent(0);
}
void CommonBiaodashi::SetPreparedStmtLongParam(enum enum_field_types arg_param_type) {
    param_type = arg_param_type;
    is_param_set = true;
    SetContent(long_data);
    SetType(aries::BiaodashiType::Zifuchuan);
    SetValueType(aries::BiaodashiValueType::TEXT);
}
void CommonBiaodashi::ClearPreparedStmtParam() {
    is_param_set = false;
    is_long_data = false;
    long_data.clear();
    content = 0;
    type = aries::BiaodashiType::QuestionMark;
}
/**
  Read the length of the parameter data and return it back to
  the caller.

    Read data length, position the packet to the first byte after it,
    and return the length to the caller.

  @param packet             a pointer to the data
  @param len                remaining packet length

  @return
    Length of data piece.
*/

ulong get_param_length(uchar **packet, ulong len)
{
    uchar *pos= *packet;
    if (len < 1)
        return 0;
    if (*pos < 251)
    {
        (*packet)++;
        return (ulong) *pos;
    }
    if (len < 3)
        return 0;
    if (*pos == 252)
    {
        (*packet)+=3;
        return (ulong) uint2korr(pos+1);
    }
    if (len < 4)
        return 0;
    if (*pos == 253)
    {
        (*packet)+=4;
        return (ulong) uint3korr(pos+1);
    }
    if (len < 5)
        return 0;
    (*packet)+=9; // Must be 254 when here
    /*
      In our client-server protocol all numbers bigger than 2^24
      stored as 8 bytes with uint8korr. Here we always know that
      parameter length is less than 2^4 so don't look at the second
      4 bytes. But still we need to obey the protocol hence 9 in the
      assignment above.
    */
    return (ulong) uint4korr(pos+1);
}
/**
  Check datetime, date, or normalized time (i.e. time without days) range.
  @param ltime   Datetime value.
  @returns
  @retval   FALSE on success
  @retval   TRUE  on error
*/
my_bool check_datetime_range(const MYSQL_TIME *ltime)
{
    /*
      In case of MYSQL_TIMESTAMP_TIME hour value can be up to TIME_MAX_HOUR.
      In case of MYSQL_TIMESTAMP_DATETIME it cannot be bigger than 23.
    */
    return
            ltime->year > 9999U || ltime->month > 12U  || ltime->day > 31U ||
            ltime->minute > 59U || ltime->second > 59U || ltime->second_part > 999999U ||
            (ltime->hour >
             (ltime->time_type == MYSQL_TIMESTAMP_TIME ? TIME_MAX_HOUR : 23U));
}

void CommonBiaodashi::SetPreparedStmtParam(enum enum_field_types arg_param_type,
                                           unsigned char **arg_data,
                                           unsigned char *arg_data_end) {
    uchar* data = *arg_data;
    is_param_set = true;
    param_type = arg_param_type;
    switch (param_type) {
        case MYSQL_TYPE_TINY: {
            int8 value= (int8) *data;
            SetContent(value);
            SetType(aries::BiaodashiType::Zhengshu);
            SetValueType(aries::BiaodashiValueType::TINY_INT);
            (*arg_data) += 1;
            break;
        }
        case MYSQL_TYPE_SHORT: {
            int16 value = 0;
            if (arg_data_end - data >= 2) {
                value= sint2korr(data);
            }
            SetContent(value);
            SetType(aries::BiaodashiType::Zhengshu);
            SetValueType(aries::BiaodashiValueType::SMALL_INT);
            (*arg_data) += 2;
            break;
        }
        case MYSQL_TYPE_LONG: {
            int32 value = 0;
            if (arg_data_end - data >= 4) {
                value= sint4korr(data);
            }
            SetContent(value);
            SetType(aries::BiaodashiType::Zhengshu);
            SetValueType(aries::BiaodashiValueType::INT);
            (*arg_data) += 4;
            break;
        }
        case MYSQL_TYPE_LONGLONG: {
            longlong value = 0;
            if (arg_data_end - data >= 8) {
                value= sint8korr(data);
            }
            SetContent((int)value);
            SetType(aries::BiaodashiType::Zhengshu);
            SetValueType(aries::BiaodashiValueType::LONG_INT);
            (*arg_data) += 8;
            break;
        }
        case MYSQL_TYPE_FLOAT: {
            float value = 0;
            if (arg_data_end - data >= 4) {
                float4get(&value, data);
            }
            SetContent(value);
            SetType(aries::BiaodashiType::Fudianshu);
            SetValueType(aries::BiaodashiValueType::FLOAT);
            (*arg_data) += 4;
            break;
        }
        case MYSQL_TYPE_DOUBLE: {
            double value = 0;
            if (arg_data_end - data >= 8) {
                float8get(&value, data);
            }
            SetContent(value);
            SetType(aries::BiaodashiType::Fudianshu);
            SetValueType(aries::BiaodashiValueType::DOUBLE);
            (*arg_data) += 8;
            break;
        }
        case MYSQL_TYPE_DECIMAL:
        case MYSQL_TYPE_NEWDECIMAL: {
            ulong length= get_param_length(arg_data, arg_data_end - data);
            string decimalStr((const char*)*arg_data, length);
            SetContent(decimalStr);
            SetType(aries::BiaodashiType::Decimal);
            SetValueType(aries::BiaodashiValueType::DECIMAL);
            (*arg_data) += length;
            break;
        }
        case MYSQL_TYPE_TIME: {
            ulong length= get_param_length(arg_data, arg_data_end - data);
            AriesTime ariesTime;

            if (length >= 8)
            {
                uchar *to= *arg_data;
                uint day;

                bool neg = (bool) to[0];
                ariesTime.sign = neg ? -1 : 1;
                day= (uint) sint4korr(to+1);
                ariesTime.hour=   (uint) to[5] + day * 24;
                ariesTime.minute= (uint) to[6];
                ariesTime.second= (uint) to[7];
                ariesTime.second_part= (length > 8) ? (ulong) sint4korr(to+8) : 0;
                if (ariesTime.hour > 838)
                {
                    /* TODO: add warning 'Data truncated' here */
                    LOG(WARNING) << "Time truncated: " << ariesTime.hour;
                    ariesTime.hour= 838;
                    ariesTime.minute= 59;
                    ariesTime.second= 59;
                }
            }
            string timeStr = AriesDatetimeTrans::GetInstance().ToString(ariesTime);
            SetType(aries::BiaodashiType::Zifuchuan);
            SetContent(timeStr);
            SetValueType(aries::BiaodashiValueType::TEXT);
            (*arg_data)+= length;
            break;
        }
        case MYSQL_TYPE_DATE: {
            ulong length= get_param_length(arg_data, arg_data_end - data);
            AriesDate ariesDate;

            if (length >= 4)
            {
                uchar *to= *arg_data;

                ariesDate.year=  (uint) sint2korr(to);
                ariesDate.month=  (uint) to[2];
                ariesDate.day= (uint) to[3];
            }
            string dateStr = AriesDatetimeTrans::GetInstance().ToString(ariesDate);
            SetType(aries::BiaodashiType::Zifuchuan);
            SetContent(dateStr);
            SetValueType(aries::BiaodashiValueType::TEXT);
            (*arg_data)+= length;
            break;
        }
        case MYSQL_TYPE_TIMESTAMP:
        case MYSQL_TYPE_DATETIME: {
            ulong length= get_param_length(arg_data, arg_data_end - data);
            AriesDatetime ariesDatetime;

            if (length >= 4)
            {
                uchar *to= *arg_data;

                // ariesDatetime.neg=    0;
                ariesDatetime.year=   (uint) sint2korr(to);
                ariesDatetime.month=  (uint) to[2];
                ariesDatetime.day=    (uint) to[3];
                if (length > 4)
                {
                    ariesDatetime.hour=   (uint) to[4];
                    ariesDatetime.minute= (uint) to[5];
                    ariesDatetime.second= (uint) to[6];
                }
                else
                    ariesDatetime.hour= ariesDatetime.minute= ariesDatetime.second= 0;

                ariesDatetime.second_part= (length > 7) ? (ulong) sint4korr(to+7) : 0;
            }
            string datetimeStr = AriesDatetimeTrans::GetInstance().ToString(ariesDatetime);
            SetType(aries::BiaodashiType::Zifuchuan);
            SetContent(datetimeStr);
            SetValueType(aries::BiaodashiValueType::TEXT);
            (*arg_data)+= length;
            break;
        }
        case MYSQL_TYPE_TINY_BLOB:
        case MYSQL_TYPE_MEDIUM_BLOB:
        case MYSQL_TYPE_LONG_BLOB:
        case MYSQL_TYPE_BLOB:
            // just binary string
            assert(0);
        default: {
            // TODO: use correct charsets
            ulong len = arg_data_end - data;
            ulong length= get_param_length(arg_data, len);
            if (length > len)
                length= len;
            std::string s((const char*)*arg_data, length);
            SetContent(s);
            (*arg_data) += length;
            SetType(aries::BiaodashiType::Zifuchuan);
            SetValueType(aries::BiaodashiValueType::TEXT);
            break;
        }
    }
}
void CommonBiaodashi::SetPreparedStmtParam(const std::shared_ptr<user_var_entry>& userVarEntryPtr) {
    auto ariesExpr = userVarEntryPtr->getExpr();
    is_param_set = true;
    switch (ariesExpr->GetType()) {
        case AriesExprType::INTEGER: {
            SetType(aries::BiaodashiType::Zhengshu);
            if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT64) {
                auto value = boost::get<int64_t>(ariesExpr->GetContent());
                SetValueType(aries::BiaodashiValueType::LONG_INT);
                SetContent(value);
                numeric_type_value = NumericTypeOffsetInt64;
            } else if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT8) {
                auto value = boost::get<int8_t>(ariesExpr->GetContent());
                SetValueType(aries::BiaodashiValueType::INT);
                SetContent(value);
                numeric_type_value = NumericTypeOffsetInt32;
            } else if (ariesExpr->GetValueType().DataType.ValueType == AriesValueType::INT16) {
                auto value = boost::get<int16_t>(ariesExpr->GetContent());
                SetValueType(aries::BiaodashiValueType::INT);
                SetContent(value);
                numeric_type_value = NumericTypeOffsetInt32;
            } else {
                auto value = boost::get<int>(ariesExpr->GetContent());
                SetValueType(aries::BiaodashiValueType::INT);
                SetContent(value);
                numeric_type_value = NumericTypeOffsetInt32;
            }
            break;
        }
        case AriesExprType::FLOATING: {
            double value = boost::get<double>(ariesExpr->GetContent());
            SetType(aries::BiaodashiType::Fudianshu);
            SetValueType(aries::BiaodashiValueType::FLOAT);
            SetContent(value);
            numeric_type_value = NumericTypeOffsetFloat;
            break;
        }
        case AriesExprType::DECIMAL: {
            auto decimal_content = boost::get<aries_acc::Decimal>(ariesExpr->GetContent());
            char tmp_buffer[64] = {0};
            decimal_content.GetDecimal(tmp_buffer);
            SetType(aries::BiaodashiType::Decimal);
            SetValueType(aries::BiaodashiValueType::DECIMAL);
            SetContent(tmp_buffer);
            numeric_type_value = NumericTypeOffsetDecimal;
            break;
        }
        case AriesExprType::STRING: {
            SetType(BiaodashiType::Zifuchuan);
            SetValueType(aries::BiaodashiValueType::TEXT);
            string content_string = boost::get<string>(ariesExpr->GetContent());
            SetContent(content_string);
            length = content_string.size();
            break;
        }
        case AriesExprType::TRUE_FALSE: {
            SetType(BiaodashiType::Zhenjia);
            SetValueType(aries::BiaodashiValueType::BOOL);
            SetContent(boost::get<bool>(ariesExpr->GetContent()));
            break;
        }
        case AriesExprType::NULL_VALUE: {
            SetType(BiaodashiType::Null);
            SetValueType(aries::BiaodashiValueType::UNKNOWN);
            SetContent(0);
            break;
        }
        default: {
            assert(0);
            break;
        }
    }
}

size_t CommonBiaodashi::GetChildrenCount() { return this->children.size(); }

BiaodashiPointer CommonBiaodashi::GetChildByIndex(size_t arg_index)
{
    return this->children[arg_index];
}

void CommonBiaodashi::SetExprContext(ExprContextPointer arg)
{
    this->expr_context = arg;
}

ExprContextPointer CommonBiaodashi::GetExprContext() { return this->expr_context; }

void CommonBiaodashi::AddChild(BiaodashiPointer arg_child)
{

    if ( arg_child )
        arg_child->SetParent( this );
    this->children.push_back((arg_child));

    if (arg_child && ((CommonBiaodashi*)arg_child.get())->GetContainSubquery()) {
        contain_subquery = true;
    }
}

/*This function is used to judge whether the whole xpr or one part is in groupby list --
 * then even a column is naked, it is safe*/
bool CommonBiaodashi::helpNakedColumnInExpr(int arg_column_index,
                                            ExprContextPointer arg_expr_context)
{

    if (this->checkExprinGroupbyList(arg_expr_context->GetQueryContext()) == true)
        return true;

    switch (this->type)
    {
    case BiaodashiType::Hanshu:
    case BiaodashiType::SQLFunc:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Kuohao:

        for (size_t i = 0; i < this->children.size(); i++)
        {

            if (this->children[i] == nullptr)
                return false;

            CommonBiaodashi *child_pointer =
                (CommonBiaodashi *)(this->children[i].get());

            if (arg_column_index >= child_pointer->GetColumnStartingPosition() &&
                arg_column_index < child_pointer->GetColumnEndingPosition())
            {
                /*the naked column belongs to this child*/
                return child_pointer->helpNakedColumnInExpr(arg_column_index,
                                                            arg_expr_context);
            }
        }

        break;
    /**
     * If it's a column(alias) in order-by part,
     * try to find it as alias in the select part.
     * If it do exist in select part and is one aggregation function indeed, return true
     */
    case BiaodashiType::Lie: {
        auto column = boost::get<ColumnShellPointer>(content);
        auto query = (SelectStructure*) (arg_expr_context->GetQueryContext()->GetSelectStructure().get());
        auto& select_part = query->GetOriginalSelectPart();
        for (size_t i = 0; i < select_part->GetSelectItemCount(); i ++) {
            auto alias = select_part->GetSelectAlias(i);
            auto expr = (CommonBiaodashi*) (select_part->GetSelectExpr(i).get());

            if (alias && *alias == column->GetColumnName()) {
                // It should not be a real column but alias
                arg_expr_context->GetQueryContext()->unsolved_ID_array.emplace_back(column);
                if (expr->type == BiaodashiType::Hanshu) {
                    auto function_name = boost::get<std::string>(expr->content);
                    SQLFunction function(function_name);
                    return function.GetIsAggFunc();
                }
                break;
            }
        }
        break;
    }
    default:
        return false;
    }

    return false;
}

bool CommonBiaodashi::__compareTwoExpr4Types(CommonBiaodashi *left,
                                             CommonBiaodashi *right)
{
    assert(left && right);

    if (BiaodashiType::QuestionMark == left->type || BiaodashiType::QuestionMark == right->type) {
        return true;
    }
    if (left->value_type == BiaodashiValueType::UNKNOWN || right->value_type == BiaodashiValueType::UNKNOWN) {
        return false;
    }

    if (left->value_type == right->value_type) {
        return true;
    }

    switch (left->type) {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Decimal:
            return true;
        default: break;
    }

    switch (right->type) {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Decimal:
            return true;
        default: break;
    }

    if (left->value_type == BiaodashiValueType::TEXT || right->value_type == BiaodashiValueType::TEXT) {
        return true;
    }

    switch (left->value_type) {
        case BiaodashiValueType::TINY_INT:
        case BiaodashiValueType::SMALL_INT:
        case BiaodashiValueType::INT:
        case BiaodashiValueType::LONG_INT:
        case BiaodashiValueType::DECIMAL:
        case BiaodashiValueType::FLOAT:
        case BiaodashiValueType::DOUBLE:
        case BiaodashiValueType::BOOL: {
            switch (right->value_type) {
                case BiaodashiValueType::TINY_INT:
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::INT:
                case BiaodashiValueType::LONG_INT:
                case BiaodashiValueType::DECIMAL:
                case BiaodashiValueType::FLOAT:
                case BiaodashiValueType::DOUBLE:
                case BiaodashiValueType::BOOL:
                    return true;
                default: return false;
            }
        }
        case BiaodashiValueType::DATE:
        case BiaodashiValueType::DATE_TIME:
        case BiaodashiValueType::TIME:
        case BiaodashiValueType::TIMESTAMP:
        case BiaodashiValueType::YEAR: {
            switch (right->value_type) {
                case BiaodashiValueType::DATE:
                case BiaodashiValueType::DATE_TIME:
                case BiaodashiValueType::TIME:
                case BiaodashiValueType::TIMESTAMP:
                case BiaodashiValueType::YEAR:
                    return true;
                default: return false;
            }
        }
        default:
            ARIES_ASSERT(0, "unhandled type: " + get_name_of_value_type(left->value_type));

    }
}

void CommonBiaodashi::CheckStarExpr(ExprContextPointer arg_expr_context)
{
    // todo
    this->value_type = BiaodashiValueType::UNKNOWN;
}

/* convert biaoshifu's content from string to ColumnShell */
void CommonBiaodashi::convertBiaoshifu()
{

    std::string table_name;
    std::string column_name;

    assert(this->content.which() == 8);
    SQLIdentPtr ident = boost::get<SQLIdentPtr>(this->content);
    ColumnShellPointer column_shell = std::make_shared<ColumnShell>(ident->table, ident->id);

    this->type = BiaodashiType::Lie;
    this->content = column_shell;
}

bool CommonBiaodashi::findColumnInTableArray(
    ColumnShellPointer arg_column, const std::vector<BasicRelPointer>& arg_table_array
    /* int arg_level */ )
{
    bool dup_error = false;
    bool column_found = false;
    bool get_error = false;

    assert(arg_column != nullptr);
    std::string table_name = arg_column->GetTableName();
    std::string column_name = arg_column->GetColumnName();
    BasicRelPointer candidateTable;
    for (size_t i = 0; i != arg_table_array.size(); i++)
    {

        /* a table is a basicrel*/
        BasicRelPointer a_table = arg_table_array[i];
        RelationStructurePointer a_table_structure = a_table->GetRelationStructure();

        /*given table name, we have to check that table only*/
        if (!(table_name.empty()))
        {
            if (table_name == a_table_structure->GetName())
            {
                arg_column->SetTable(a_table);

                /*debug*/
                //		    LOG(INFO) << "\n we are in CommonBiaodashi: we found a
                // column: we try to print the BasicRel: " << a_table->ToString() <<
                // "\n";

                // arg_column->SetQueryLevel(arg_level);
                /*ok, same table, now we locate the column*/

                ColumnStructurePointer the_column_structure =
                    a_table_structure->FindColumn(column_name);

                if (the_column_structure == nullptr)
                {
                    get_error = true;
                    break;
                }
                else
                {

                    int location_in_table =
                        a_table_structure->LocateColumn(column_name);
                    arg_column->SetLocationInTable(location_in_table);
                    arg_column->SetColumnStructure(the_column_structure);

                    column_found = true;

                    //					/* 5-18-2017 add the column into BasicRel's
                    // usage list*/ 					var found_col = false;
                    //
                    //					for (var col_index = 0; col_index <
                    // a_table.column_usage_indicator_array.length; col_index++) {
                    // if (a_table.column_usage_indicator_array[col_index] ==
                    // column_location) { 							found_col = true;
                    // break;
                    //						}
                    //					}
                    //					if (found_col == false) {
                    //						a_table.column_usage_indicator_array.push_back(column_location);
                    //					}

                    break;
                }
            }
            else if( table_name == a_table->GetID() )
            {
                candidateTable = a_table;
            }
        }
        /*we have to check all tables -- also need to check ambiguous columns*/
        else
        {

            ColumnStructurePointer the_column_structure =
                a_table_structure->FindColumn(column_name);

            if (the_column_structure != nullptr)
            {
                if (column_found == false)
                {
                    column_found = true;
                    arg_column->SetTableName(a_table_structure->GetName());
                    arg_column->SetTable(a_table);
                    /*debug*/
                    //			LOG(INFO) << "\n we are in CommonBiaodashi: we found a
                    // column: we try to print the BasicRel: " << a_table->ToString() <<
                    //"\n";

                    //			LOG(INFO) << "\n we are in CommonBiaodashi: we found a
                    // column: we try to print the BasicRel->NODE: " <<
                    // a_table->GetMyRelNode()->ToString(0) << "\n";

                    int location_in_table =
                        a_table_structure->LocateColumn(column_name);
                    arg_column->SetLocationInTable(location_in_table);
                    arg_column->SetColumnStructure(the_column_structure);
                    ///* 5-18-2017 add the column into BasicRel's usage list*/
                    // a_table.column_usage_indicator_array.push_back(column_location);
                }
                else
                {
                    dup_error = true;
                }
            }
        }
    }

    // if( !column_found && candidateTable )
    // {
    //     auto candidate_table_structure = candidateTable->GetRelationStructure();
    //     arg_column->SetTable( candidateTable );
    //     arg_column->SetTableName( candidate_table_structure->GetName() );
        
    //     ColumnStructurePointer the_column_structure = candidate_table_structure->FindColumn( column_name );

    //     if( the_column_structure )
    //     {
    //         int location_in_table = candidate_table_structure->LocateColumn( column_name );
    //         arg_column->SetLocationInTable( location_in_table );
    //         arg_column->SetColumnStructure( the_column_structure );
    //         column_found = true;
    //     }
    //     else
    //         get_error = true;
    // }

    if (get_error)
    {
        ARIES_EXCEPTION( ER_BAD_FIELD_ERROR,
                         column_name.data(),
                         table_name.data());
    }

    if (dup_error)
    {
        // ERROR 1052 (23000): Column 'i1' in field list is ambiguous
        // ARIES_EXCEPTION( ER_NON_UNIQ_ERROR, column_name.data(), " field list" );
        auto query = std::dynamic_pointer_cast<SelectStructure>(expr_context->GetQueryContext()->GetSelectStructure());
        query->AddAmbiguousColumn(arg_column);
    }

    return column_found;
}

bool CommonBiaodashi::__compareTwoExprs(CommonBiaodashi *left, CommonBiaodashi *right)
{

    if (left == NULL && right == NULL)
        return true;

    if ((left == NULL && right != NULL) || (left != NULL && right == NULL))
        return false;

    /*compare type!*/
    if (left->GetType() != right->GetType())
        return false;

    /*compare content*/
    bool blv;
    bool brv;
    int64_t ilv;
    int64_t irv;
    double dlv;
    double drv;
    std::string slv;
    std::string srv;
    SQLIdentPtr identlv, identrv;
    ColumnShellPointer clv;
    ColumnShellPointer crv;
    SQLFunctionPointer sfplv;
    SQLFunctionPointer sfprv;

    bool ret = false;
    switch (left->GetType())
    {
    case BiaodashiType::Zhenjia:
        blv = boost::get<bool>(left->content);
        brv = boost::get<bool>(right->content);
        ret = (blv == brv);
        break;

    case BiaodashiType::Zhengshu:
        if (left->value_type == BiaodashiValueType::LONG_INT) {
            ilv = boost::get<int64_t>(left->content);
        } else {
            ilv = boost::get<int>(left->content);
        }

        if (right->value_type == BiaodashiValueType::LONG_INT) {
            irv = boost::get<int64_t>(right->content);
        } else {
            irv = boost::get<int>(right->content);
        }
        ret = (ilv == irv);
        break;
    case BiaodashiType::Shuzu:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::IsNotNull:
    case BiaodashiType::IsNull:
    case BiaodashiType::Between:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Case:
    case BiaodashiType::Kuohao:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Andor:
        ilv = boost::get<int>(left->content);
        irv = boost::get<int>(right->content);
        ret = (ilv == irv);
        break;

    case BiaodashiType::Fudianshu:
        dlv = boost::get<double>(left->content);
        drv = boost::get<double>(right->content);

        /*NOTE: You cannot compare just two float values! But for this code, we are safe
         * to compare as string!*/
        slv = std::to_string(dlv);
        srv = std::to_string(drv);
        ret = (slv == srv);
        break;

    case BiaodashiType::Biaoshifu: {
        identlv = boost::get<SQLIdentPtr>(left->content);
        identrv = boost::get<SQLIdentPtr>(right->content);
        slv = identlv->ToString();
        srv = identrv->ToString();
        ret = (slv == srv);
        break;
    }
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Star:
    case BiaodashiType::Hanshu:
        slv = boost::get<std::string>(left->content);
        srv = boost::get<std::string>(right->content);
        ret = (slv == srv);
        break;

    case BiaodashiType::Query:
        ret = false;
        break;

    case BiaodashiType::Lie:
        clv = boost::get<ColumnShellPointer>(left->content);
        crv = boost::get<ColumnShellPointer>(right->content);
        assert(clv != nullptr && crv != nullptr);
        ret = (clv->GetTableName() == crv->GetTableName() &&
               clv->GetColumnName() == crv->GetColumnName());

        if (!ret) {
            ret = left->name == crv->GetColumnName();
        }
        break;

    case BiaodashiType::SQLFunc:
        sfplv = boost::get<SQLFunctionPointer>(left->content);
        sfprv = boost::get<SQLFunctionPointer>(right->content);
        assert(sfplv != nullptr && sfprv != nullptr);
        ret = (sfplv->GetName() == sfprv->GetName());
        break;

    case BiaodashiType::IfCondition:
    case BiaodashiType::ExprList:
        ret = true;
        break;
    default:

        LOG(INFO) << "CommonBiaodashi::__compareTwoExprs: default";
        LOG(INFO) << ", left type: " << static_cast<int>(left->GetType());
        LOG(INFO) << ", right type: " << static_cast<int>(right->GetType()) << std::endl;
        ret = false;
    }

    if (ret == false)
        return false;

    /*compare children*/

    if (left->GetChildrenCount() != right->GetChildrenCount())
        return false;

    for (size_t i = 0; i < left->GetChildrenCount(); i++)
    {
        if (left->GetChildByIndex(i) == nullptr && right->GetChildByIndex(i) == nullptr)
            continue;

        if ((left->GetChildByIndex(i) == nullptr &&
             right->GetChildByIndex(i) != nullptr) ||
            (left->GetChildByIndex(i) != nullptr &&
             right->GetChildByIndex(i) == nullptr))
            return false;

        if (__compareTwoExprs((CommonBiaodashi *)(left->GetChildByIndex(i).get()),
                              (CommonBiaodashi *)(right->GetChildByIndex(i).get())) ==
            false)
            return false;
    }

    return true;
}

bool CommonBiaodashi::CompareMyselfToAnotherExpr(BiaodashiPointer arg_expr)
{
    return __compareTwoExprs(this, (CommonBiaodashi *)arg_expr.get());
}

bool CommonBiaodashi::checkColumninGroupbyList(QueryContextPointer arg_query_context)
{
    return this->checkExprinGroupbyList(arg_query_context);
}

bool CommonBiaodashi::findColumnInLoop(ExprContextPointer arg_expr_context)
{
    bool found = false;
    QueryContextPointer current_query_context = arg_expr_context->GetQueryContext();
    // int current_level = 0;

    ExprContextPointer current_expr_context = arg_expr_context;

    QueryContextPointer subquery_context = nullptr;

    /*we first check current query, then up-query, until top*/
    while (true)
    {

        // found = this.findColumnInQueryContext(arg_biaoshifu, current_query_context,
        // arg_expr_context, current_level, subquery_context);
        found =
            this->findColumnInQueryContext(current_query_context, current_expr_context,
                                           /* current_level, */ subquery_context);

        if (!found)
        {
            if (current_query_context->type != QueryContextType::TheTopQuery)
            {
                subquery_context = current_query_context;
                current_query_context = current_query_context->GetParent();

                /*expr_context can be NULL!!!!*/
                if (current_expr_context == nullptr)
                {
                    break;
                }
                current_expr_context = current_expr_context->GetParent();
                // current_level += 1;
            }
            else
            {
                break;
            }
        }
        else
        {
            break;
        }
    }

    return found;
}

/*arg_biaoshifu could be either just a column_name or table_name.column_name*/
void CommonBiaodashi::checkBiaoshifu(ExprContextPointer arg_expr_context, bool expectOnlyConst)
{
    this->convertBiaoshifu();


    // var the_column = arg_biaoshifu.content;
    bool is_alias = false;

    ColumnShellPointer the_column = boost::get<ColumnShellPointer>(this->content);
    if ( expectOnlyConst )
    {
        ARIES_EXCEPTION( ER_BAD_FIELD_ERROR,
                         the_column->GetColumnName().c_str(),
                         "field list" );
    }
    if (arg_expr_context->type == ExprContextType::OrderbyExpr) {
        auto query = std::dynamic_pointer_cast<SelectStructure>(arg_expr_context->GetQueryContext()->GetSelectStructure());
        auto select_part = query->GetSelectPart();

        if (the_column->GetTableName().empty() && select_part) {
            for (size_t i = 0; i < select_part->GetSelectItemCount(); i++) {
                const auto& alias = select_part->GetSelectAlias(i);
                if (alias) {
                    if (the_column->GetColumnName() == *alias) {
                        is_alias = true;
                    }
                }
            }
        }
    }

    if (arg_expr_context->type == ExprContextType::HavingExpr) {
        auto query = std::dynamic_pointer_cast<SelectStructure>(arg_expr_context->GetQueryContext()->GetSelectStructure());
        auto select_part = query->GetSelectPart();

        if (the_column->GetTableName().empty() && select_part) {
            for (size_t i = 0; i < select_part->GetSelectItemCount(); i++) {
                const auto& alias = select_part->GetSelectAlias(i);
                if (alias) {
                    if (the_column->GetColumnName() == *alias) {
                        is_alias = true;
                        auto expr = std::dynamic_pointer_cast<CommonBiaodashi>( select_part->GetSelectExpr( i ) )->Clone();
                        expr->CheckExpr( arg_expr_context, expectOnlyConst );
                        nullable = expr->nullable;
                        value_type = expr->value_type;
                        length = expr->length;
                        switch (value_type)
                        {
                        case BiaodashiValueType::TINY_INT:
                            numeric_type_value = NumericTypeOffsetInt8;
                            break;
                        case BiaodashiValueType::SMALL_INT:
                            numeric_type_value = NumericTypeOffsetInt16;
                            break;
                        case BiaodashiValueType::INT:
                            numeric_type_value = NumericTypeOffsetInt32;
                            break;
                        case BiaodashiValueType::LONG_INT:
                            numeric_type_value = NumericTypeOffsetInt64;
                            break;
                        case BiaodashiValueType::DECIMAL:
                            numeric_type_value = NumericTypeOffsetDecimal;
                            length = the_column->GetPrecision();
                            associated_length = the_column->GetScale();
                            break;
                        case BiaodashiValueType::FLOAT:
                            numeric_type_value = NumericTypeOffsetFloat;
                            break;
                        case BiaodashiValueType::DOUBLE:
                            numeric_type_value = NumericTypeOffsetDouble;
                            break;
                        default:
                            break;
                        }
                        break;
                    }
                }
            }
        }
    }

    bool found = false;
    if (!is_alias) {
        found = this->findColumnInLoop(arg_expr_context);
    }

    if (found)
    {
        // This is from own query

        arg_expr_context->referenced_column_array.push_back(the_column);
        arg_expr_context->referenced_column_agg_status_array.push_back(false);
        arg_expr_context->referenced_column_agg_status_array.emplace_back(false);
        arg_expr_context->referenced_column_agg_status_array.emplace_back(false);
        nullable = the_column->GetColumnStructure()->IsNullable();
        value_type = the_column->GetColumnStructure()->GetValueType();
        length = the_column->GetLength();

        // TODO: support more numeric types
        switch (the_column->GetValueType())
        {
        case BiaodashiValueType::TINY_INT:
            numeric_type_value = NumericTypeOffsetInt8;
            break;
        case BiaodashiValueType::SMALL_INT:
            numeric_type_value = NumericTypeOffsetInt16;
            break;
        case BiaodashiValueType::INT:
            numeric_type_value = NumericTypeOffsetInt32;
            break;
        case BiaodashiValueType::LONG_INT:
            numeric_type_value = NumericTypeOffsetInt64;
            break;
        case BiaodashiValueType::DECIMAL:
            numeric_type_value = NumericTypeOffsetDecimal;
            length = the_column->GetPrecision();
            associated_length = the_column->GetScale();
            break;
        case BiaodashiValueType::FLOAT:
            numeric_type_value = NumericTypeOffsetFloat;
            break;
        case BiaodashiValueType::DOUBLE:
            numeric_type_value = NumericTypeOffsetDouble;
            break;
        default:
            break;
        }
    }
    else
    {
        bool absolutely_wrong = false;

        /*it could be a select alias --- from groupby, orderby*/

        if ((arg_expr_context->type == ExprContextType::GroupbyExpr ||
             arg_expr_context->type == ExprContextType::OrderbyExpr ||
             arg_expr_context->type == ExprContextType::HavingExpr) &&
            arg_expr_context->check_serial >= 1)
        {
            absolutely_wrong = false;
        }
        else
        {
            absolutely_wrong = true;
        }

        if (absolutely_wrong)
        {
            LOG(ERROR) << "Cannot find column: " + this->ToString();
            ARIES_EXCEPTION(ER_BAD_FIELD_ERROR,
                            the_column->GetColumnName().c_str(),
                            (the_column->GetTableName().empty() ? "field list" : the_column->GetTableName().c_str()));
        }
        else
        {
            /*put this ID into the unsolved array*/
            //	    console.log("we now need to put (" +
            // arg_biaoshifu.content.toString() + ") into unsolved_ID_array");
            arg_expr_context->GetQueryContext()->unsolved_ID_array.push_back(the_column);

            // console.log("yes, i push it!!!!!");
            /*how about its type?*/
        }
    }
}

std::string CommonBiaodashi::ContentToString()
{
    std::string ret = "";

    AbstractQueryPointer aqp;

    // LOG(INFO) << "In ContentToString(): " << std::to_string((int)this->type) << "\n";

    ColumnShellPointer colsh;

    SQLFunctionPointer sfp;

    SQLIdentPtr identPtr;

    int op;

    switch (this->type)
    {

    case BiaodashiType::Zhenjia:
        assert(this->content.which() == 0);
        ret = std::to_string(boost::get<bool>(this->content));
        break;

    case BiaodashiType::Zhengshu: {
        if (value_type == BiaodashiValueType::LONG_INT) {
            ret = std::to_string(boost::get<int64_t>(this->content));
        } else {
            ret = std::to_string(boost::get<int>(this->content));
        }
        break;
    }
    case BiaodashiType::Shuzu:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::Between:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Case:
    case BiaodashiType::Kuohao:
    case BiaodashiType::ExprList:
        assert(this->content.which() == 1);
        ret = std::to_string(boost::get<int>(this->content));
        break;

    case BiaodashiType::Bijiao:
        assert(this->content.which() == 1);
        op = boost::get<int>(this->content);

        ret = ShowUtility::GetInstance()->GetTextFromComparisonType(
            static_cast<ComparisonType>(op));
        break;

    case BiaodashiType::Yunsuan:
        assert(this->content.which() == 1);
        op = boost::get<int>(this->content);

        ret =
            ShowUtility::GetInstance()->GetTextFromCalcType(static_cast<CalcType>(op));
        break;

    case BiaodashiType::Andor:
        assert(this->content.which() == 1);
        op = boost::get<int>(this->content);

        ret = ShowUtility::GetInstance()->GetTextFromLogicType(
            static_cast<LogicType>(op));
        break;

    case BiaodashiType::Fudianshu:
        assert(this->content.which() == 2);
        ret = std::to_string(boost::get<double>(this->content));
        break;

    case BiaodashiType::Biaoshifu: {
        assert(this->content.which() == 8);
        identPtr = boost::get<SQLIdentPtr>(this->content);
        ret = identPtr->ToString();
        break;
    }
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Star:
    case BiaodashiType::Hanshu:
        assert(this->content.which() == 3);
        ret = boost::get<std::string>(this->content);
        break;

    case BiaodashiType::Query:
        assert(this->content.which() == 4);
        aqp = boost::get<AbstractQueryPointer>(this->content);
        assert(aqp != nullptr);
        ret = "\nSUBQUERY BEGIN\t" + aqp->ToString() + "\t SUBQUERY END\n";
        break;

    case BiaodashiType::Lie:
        assert(this->content.which() == 5);
        colsh = boost::get<ColumnShellPointer>(this->content);
        assert(colsh != nullptr);
        ret = colsh->ToString();
        break;

    case BiaodashiType::SQLFunc:
        assert(this->content.which() == 6);
        sfp = boost::get<SQLFunctionPointer>(this->content);
        assert(sfp != nullptr);
        ret = sfp->ToString();
        break;
    case BiaodashiType::Decimal:
    {
        assert(content.which() == 3);
        ret = boost::get<std::string>(content);
        break;
    }
    case BiaodashiType::IfCondition:
        ret = "IFCondition";
        break;
    case BiaodashiType::IsNotNull:
        ret = "Is Not Null";
        break;
    case BiaodashiType::IsNull:
        ret = "Is Null";
        break;
    case BiaodashiType::Null:
    {
        ret = "Null";
        break;
    }

    case BiaodashiType::Distinct:
        ret = "Distinct";
        break;
    case BiaodashiType::IntervalExpression:
        ret = boost::get<std::string>(content);
        break;

    case BiaodashiType::QuestionMark:
        ret = "?";
        break;

    case BiaodashiType::Buffer:
        ret = "[Buffer]";
        break;

    default:
        ARIES_ASSERT(0, "unknown expression type: " + std::to_string(static_cast<int>(type)));
        break;
    }

    return ret;
}

std::string CommonBiaodashi::ChildrenToString()
{
    std::string ret = "";
    size_t i = 0;
    for (i = 0; i != this->children.size(); i++)
    {
        //		ret = ret + std::to_string(i) + std::string(":   ");
        ret += this->children[i] ? this->children[i]->ToString() : std::string("NULL");
        ret += std::string(", ");
    }

    if (i > 0)
    {
        ret = ret.substr(0, ret.length() - 2);
    }

    return ret;
}

std::string CommonBiaodashi::ChildrenToString_Skip0()
{
    std::string ret = "";
    size_t i = 1;
    for (i = 1; i != this->children.size(); i++)
    {
        //		ret = ret + std::to_string(i) + std::string(":   ");
        ret += this->children[i] ? this->children[i]->ToString() : std::string("NULL");
        ret += std::string(", ");
    }

    if (i > 0)
    {
        ret = ret.substr(0, ret.length() - 2);
    }

    return ret;
}

std::string CommonBiaodashi::CaseToString()
{
    /*[case_expr, (whenexpr, thenepxr)+, elseexpr]*/

    std::string ret;

    ret = "(CASE ";

    std::string case_expr_str =
        this->children[0] ? this->children[0]->ToString() : std::string("");

    ret += case_expr_str;
    ret += " ";

    size_t tl = this->children.size();
    for (size_t i = 1; i < tl / 2; i++)
    {
        ret += "WHEN ";
        ret += this->children[i] ? this->children[i]->ToString() : std::string("NULL");
        ret += " THEN ";
        ret += this->children[i + 1] ? this->children[i + 1]->ToString()
                                     : std::string("NULL");
        ret += " ";
    }

    if (this->children[tl - 1])
    {
        ret += "ELSE ";
        ret += this->children[tl - 1]->ToString();
    }

    ret += ")";

    return ret;
}

std::string CommonBiaodashi::ToString()
{
    std::string ret = "";

    std::string content_str = this->ContentToString();

    switch (this->type)
    {
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Biaoshifu:
    case BiaodashiType::Lie:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Query:
        ret = content_str;
        break;

    case BiaodashiType::Zifuchuan:
        ret = "\"" + content_str + "\"";
        break;

    case BiaodashiType::Star:
        ret = "(";
        ret += content_str.empty() ? ("*") : (content_str + ".*");
        ret += ")";
        break;
    case BiaodashiType::ExprList:
        ret = "(";
        ret += this->ChildrenToString();
        ret += ")";
        break;

    case BiaodashiType::Hanshu:
    case BiaodashiType::SQLFunc:
        ret = content_str;
        ret += "(";
        ret += this->ChildrenToString();
        ret += ")";
        break;

    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Andor:
        assert(this->children.size() == 2);
        ret = "(";
        ret += this->children[0]->ToString();
        ret += " " + content_str + " ";
        ret += this->children[1]->ToString();
        ret += ")";
        break;

    case BiaodashiType::Shuzu:
        assert(this->children.size() == 2);
        ret = this->children[0]->ToString();
        ret += "[";
        ret += this->children[1]->ToString();
        ret += "]";
        break;

    case BiaodashiType::Qiufan:
        assert(this->children.size() == 1);
        ret = "NOT (" + this->children[0]->ToString() + ")";
        break;

    case BiaodashiType::Likeop:
        assert(this->children.size() == 2);
        ret = this->children[0]->ToString();
        ret += " like ";
        ret += this->children[1]->ToString();
        break;

    case BiaodashiType::Inop:
        ret = this->children[0]->ToString();
        ret += " in (";
        ret += this->ChildrenToString_Skip0();
        ret += ")";
        break;

    case BiaodashiType::NotIn:
        ret = this->children[0]->ToString();
        ret += " not in (";
        ret += this->ChildrenToString_Skip0();
        ret += ")";
        break;

    case BiaodashiType::Between:
        assert(this->children.size() == 3);
        ret = this->children[0]->ToString();
        ret += " between ";
        ret += this->children[1]->ToString();
        ret += " and ";
        ret += this->children[2]->ToString();
        break;

    case BiaodashiType::Cunzai:
        assert(this->children.size() == 1);
        ret = "exists (";
        ret += this->children[0]->ToString();
        ret += ")";
        break;

    case BiaodashiType::Kuohao:
        assert(this->children.size() == 1);
        ret = "(";
        ret += this->children[0]->ToString();
        ret += ")";
        break;

    case BiaodashiType::Case:
        ret = this->CaseToString();
        break;
    case BiaodashiType::Decimal:
        ret = boost::get<std::string>(content);
        break;
    case BiaodashiType::IfCondition:
        ret = "IF ( ";
        ret += children[0]->ToString();
        ret += " ) ";
        ret += children[1]->ToString();
        ret += " ELSE ";
        ret += children[2]->ToString();
        ret += " ENDIF";
        break;
    case BiaodashiType::IsNotNull:
        ret = children[0]->ToString();
        ret += " Is Not Null";
        break;
    case BiaodashiType::IsNull:
        ret = children[0]->ToString();
        ret += " Is Null";
        break;
    case BiaodashiType::Null:
        ret = "Null";
        break;
    case BiaodashiType::Distinct:
        ret = "Distinct";
        break;
    case BiaodashiType::IntervalExpression:
        ret = "Interval " + children[0]->ToString() + " " + boost::get<std::string>(content);
        break;
    case BiaodashiType::QuestionMark:
        ret = "?";
        break;

    case BiaodashiType::Buffer:
        ret = "[Buffer]";
        break;

    default:
        ARIES_ASSERT(0, "unknown expression type: " + std::to_string(static_cast<int>(type)));
        break;
    }

    return ret;
}

void CommonBiaodashi::CheckExprPostWork(ExprContextPointer arg_expr_context)
{
    // std::vector<std::shared_ptr<ColumnShell>> referenced_column_array; /*all columns
    // so far found*/ std::vector<bool> referenced_column_agg_status_array; /*am i in
    // agg*/

    std::vector<std::shared_ptr<ColumnShell>> ca =
        arg_expr_context->referenced_column_array;
    std::vector<bool> sa = arg_expr_context->referenced_column_agg_status_array;

    for (size_t i = 0; i < ca.size(); i++)
    {
        /*we forward all the outer columns to outer query*/
        if (ca[i]->GetAbsoluteLevel() != arg_expr_context->GetQueryContext()->query_level)
        {

            /* I also want to keep the infomation in the table, outer column infomation
             * is related with table.*/
            arg_expr_context->GetQueryContext()->outer_column_array.push_back(ca[i]);
            arg_expr_context->GetQueryContext()->outer_column_agg_status_array.push_back(
                sa[i]);
        }
        else
        {

            /*This is our own column -- we need to additional check for naked column
             * --but only we have a groupby*/

            bool has_groupby = std::dynamic_pointer_cast<SelectStructure>(
                                   arg_expr_context->GetQueryContext()->GetSelectStructure())
                                   ->DoIHaveGroupBy();
            if (has_groupby == true && ca[i]->GetInGroupList() == false &&
                sa[i] == false)
            {
                /*we found a naked column --
                 * But before we report error, we have to check whether the expr or
                 * its sub_expr is in the group list
                 */

                // TODO: return different error for different situations
                // switch (arg_expr_context->type) {
                //     case ExprContextType::SelectExpr: {
                //         if (this->helpNakedColumnInExpr(i, arg_expr_context) != true)
                //         {
                //             const char* format = "Expression #%u of %s is not in GROUP BY clause and contains nonaggregated column '%-.192s' which is not functionally dependent on columns in GROUP BY clause; this is incompatible with sql_mode=only_full_group_by";
                //             // (void) snprintf(ebuff, sizeof(ebuff), "%s %s", ERRMSG_SYNTAX_ERROR, "a column must be used in an aggregate function!");
                //             string errMsg = format_err_msg("%s %s", );
                //             throw AriesFrontendException(ER_WRONG_FIELD_WITH_GROUP, errMsg);
                //         }
                //         break;
                //     }
                //     case ExprContextType::OrderbyExpr: {
                //         break;
                //     }
                //     case ExprContextType::HavingExpr: {
                //         break;
                //     }
                // }

                if (arg_expr_context->type == ExprContextType::SelectExpr ||
                    arg_expr_context->type == ExprContextType::OrderbyExpr ||
                    arg_expr_context->type == ExprContextType::HavingExpr)
                {
                    // group by errors:
                    //#define ER_MIX_OF_GROUP_FUNC_AND_FIELDS 1140
                    // mysql> select name, sum(id) from t1;
                    //ERROR 1140 (42000): In aggregated query without GROUP BY, expression #1 of SELECT list contains nonaggregated column 'db1.t1.name'; this is incompatible with sql_mode=only_full_group_by

                    // #define ER_WRONG_FIELD_WITH_GROUP 1055
                    // mysql> select height, sum(id) from t1 group by name;
                    // ERROR 1055 (42000): Expression #1 of SELECT list is not in GROUP BY clause and contains nonaggregated column 'db1.t1.height' which is not functionally dependent on columns in GROUP BY clause; this is incompatible with sql_mode=only_full_group_by

                    if (this->helpNakedColumnInExpr(i, arg_expr_context) != true)
                    {
                        char ebuff[ERRMSGSIZE];
                        (void) snprintf(ebuff, sizeof(ebuff), "%s a column(%s) must be used in an aggregate function!", ERRMSG_SYNTAX_ERROR, ca[i]->GetColumnName().c_str());
                        string errMsg(ebuff);
                        ARIES_EXCEPTION_SIMPLE(ER_WRONG_FIELD_WITH_GROUP_V2, errMsg);
                    }
                }
            }
        }
    } // for
}

bool CommonBiaodashi::checkExprinGroupbyList(QueryContextPointer arg_query_context)
{
    std::vector<BiaodashiPointer> vbp =
        std::dynamic_pointer_cast<SelectStructure>(arg_query_context->GetSelectStructure())
            ->GetGroupbyList();

    bool ret = false;
    for (size_t i = 0; i < vbp.size(); i++)
    {
        if (this->CompareMyselfToAnotherExpr(vbp[i]) == true)
        {
            ret = true;
            break;
        }

        if (GetName() == vbp[i]->GetName()) {
            return true;
        }
    }
    return ret;
}

bool CommonBiaodashi::findColumnInQueryContext(
    QueryContextPointer arg_query_context,   /* where to search?*/
    ExprContextPointer arg_expr_context,     /* whose context*/
    // int arg_level,                           /* how deep we have been so far*/
    QueryContextPointer arg_subquery_context /* who gives me this job?*/
)
{

    /*get the content -- it must be a columnshell*/
    assert(this->content.which() == 5);
    ColumnShellPointer the_column = boost::get<ColumnShellPointer>(this->content);

    QueryContextPointer current_query_context = arg_query_context;
    /*Step 1: determine whether we should check current query context*/
    /*no matter what current query context, if we search for a From-Subquery, then we
     * should not check current query context*/

    if (arg_subquery_context != nullptr &&
        arg_subquery_context->type == QueryContextType::FromSubQuery)
        return false;

    /*Step 2: determine where I should search*/

    std::vector<BasicRelPointer> my_from_table_array;

    /*An on_expr should check its own from_table_array, otherwise check the global
     * one!*/

    if (arg_expr_context != nullptr &&
        arg_expr_context->type == ExprContextType::JoinOnExpr)
    {
        // SelectStructurePointer* my_select_structure =
        // current_query_context->GetSelectStructure();
        int my_tables_index = arg_expr_context->index;
        my_from_table_array = std::dynamic_pointer_cast<SelectStructure>(
                                  current_query_context->GetSelectStructure())
                                  ->GetFromMapExprTables(my_tables_index);
    }
    else
    {
        my_from_table_array = std::dynamic_pointer_cast<SelectStructure>(
                                  current_query_context->GetSelectStructure())
                                  ->GetFromTableArray();
    }

    /*Step 3: search*/

    bool search_result =
        this->findColumnInTableArray(the_column, my_from_table_array);

    if (search_result == true)
    {
        the_column->SetAbsoluteLevel(current_query_context->query_level);

        /*Are we in its groupby list?*/

        /*03-09-2017: what if we are checking Group By? -- Do we still need to do
         * this?*/
        if (arg_expr_context != nullptr)
        {
            if (arg_expr_context->type == ExprContextType::SelectExpr ||
                arg_expr_context->type == ExprContextType::OrderbyExpr ||
                arg_expr_context->type == ExprContextType::HavingExpr)
            {

                /*othersise, totally unnecessary to check in_group_list! Even it is
                 * false, SO WHAT!?*/
                the_column->SetInGroupList(
                    this->checkColumninGroupbyList(current_query_context));
            }
        }
        this->value_type =
            the_column->GetValueType(); // arg_biaoshifu.content.column_type;
    }

    return search_result;
}

std::string CommonBiaodashi::CheckFunctionName()
{
    assert(this->content.which() == 3);
    std::string func_name = boost::get<std::string>(this->content);

    /*now I am not a Hanshu anymore!*/
    this->type = BiaodashiType::SQLFunc;
    this->content = std::make_shared<SQLFunction>(func_name);

    return func_name;
}

void CommonBiaodashi::SetAggStatus(int arg_starting_pos, int arg_ending_pos,
                                   ExprContextPointer arg_expr_context)
{
    int my_start_position = arg_starting_pos;
    int my_ending_position = arg_ending_pos;

    bool has_set_own_column = false;
    int max_outer_column_level = -1;

    /*Now we can check each column in the scope of this agg function!*/
    bool has_outer_column = false;

    /*step 1: process own columns*/
    for (int pos = my_start_position; pos < my_ending_position; pos++)
    {
        ColumnShellPointer acs = arg_expr_context->referenced_column_array[pos];
        int column_level = acs->GetAbsoluteLevel();
        if (column_level == arg_expr_context->GetQueryContext()->query_level)
        {
            /*This is a column from myself*/
            /*am i in agg already?*/
            if (arg_expr_context->referenced_column_agg_status_array[pos] == true)
            {
                ARIES_EXCEPTION(ER_INVALID_GROUP_FUNC_USE);
            }
            else
            {
                /*if this expr is in where/on/groupby error!*/
                if (arg_expr_context->type == ExprContextType::WhereExpr ||
                    arg_expr_context->type == ExprContextType::JoinOnExpr ||
                    arg_expr_context->type == ExprContextType::GroupbyExpr)
                {

                    // string errMsg("ERROR: aggregate function cannot be "
                    //               "in the WHERE or GROUP BY clause ");
                    ARIES_EXCEPTION(ER_INVALID_GROUP_FUNC_USE);
                }

                arg_expr_context->referenced_column_agg_status_array[pos] = true;
                has_set_own_column = true;
            }
        }
        else
        {
            /*This is a column from an outer query*/
            /*we do not process it in this loop --- but we need to find who is the
             * deepest one*/
            has_outer_column = true;
            if (column_level > max_outer_column_level)
            {
                max_outer_column_level = column_level;
            }
        }
    }

    /*step 2: process outer columns*/
    if (has_outer_column)
    {
        for (int pos = my_start_position; pos < my_ending_position; pos++)
        {

            ColumnShellPointer acs = arg_expr_context->referenced_column_array[pos];
            int column_level = acs->GetAbsoluteLevel();
            if (column_level != arg_expr_context->GetQueryContext()->query_level)
            {
                /*This is a column from an outer query*/
                /*I have been in agg*/
                if (arg_expr_context->referenced_column_agg_status_array[pos] == true)
                {
                    if (!has_set_own_column && column_level == max_outer_column_level)
                    {
                        /*I am the deepest one -- and I have been in agg, then this is a
                         * nested call*/
                        ARIES_EXCEPTION(ER_INVALID_GROUP_FUNC_USE);
                    }
                }
                else
                {
                    /*I have not been in agg -- but this one is for me?*/
                    if (!has_set_own_column && column_level == max_outer_column_level)
                    {
                        /*it is for me*/
                        /*we need to check whether in that outer query, the
                         * corresponding parent expr whether or not in
                         * where/on/groupby*/

                        ExprContextPointer to_be_checked_expr_context =
                            arg_expr_context;
                        while (to_be_checked_expr_context->GetQueryContext()->query_level !=
                               column_level)
                        {
                            to_be_checked_expr_context =
                                to_be_checked_expr_context->GetParent();
                        }

                        if (to_be_checked_expr_context->type ==
                                ExprContextType::WhereExpr ||
                            to_be_checked_expr_context->type ==
                                ExprContextType::JoinOnExpr ||
                            to_be_checked_expr_context->type ==
                                ExprContextType::GroupbyExpr)
                        {
                            ARIES_EXCEPTION(ER_INVALID_GROUP_FUNC_USE);
                        }

                        arg_expr_context->referenced_column_agg_status_array[pos] =
                            true;
                    }
                }
            }
        }
    }
}

int CommonBiaodashi::getStringLength() {

    std::string string_value = "";
    switch (type) {
        case BiaodashiType::Zhengshu: {
            int64_t value = value_type == BiaodashiValueType::LONG_INT ? boost::get<int64_t>(content) : boost::get<int>(content);
            string_value = std::to_string(value);
            break;
        }
        case BiaodashiType::Zifuchuan: {
            string_value = boost::get<std::string>(content);
            break;
        }
        case BiaodashiType::Decimal: {
            string_value = boost::get<std::string>(content);
            break;
        }
        case BiaodashiType::Null: {
            return 0;
        }
        default: break;
    }

    if (!string_value.empty()) {
        return string_value.size();
    }

    int string_length = length;
    switch (value_type) {
        case BiaodashiValueType::DATE: {
            string_length = 10;
            break;
        }
        case BiaodashiValueType::TINY_INT: {
            string_length = 4;
            break;
        }
        case BiaodashiValueType::SMALL_INT: {
            string_length = 6;
            break;
        }
        case BiaodashiValueType::INT: {
            string_length = 11;
            break;
        }
        case BiaodashiValueType::LONG_INT: {
            string_length = 21;
            break;
        }
        case BiaodashiValueType::DECIMAL: {
            string_length = 64;
            break;
        }
        case BiaodashiValueType::FLOAT: {
            string_length = 32;
            break;
        }
        case BiaodashiValueType::DOUBLE: {
            string_length = 64;
            break;
        }
        case BiaodashiValueType::DATE_TIME: {
            string_length = 19;
            break;
        }
        case BiaodashiValueType::TIME: {
            string_length = 8;
            break;
        }
        case BiaodashiValueType::TIMESTAMP: {
            string_length = 19;
            break;
        }
        default: break;
    }

    return string_length;
}

void CommonBiaodashi::handleNullableFunction(AriesSqlFunctionType type) {
    switch (type) {
        case aries_acc::AriesSqlFunctionType::DATE:
        case aries_acc::AriesSqlFunctionType::DATE_ADD:
        case aries_acc::AriesSqlFunctionType::DATE_SUB:
        case aries_acc::AriesSqlFunctionType::DATE_DIFF:
        case aries_acc::AriesSqlFunctionType::TIME_DIFF:
        case aries_acc::AriesSqlFunctionType::MONTH:
        case aries_acc::AriesSqlFunctionType::EXTRACT:
            nullable = true;
            break;
        default: break;
    }
}

/*
mysql> select cast("111.01" as signed);
+--------------------------+
| cast("111.01" as signed) |
+--------------------------+
|                      111 |
+--------------------------+
1 row in set, 1 warning (0.00 sec)

mysql> show warnings;
+---------+------+---------------------------------------------+
| Level   | Code | Message                                     |
+---------+------+---------------------------------------------+
| Warning | 1292 | Truncated incorrect INTEGER value: '111.01' |
+---------+------+---------------------------------------------+
1 row in set (0.00 sec)

mysql> select cast("a111.01" as signed);
+---------------------------+
| cast("a111.01" as signed) |
+---------------------------+
|                         0 |
+---------------------------+
1 row in set, 1 warning (0.00 sec)

mysql> show warnings;
+---------+------+----------------------------------------------+
| Level   | Code | Message                                      |
+---------+------+----------------------------------------------+
| Warning | 1292 | Truncated incorrect INTEGER value: 'a111.01' |
+---------+------+----------------------------------------------+

// temporal to number
mysql> create table dt(f1 date, f2 time, f3 year, f4 datetime, f5 timestamp);
Query OK, 0 rows affected (0.02 sec)

mysql> insert into dt values("2021-01-01", 210000, 2021, "2021-01-01 23:10:34.999999", "2021-01-01 23:10:34.999999");
Query OK, 1 row affected (0.01 sec)

mysql> select * from dt;
+------------+----------+------+---------------------+---------------------+
| f1         | f2       | f3   | f4                  | f5                  |
+------------+----------+------+---------------------+---------------------+
| 2021-01-01 | 21:00:00 | 2021 | 2021-01-01 23:10:35 | 2021-01-01 23:10:35 |
+------------+----------+------+---------------------+---------------------+
1 row in set (0.00 sec)

mysql> select cast(f1 as signed), cast(f2 as signed), cast(f3 as signed), cast(f4 as signed), cast(f5 as signed) from dt;
+--------------------+--------------------+--------------------+--------------------+--------------------+
| cast(f1 as signed) | cast(f2 as signed) | cast(f3 as signed) | cast(f4 as signed) | cast(f5 as signed) |
+--------------------+--------------------+--------------------+--------------------+--------------------+
|           20210101 |             210000 |               2021 |     20210101231035 |     20210101231035 |
+--------------------+--------------------+--------------------+--------------------+--------------------+
1 row in set (0.00 sec)

mysql> select cast(f1 as decimal), cast(f2 as decimal), cast(f3 as decimal), cast(f4 as decimal), cast(f5 as decimal) from dt;
+---------------------+---------------------+---------------------+---------------------+---------------------+
| cast(f1 as decimal) | cast(f2 as decimal) | cast(f3 as decimal) | cast(f4 as decimal) | cast(f5 as decimal) |
+---------------------+---------------------+---------------------+---------------------+---------------------+
|            20210101 |              210000 |                2021 |          9999999999 |          9999999999 |
+---------------------+---------------------+---------------------+---------------------+---------------------+
1 row in set, 2 warnings (0.00 sec)

mysql> show warnings;
+---------+------+--------------------------------------------------------------+
| Level   | Code | Message                                                      |
+---------+------+--------------------------------------------------------------+
| Warning | 1264 | Out of range value for column 'cast(f4 as decimal)' at row 1 |
| Warning | 1264 | Out of range value for column 'cast(f5 as decimal)' at row 1 |
+---------+------+--------------------------------------------------------------+
2 rows in set (0.00 sec)

*/
void CommonBiaodashi::CheckCastFunctionExpr()
{
    auto srcExpr = std::dynamic_pointer_cast<CommonBiaodashi>( children[ 0 ] );
    auto srcValueType = srcExpr->GetValueType();
    switch ( value_type )
    {
        case BiaodashiValueType::INT:
        {
            if ( BiaodashiValueType::TEXT != srcValueType &&
                 BiaodashiValueType::BOOL != srcValueType &&
                 BiaodashiValueType::TINY_INT != srcValueType &&
                 BiaodashiValueType::SMALL_INT != srcValueType &&
                 BiaodashiValueType::INT != srcValueType
               )
            {
                string msg( "cast " + get_name_of_value_type( srcValueType ) + " as integer" );
                ThrowNotSupportedException( msg.data() );
            }
            break;
        }
        case BiaodashiValueType::LONG_INT:
        {
            if ( BiaodashiValueType::TEXT != srcValueType &&
                 BiaodashiValueType::BOOL != srcValueType &&
                 BiaodashiValueType::TINY_INT != srcValueType &&
                 BiaodashiValueType::SMALL_INT != srcValueType &&
                 BiaodashiValueType::INT != srcValueType &&
                 BiaodashiValueType::LONG_INT != srcValueType
               )
            {
                string msg( "cast " + get_name_of_value_type( srcValueType ) + " as integer" );
                ThrowNotSupportedException( msg.data() );
            }
            break;
        }
        case BiaodashiValueType::DATE:
        {
            // TODO: simplify if cast date to date
            if ( BiaodashiValueType::DATE != srcValueType &&
                 BiaodashiValueType::DATE_TIME != srcValueType &&
                 BiaodashiValueType::TIMESTAMP != srcValueType &&
                 BiaodashiValueType::TEXT != srcValueType )
            {
                string msg( "cast " + get_name_of_value_type( srcValueType ) + " as date" );
                ThrowNotSupportedException( msg.data() );
            }
            break;
        }
        case BiaodashiValueType::DATE_TIME:
        {
            // TODO: simplify if cast datetime to datetime
            if ( BiaodashiValueType::DATE != srcValueType &&
                 BiaodashiValueType::DATE_TIME != srcValueType &&
                 BiaodashiValueType::TIMESTAMP != srcValueType &&
                 BiaodashiValueType::TEXT != srcValueType )
            {
                string msg( "cast " + get_name_of_value_type( srcValueType ) + " as date" );
                ThrowNotSupportedException( msg.data() );
            }
            break;
        }
        case BiaodashiValueType::DECIMAL:
        {
            if ( BiaodashiValueType::TEXT != srcValueType &&
                 BiaodashiValueType::DECIMAL != srcValueType &&
                 BiaodashiValueType::BOOL != srcValueType &&
                 BiaodashiValueType::TINY_INT != srcValueType &&
                 BiaodashiValueType::SMALL_INT != srcValueType &&
                 BiaodashiValueType::INT != srcValueType &&
                 BiaodashiValueType::LONG_INT != srcValueType )
            {
                string msg( "cast " + get_name_of_value_type( srcValueType ) + " as decimal" );
                ThrowNotSupportedException( msg.data() );
            }
            break;
        }
    
        default:
        {
            string msg( "cast " + get_name_of_value_type( srcValueType ) + " as " + get_name_of_value_type( value_type ) );
            ThrowNotSupportedException( msg.data() );
            break;
        }
    }
}

void CommonBiaodashi::CheckFunctionExpr(ExprContextPointer arg_expr_context, bool expectOnlyConst)
{
    std::string func_name = this->CheckFunctionName();
    auto function = boost::get<SQLFunctionPointer>(content);
    aries_utils::to_upper(func_name);

    int my_start_position = -1;
    int my_ending_position = -1;

    /*specially handle aggregate functions -- entering*/
    bool is_agg_func = (function->GetIsAggFunc() == true);
    if (is_agg_func)
    {
        my_start_position = arg_expr_context->referenced_column_array.size();
    }

    /*check all the parameters! -- a lot of work should be done!!!*/
    this->CheckExpr4Children(arg_expr_context, expectOnlyConst);

    /*additional check for agg functions*/
    switch (function->GetType()) {
    case aries_acc::AriesSqlFunctionType::AVG:
    case aries_acc::AriesSqlFunctionType::SUM:
    case aries_acc::AriesSqlFunctionType::MAX:
    case aries_acc::AriesSqlFunctionType::MIN:
    case aries_acc::AriesSqlFunctionType::COUNT:
    case aries_acc::AriesSqlFunctionType::ABS:
    {
        std::shared_ptr<CommonBiaodashi> child;
        for (const auto& child_ptr : children) {
            auto ptr = std::dynamic_pointer_cast<CommonBiaodashi>(child_ptr);
            if (ptr->type != BiaodashiType::Distinct) {
                child = ptr;
                break;
            }
        }

        if ( !child )
        {
           ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, func_name.data() ) ;
        }

        if ( function->GetType() != aries_acc::AriesSqlFunctionType::COUNT) {
            std::vector<BiaodashiValueType> tmp{
                    BiaodashiValueType::TINY_INT, BiaodashiValueType::SMALL_INT,
                    BiaodashiValueType::INT, BiaodashiValueType::LONG_INT,
                    BiaodashiValueType::DECIMAL, BiaodashiValueType::FLOAT,
                    BiaodashiValueType::DOUBLE};
            child->CheckExprTypeMulti(tmp);
        } else {
            nullable = false;
        }

        if (function->GetType() == aries_acc::AriesSqlFunctionType::AVG)
        {
            switch (child->GetValueType())
            {
            case BiaodashiValueType::FLOAT:
                value_type = BiaodashiValueType::FLOAT;
                numeric_type_value = NumericTypeOffsetFloat;
                break;
            case BiaodashiValueType::DOUBLE:
                value_type = BiaodashiValueType::DOUBLE;
                numeric_type_value = NumericTypeOffsetDouble;
                break;
            default:
                value_type = BiaodashiValueType::DECIMAL;
                numeric_type_value = NumericTypeOffsetDecimal;
                break;
            }
        }
        else if (function->GetType() == aries_acc::AriesSqlFunctionType::COUNT)
        {
            value_type = BiaodashiValueType::LONG_INT;
            numeric_type_value = NumericTypeOffsetInt64;
            nullable = false;
        }
        else if (function->GetType() == aries_acc::AriesSqlFunctionType::SUM)
        {
            switch (child->GetValueType())
            {
            case BiaodashiValueType::TINY_INT:
            case BiaodashiValueType::BOOL:
            case BiaodashiValueType::SMALL_INT:
            case BiaodashiValueType::INT:
                value_type = BiaodashiValueType::LONG_INT;
                numeric_type_value = NumericTypeOffsetInt64;
                break;
            case BiaodashiValueType::LONG_INT:
            case BiaodashiValueType::DECIMAL:
            case BiaodashiValueType::FLOAT:
            case BiaodashiValueType::DOUBLE:
                value_type = child->GetValueType();
                numeric_type_value = child->GetNumericTypeValue();
                break;
            default:break;
            }
        }
        else if (function->GetType() == aries_acc::AriesSqlFunctionType::MAX || function->GetType() == aries_acc::AriesSqlFunctionType::MIN)
        {
            value_type = child->GetValueType();
            numeric_type_value = child->GetNumericTypeValue();
        }
        else if (function->GetType() == aries_acc::AriesSqlFunctionType::ABS)
        {
            value_type = child->GetValueType();
            numeric_type_value = child->GetNumericTypeValue();
        }
        if (function->GetType() != aries_acc::AriesSqlFunctionType::ABS && function->GetType() != aries_acc::AriesSqlFunctionType::COUNT) {
            nullable = true;
        }
        break;
    }
    case aries_acc::AriesSqlFunctionType::EXTRACT:
    {
        auto arg = std::dynamic_pointer_cast<CommonBiaodashi>(children[0]);
        auto arg_value = boost::get<std::string>(arg->GetContent());
        boost::to_upper(arg_value);
        if (arg_value == "HOUR_MICROSECOND" || arg_value == "DAY_MICROSECOND")
        {
            value_type = BiaodashiValueType::LONG_INT;
        }
        else
        {
            value_type = BiaodashiValueType::INT;
        }
        length = 1;
        break;
    }
    case aries_acc::AriesSqlFunctionType::DATE:
    {
        value_type = BiaodashiValueType::DATE;
        break;
    }
    case aries_acc::AriesSqlFunctionType::NOW:
    {
        value_type = BiaodashiValueType::DATE_TIME;
        break;
    }
    case aries_acc::AriesSqlFunctionType::UNIX_TIMESTAMP:
    {
        if( children.size() > 1 )
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, func_name.data() ) ;

        // TODO: use actual time zone offset
        auto second_param = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhengshu, -8 * 3600);
        second_param->value_type = BiaodashiValueType::INT;
        value_type = BiaodashiValueType::DECIMAL;
        numeric_type_value |= NumericTypeOffsetDecimal;

        AddChild( second_param );

        break;
    }
    case aries_acc::AriesSqlFunctionType::TIME_DIFF:
    {
        if( children.size() != 2 )
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, func_name.data() ) ;
        value_type = BiaodashiValueType::TIME;
        break;
    }
    case aries_acc::AriesSqlFunctionType::DATE_DIFF:
    {
        if( children.size() != 2 )
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, func_name.data() ) ;
        value_type = BiaodashiValueType::INT;
        numeric_type_value = NumericTypeOffsetInt32;
        break;
    }
    case aries_acc::AriesSqlFunctionType::DATE_SUB:
    case aries_acc::AriesSqlFunctionType::DATE_ADD:
    {
        auto first = std::dynamic_pointer_cast<CommonBiaodashi>(children[0]);
        auto second = std::dynamic_pointer_cast<CommonBiaodashi>(children[1]);

        std::string interval_type = "DAY";

        if (second->type != BiaodashiType::IntervalExpression)
        {
            auto tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::IntervalExpression, interval_type);
            tmp->AddChild(second);
            SetChild( 1, tmp );
        }
        else
        {
            interval_type = boost::get<std::string>(second->content);
            aries_utils::to_upper(interval_type);
        }

        // auto has_time_parts = (interval_type == "HOURS" || interval_type == "MINUTES" || interval_type == "SECONDS");
        auto has_date_parts = (interval_type == "YEAR" || interval_type == "MONTH" || interval_type == "DAY");
        bool has_date_arg = false;

        if (first->type == BiaodashiType::Zifuchuan)
        {
                auto first_string = boost::get<std::string>(first->content);

                aries_acc::MYSQL_TIME my_time = {0};
                int was_cut = 0;
                auto time_type = aries_acc::str_to_datetime(first_string.c_str(), first_string.size(), &my_time, 0, &was_cut);
                switch (time_type) {
                case aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATE:
                    has_date_arg = true;
                    break;
                case aries_acc::enum_mysql_timestamp_type::MYSQL_TIMESTAMP_DATETIME:
                    break;
                default:
                    LOG(INFO) << "invalid string arg as date/datetime: " << first_string << std::endl;
                    assert(0);
                    break;
                }
        }
        else
        {
            has_date_arg = first->value_type == BiaodashiValueType::DATE;
        }

        if (has_date_arg && has_date_parts)
        {
            value_type = BiaodashiValueType::DATE;
        }
        else
        {
            value_type = BiaodashiValueType::DATE_TIME;
        }
        break;
    }
    case aries_acc::AriesSqlFunctionType::CAST: {
        CheckCastFunctionExpr();
        // if (value_type == BiaodashiValueType::TEXT && length == 0) {
        //     auto value = std::dynamic_pointer_cast<CommonBiaodashi>(children[0]);
        //     length = value->length;
        // }

        break;
    }
    case aries_acc::AriesSqlFunctionType::CONVERT: {
        ARIES_ASSERT( 0, "Not support function convert" );
        break;
    }
    case aries_acc::AriesSqlFunctionType::MONTH: {
        value_type = BiaodashiValueType::INT;
        break;
    }
    case aries_acc::AriesSqlFunctionType::DATE_FORMAT: {
        value_type = BiaodashiValueType::TEXT;
        if ( children.size() != 2 )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, "date_format" );
        }
        auto second_param = ((CommonBiaodashi* )(children[1].get()));
        if ( second_param->GetType() != BiaodashiType::Zifuchuan )
        {
            ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "DATE_FORMAT's 2nd parameter was non-string");
        }

        auto format_string = boost::get< std::string >( second_param->GetContent() );
        length = aries_acc::get_format_length( format_string.c_str() );
        break;
    }
    case aries_acc::AriesSqlFunctionType::COALESCE: {
        auto first_child = (CommonBiaodashi*)(children[0].get());
        value_type = first_child->value_type;
        nullable = true;
        length = first_child->length;
        int string_len = first_child->getStringLength();
        for (size_t i = 1; i < children.size(); i++) {
            auto index = static_cast<int>(value_type);
            auto child = (CommonBiaodashi*)(children[i].get());
            auto offset = static_cast<int>(child->value_type);
            auto child_len = child->length;
            value_type = static_cast<BiaodashiValueType>(column_types_merge_rules[index][offset]);
            if (child_len > length) {
                length = child_len;
            }

            int child_string_len = child->getStringLength();

            if (child_string_len > string_len) {
                string_len = child_string_len;
            }
        }

        if (value_type == BiaodashiValueType::TEXT) {
            length = string_len;
        }
        break;
    }
    case aries_acc::AriesSqlFunctionType::CONCAT: {
        value_type = BiaodashiValueType::TEXT;

        if ( children.size() == 0 )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, "concat" );
        }

        length = 0;
        for (size_t i = 0; i < children.size(); i++) {
            auto child = (CommonBiaodashi*)(children[i].get());
            length += child->getStringLength();
        }
        nullable = true;
        break;
    }
    case aries_acc::AriesSqlFunctionType::SUBSTRING: {
        if( children.size() != 3 )
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, func_name.data() ) ;
        value_type = BiaodashiValueType::TEXT;
        auto first_child = (CommonBiaodashi*)(children[0].get());
        length = first_child->getStringLength();

        auto second_child =  (CommonBiaodashi*)(children[1].get());
        auto thrid_child = (CommonBiaodashi*)(children[2].get());
        int start = 0;
        if (second_child->type == BiaodashiType::Zhengshu) {
            start = boost::get<int>(second_child->content) - 1;
            if (start < 0) {
                start = length + start + 1;
            }
        }

        length = max(0, length - start);
        int target_length = length;
        if (thrid_child->type == BiaodashiType::Zhengshu) {
            auto len = boost::get<int>(thrid_child->content);
            if (len != -1) {
                target_length = min(len, length);
            }
        }
        length = target_length;
        break;
    }
    case aries_acc::AriesSqlFunctionType::TRUNCATE: {
        if ( children.size() != 2 )
        {
            ARIES_EXCEPTION( ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT, "truncate" );
        }
        auto data = (CommonBiaodashi *) ( children[0].get() );
        value_type = data->value_type;
        switch ( value_type )
        {
        case BiaodashiValueType::TINY_INT:
        case BiaodashiValueType::SMALL_INT:
        case BiaodashiValueType::INT:
        case BiaodashiValueType::LONG_INT:
        case BiaodashiValueType::DECIMAL:
            break;
        default:
            ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, ( "truncate data type " + get_name_of_value_type(value_type) ).c_str() );
        }
        length = 1;
        break;
    }
    case aries_acc::AriesSqlFunctionType::DICT_INDEX:
    {
        auto data = ( CommonBiaodashi * )( children[0].get() );
        if( data->type == BiaodashiType::Lie )
        {
            auto col = boost::get< ColumnShellPointer >( data->content );
            if( col->GetColumnStructure()->GetEncodeType() == EncodeType::DICT )
            {
                auto parent = ( CommonBiaodashi * )GetParent();
                if ( parent )
                {
                    /*
                    if ( BiaodashiType::Bijiao == parent->type )
                    {
                        assert(this->content.which() == 1);
                        auto cmpOpContent = boost::get< int >( content );
                        auto cmpOp = static_cast< ComparisonType >( cmpOpContent );
                        if ( ComparisonType::DengYu != cmpOp && ComparisonType::BuDengYu != cmpOp )
                        {
                            string errMsg( "function dict_index in expression '" );
                            errMsg.append( get_name_of_expr_type( parent->type ) ).append( "'" );
                            ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, errMsg.data() );
                        }
                    }
                    else
                    */
                    {
                        string errMsg( "function dict_index in expression '" );
                        errMsg.append( get_name_of_expr_type( parent->type ) ).append( "'" );
                        ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, errMsg.data() );
                    }
                }
                if ( ExprContextType::SelectExpr != arg_expr_context->type )
                {
                    string errMsg( "function dict_index in non-select context" );
                    ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, errMsg.data() );
                }

                value_type = col->GetColumnStructure()->GetEncodedIndexType();
                nullable = data->nullable;
                length = 1;
            }
            else
            {
                ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "column is not dict encoded" );
            }
        }
        else
        {
            ARIES_EXCEPTION_SIMPLE( ER_NOT_SUPPORTED_YET, "argument type is not a simple column" );
        }
        break;
    }
    default:
    {
        this->value_type =
            (this->children.size() > 0)
                ? ((CommonBiaodashi *)(this->children[0].get()))->GetValueType()
                : BiaodashiValueType::UNKNOWN;
        break;
    }
    }

    handleNullableFunction(function->GetType());

    /*specially handle aggregate functions -- leaving*/
    if (is_agg_func)
    {
        my_ending_position = arg_expr_context->referenced_column_array.size();
        this->SetAggStatus(my_start_position, my_ending_position, arg_expr_context);
    }

    if (is_agg_func)
    {
        arg_expr_context->see_agg_func = true;
    }
}

static aries_acc::Decimal GetDecimalValue( const std::shared_ptr<CommonBiaodashi>& expression )
{
    aries_acc::Decimal decimal_value;
    if ( expression->GetType() == BiaodashiType::Zhengshu )
    {
        switch ( expression->GetValueType() )
        {
            case BiaodashiValueType::TINY_INT:
            {
                auto value = static_cast< int8_t >( boost::get< int >( expression->GetContent() ) );
                decimal_value = aries_acc::Decimal( value );
                break;
            }
            case BiaodashiValueType::SMALL_INT:
            {
                auto value = static_cast< int16_t >( boost::get< int >( expression->GetContent() ) );
                decimal_value = aries_acc::Decimal( value );
                break;
            }
            case BiaodashiValueType::INT:
            {
                auto value = boost::get< int >( expression->GetContent() );
                decimal_value = aries_acc::Decimal( value );
                break;
            }
            case BiaodashiValueType::LONG_INT:
            {
                auto value = boost::get< int64_t >( expression->GetContent() );
                decimal_value = aries_acc::Decimal( value );
                break;
            }
            default:
                ARIES_ASSERT( 0, "invalid value type for BiaodashiType::Zhengshu" );
                break;
        }
        decimal_value.CheckAndSetRealPrecision();
    }
    else if (expression->GetType() == BiaodashiType::Decimal)
    {
        auto string_value = boost::get< std::string >( expression->GetContent() );
        decimal_value = aries_acc::Decimal( string_value.c_str() );
        decimal_value.CheckAndSetRealPrecision();
    }
    else
    {
        switch ( expression->GetValueType() )
        {
            case BiaodashiValueType::TINY_INT:
            {
                decimal_value = aries_acc::Decimal( TINYINT_PRECISION, DEFAULT_SCALE );
                break;
            }
            case BiaodashiValueType::SMALL_INT:
            {
                decimal_value = aries_acc::Decimal( SMALLINT_PRECISION, DEFAULT_SCALE );
                break;
            }
            case BiaodashiValueType::INT:
            {
                decimal_value = aries_acc::Decimal( INT_PRECISION, DEFAULT_SCALE );
                break;
            }
            case BiaodashiValueType::LONG_INT:
            {
                decimal_value = aries_acc::Decimal( BIGINT_PRECISION, DEFAULT_SCALE );
                break;
            }
            case BiaodashiValueType::DECIMAL:
            {
                decimal_value = aries_acc::Decimal( expression->GetLength(), expression->GetAssociatedLength() );
                break;
            }
            default:
                ARIES_ASSERT( 0, "invalid value type for AriesExprType::INTEGER" );
                break;
        }
    }

    return decimal_value;
}

/*touch work!*/
void CommonBiaodashi::CheckExpr(ExprContextPointer arg_expr_context, bool expectOnlyConst )
{

    this->SetExprContext(arg_expr_context);

    arg_expr_context->check_serial += 1; // check_serial
    this->SetColumnStartingPosition(arg_expr_context->referenced_column_array.size());

    switch (this->type)
    {
    case BiaodashiType::Zhenjia:
        this->value_type = BiaodashiValueType::BOOL;
        break;

    case BiaodashiType::Zhengshu:
        if (value_type == BiaodashiValueType::LONG_INT) {
            numeric_type_value = NumericTypeOffsetInt64;
        } else {
            numeric_type_value = NumericTypeOffsetInt32;
        }
        break;

    case BiaodashiType::Fudianshu:
        this->value_type = BiaodashiValueType::FLOAT;
        numeric_type_value = NumericTypeOffsetFloat;
        break;
    case BiaodashiType::Decimal:
        value_type = BiaodashiValueType::DECIMAL;
        numeric_type_value = NumericTypeOffsetDecimal;
        break;

    case BiaodashiType::Zifuchuan:
    {
        this->value_type = BiaodashiValueType::TEXT;
        auto content_string = boost::get<std::string>(content);
        length = content_string.size();
        break;
    }
    case BiaodashiType::Biaoshifu:
        this->checkBiaoshifu(arg_expr_context, expectOnlyConst);
        is_calculable = false;
        break;

    case BiaodashiType::Shuzu:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        std::dynamic_pointer_cast<CommonBiaodashi>(this->children[1])
            ->CheckExprType(BiaodashiValueType::INT);
        this->value_type = BiaodashiValueType::UNKNOWN; // todo

        break;

    case BiaodashiType::Qiufan:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = BiaodashiValueType::BOOL;
        break;

    case BiaodashiType::Likeop:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        std::dynamic_pointer_cast<CommonBiaodashi>(this->children[0])
            ->CheckExprType(BiaodashiValueType::TEXT);
        std::dynamic_pointer_cast<CommonBiaodashi>(this->children[1])
            ->CheckExprType(BiaodashiValueType::TEXT);
        this->value_type = BiaodashiValueType::BOOL;
        break;

    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = BiaodashiValueType::BOOL;
        break;
    case BiaodashiType::ExprList:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = BiaodashiValueType::INT;
        break;
    case BiaodashiType::Between: {
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);

        /*x between a and b*/
        auto target = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[0]);
        auto low = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[1]);
        auto high = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[2]);

        if (!__compareTwoExpr4Types(target.get(), low.get()) || !__compareTwoExpr4Types(target.get(), high.get())) {
            string errMsg("CheckExpr: Between value type: "
                          + get_name_of_value_type(std::dynamic_pointer_cast<CommonBiaodashi>(this->children[0])->GetValueType())
                          + " is not allowed!");
            ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, errMsg);
        }

        this->value_type = BiaodashiValueType::BOOL;
        break;
    }

    case BiaodashiType::Cunzai:

        /*tell the child that he is in Exists!*/
        arg_expr_context->exist_begin_mark = true;
        arg_expr_context->exist_end_mark = false;

        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = BiaodashiValueType::BOOL;

        arg_expr_context->exist_begin_mark = false;
        arg_expr_context->exist_end_mark = true;
        break;

    case BiaodashiType::Case:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);

        /*case expr (when then)+ else*/
        /*TODO: we need to check all exprs in then and else!*/

        /**
         * Case 表达式是否为 nullable 取决于 then 表达式和 else 表达式是否有 nullable
         * *如果没有 else，则一定是 nullable*
         */
        nullable = false;
        for (size_t i = 2; i < children.size(); i++) {
            if ((i % 2 != 0) && (i != children.size() - 1)) {
                continue;
            }

            auto child = (CommonBiaodashi*)(children[i].get());
            if (!child || child->nullable) {
                nullable = true;
            }
        }

        if (1 > 0)
        {
            int string_len = 1;
            size_t tl = children.size();
            bool have_else = false;

            int max_intg = 0;
            int max_scale = 0;

            /*get else expr type!*/
            if (children[tl - 1] != nullptr)
            {
                have_else = true;
                auto else_item =
                    std::dynamic_pointer_cast<CommonBiaodashi>(children[tl - 1]);
                value_type = else_item->GetValueType();
                LOG(INFO) << "case else_item type: " << static_cast<int>(value_type) << std::endl;
                numeric_type_value |= else_item->GetNumericTypeValue();
                string_len = else_item->getStringLength();
                length = else_item->length;
                max_intg = max( max_intg, else_item->length - else_item->associated_length );
                max_scale = max( max_scale, else_item->associated_length );

                if (nullable)
                {
                    switch(else_item->type)
                    {
                        case BiaodashiType::Zhengshu:
                        case BiaodashiType::Zifuchuan:
                        case BiaodashiType::Fudianshu:
                        case BiaodashiType::Decimal:
                        case BiaodashiType::Null:
                            else_item->nullable = true;
                        default: break;
                    }
                }
            }

            /*check each then expr!*/
            for (size_t i = 2; i < tl - 1; i += 2)
            {
                auto item = std::dynamic_pointer_cast<CommonBiaodashi>(children[i]);
                numeric_type_value |= item->GetNumericTypeValue();

                string_len = max(item->getStringLength(), string_len);
                max_intg = max( max_intg, item->length - item->associated_length );
                max_scale = max( max_scale, item->associated_length );

                if (i == 2 && !have_else) {
                    value_type = item->value_type;
                } else {
                    value_type = column_types_merge_rules[static_cast<int>(item->value_type)][static_cast<int>(value_type)];
                }

                length = max(length, item->length);

                if (nullable)
                {
                    switch(item->type)
                    {
                        case BiaodashiType::Zhengshu:
                        case BiaodashiType::Zifuchuan:
                        case BiaodashiType::Fudianshu:
                        case BiaodashiType::Decimal:
                        case BiaodashiType::Null:
                            item->nullable = true;
                        default: break;
                    }
                }
            }

            if (value_type == BiaodashiValueType::TEXT) {
                length = string_len;
            } else if (value_type == BiaodashiValueType::DECIMAL) {
                aries_acc::Decimal target( max_intg + max_scale, max_scale );
                length = target.prec;
                associated_length = target.frac;
            }
            LOG(INFO) << "case type: " << static_cast<int>(value_type) << std::endl;

            //		LOG(INFO) << "\nyelv:" << this->value_type << "\n";
        }

        break;

    case BiaodashiType::Kuohao:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[0])
                               ->GetValueType();
        break;

    case BiaodashiType::Bijiao: {
        // arg_expr_context.query_one_row_requirement = true; //Is this right???
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        auto left = (CommonBiaodashi *)(children[0].get());
        auto right = (CommonBiaodashi *)(children[1].get());
        if (!__compareTwoExpr4Types(left, right))
        {
            string errMsg("two things cannot be compared: ");
            errMsg.append("left: " + get_name_of_value_type(left->value_type));
            errMsg.append(", right: " + get_name_of_value_type(right->value_type));
            ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, errMsg);
        }
        this->value_type = BiaodashiValueType::BOOL;
        break;
    }

    case BiaodashiType::Yunsuan:
    {
        // arg_expr_context.query_one_row_requirement = true;
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        auto left = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[0]);
        auto right = std::dynamic_pointer_cast<CommonBiaodashi>(this->children[1]);

        numeric_type_value = left->numeric_type_value | right->numeric_type_value;
        numeric_type_value |= NumericTypeOffsetInt32;

        if (1 > 0)
        {
            std::vector<BiaodashiValueType> tmp{
                BiaodashiValueType::TINY_INT, BiaodashiValueType::SMALL_INT,
                BiaodashiValueType::INT,      BiaodashiValueType::LONG_INT,
                BiaodashiValueType::DECIMAL,  BiaodashiValueType::FLOAT,
                BiaodashiValueType::DOUBLE};
            left->CheckExprTypeMulti(tmp);
            right->CheckExprTypeMulti(tmp);
        }

        /**
         * Division's result should be decimal/float/double:
         * mysql> select 10/5;
         * +--------+
         * | 10/5   |
         * +--------+
         * | 2.0000 |
         * +--------+
         */
        auto calc_type =  static_cast< CalcType >( boost::get<int>(content) );
        if (calc_type == CalcType::DIV) {
            numeric_type_value |= NumericTypeOffsetDecimal;
        }

        value_type = GetPromotedValueType();

        if (value_type == BiaodashiValueType::DECIMAL && left->CouldBeCompactDecimal() && right->CouldBeCompactDecimal())
        {
            auto left_value = GetDecimalValue( left );
            auto right_value = GetDecimalValue( right );
            switch ( calc_type )
            {
                case CalcType::ADD:
                {
                    left_value.CalcAddTargetPrecision( right_value );
                    break;
                }
                case CalcType::SUB:
                {
                    left_value.CalcSubTargetPrecision( right_value );
                    break;
                }
                case CalcType::MUL:
                {
                    left_value.CalcMulTargetPrecision( right_value );
                    break;
                }
                case CalcType::DIV:
                {
                    left_value.CalcDivTargetPrecision( right_value );
                    break;
                }
                case CalcType::MOD:
                {
                    left_value.CalcModTargetPrecision( right_value );
                    break;
                }

            }

            length = left_value.prec;
            associated_length = left_value.frac;
        }
        break;
    }
    case BiaodashiType::Andor:
        this->CheckExpr4Children(arg_expr_context, expectOnlyConst);
        this->value_type = BiaodashiValueType::BOOL;
        break;

    case BiaodashiType::Star:
        this->CheckStarExpr(arg_expr_context);
        break;

    case BiaodashiType::Hanshu:
    case BiaodashiType::SQLFunc:
        this->CheckFunctionExpr(arg_expr_context, expectOnlyConst);
        break;

    case BiaodashiType::Query:
        this->CheckQueryExpr(arg_expr_context);
        break;

    case BiaodashiType::IfCondition: {
        CheckExpr4Children(arg_expr_context, expectOnlyConst);

        auto value_child = (CommonBiaodashi*)(children[1].get());
        value_type = value_child->value_type;
        int string_len = value_child->getStringLength();
        auto max_intg = value_child->length - value_child->associated_length;
        auto max_scale = value_child->associated_length;

        auto else_child = (CommonBiaodashi*)(children[2].get());
        if (else_child->GetType() != BiaodashiType::Null) {
            value_type = column_types_merge_rules[static_cast<int>(else_child->value_type)][static_cast<int>(value_type)];
            max_intg = max( max_intg, else_child->length - else_child->associated_length );
            max_scale = max( max_scale, else_child->associated_length );
        }

        if (value_type == BiaodashiValueType::TEXT) {
            length = max(string_len, else_child->getStringLength());
        }
        else if ( value_type == BiaodashiValueType::DECIMAL )
        {
            aries_acc::Decimal target( max_intg + max_scale, max_scale );
            length = target.prec;
            associated_length = target.frac;
        } else {
            length = max(value_child->length, else_child->length);
        }

        break;
    }

    case BiaodashiType::IsNotNull:
    case BiaodashiType::IsNull: {
        CheckExpr4Children(arg_expr_context, expectOnlyConst);
        value_type = BiaodashiValueType::BOOL;
        break;
    }

    case BiaodashiType::Null:
        value_type = BiaodashiValueType::BOOL;
        nullable = true;
        break;

    case BiaodashiType::Distinct:
        value_type = BiaodashiValueType::BOOL;
        break;
    case BiaodashiType::IntervalExpression:
        CheckExpr4Children(arg_expr_context, expectOnlyConst);
        value_type = BiaodashiValueType::TEXT;
        break;
    case BiaodashiType::QuestionMark:
        value_type = BiaodashiValueType::UNKNOWN;
        break;
    default:
        char ebuff[ERRMSGSIZE];
        (void) snprintf(ebuff, sizeof(ebuff), "%s Invalid expression type %d.", ERRMSG_SYNTAX_ERROR, type);
        string errMsg(ebuff);
        ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, errMsg);
    }

    this->SetColumnEndingPosition(arg_expr_context->referenced_column_array.size());
}

bool CommonBiaodashi::CheckExprType(BiaodashiValueType arg_type)
{
    if (this->value_type == BiaodashiValueType::UNKNOWN)
    {
        return true; // todo
    }

    if (this->value_type != arg_type)
    {
        char ebuff[ERRMSGSIZE];
        (void) snprintf(ebuff, sizeof(ebuff), "%s ERROR: expression [%s] should have a type [%d].",
                        ERRMSG_SYNTAX_ERROR, ToString().data(), (int)arg_type);
        string errMsg(ebuff);
        ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, errMsg);
    }

    return true;
}

void CommonBiaodashi::CheckExpr4Children(ExprContextPointer arg_expr_context, bool expectOnlyConst)
{

    for (const auto &child : children)
    {
        if (child != nullptr)
        {
            auto child_expr = std::dynamic_pointer_cast<CommonBiaodashi>(child);
            child_expr->CheckExpr(arg_expr_context, expectOnlyConst);
            if (child_expr->nullable)
            {
                nullable = true;
            }

            if (!child_expr->is_calculable) {
                is_calculable = false;
            }
        }
    }
}

void CommonBiaodashi::CheckQueryExpr(ExprContextPointer arg_expr_context)
{
    QueryContextType fill_type =
        QueryContextType::ExprSubQuery; /*we need to know which QueryContextType
                                           according to the expr context to do*/
    int fill_query_level = arg_expr_context->GetQueryContext()->query_level + 1;

    AbstractQueryPointer agp = boost::get<AbstractQueryPointer>(this->content);
    // SelectStructure* fill_select_structure = (SelectStructure *)(agp.get());

    QueryContextPointer fill_query_context = arg_expr_context->GetQueryContext();

    ExprContextPointer fill_expr_context = arg_expr_context;

    QueryContextPointer new_query_context =
        std::make_shared<QueryContext>(fill_type, fill_query_level,
                                       agp, // fill_select_structure,
                                       fill_query_context, fill_expr_context);

    fill_query_context->subquery_context_array.push_back(new_query_context);

    std::shared_ptr<SelectStructure> real_subquery =
        std::dynamic_pointer_cast<SelectStructure>(agp);

    SQLTreeNodeBuilderPointer new_node_builder =
        std::make_shared<SQLTreeNodeBuilder>(agp);

    if (arg_expr_context->exist_begin_mark == true &&
        arg_expr_context->exist_end_mark == false)
    {
        real_subquery->SetIAmInExist(true);
    }

    real_subquery->CheckQueryGate2(
            std::dynamic_pointer_cast<SelectStructure>(
                    arg_expr_context->GetQueryContext()->GetSelectStructure())
                    ->GetDefaultSchemaAgent(),
        new_node_builder, new_query_context);

    /*our core purpose is to setup this expr's value_type, ok what is it?*/

    /*
      var result_of_subquery = this.checkQuery(arg_expr.content, new_query_context);
      //will have new exspression context in the subquery


      var result_name_array = result_of_subquery[0];
      var result_type_array = result_of_subquery[1];

      if (result_type_array.length == 1) {
      arg_expr.value_type = result_type_array[0];
      } else {
      arg_expr.value_type = BiaodashiValueType.LIST;
      arg_expr.value_type_array = result_type_array;
      }

    */

    RelationStructurePointer rsp = real_subquery->GetRelationStructure();

    assert(rsp != nullptr);
    if (rsp->GetColumnCount() == 1)
    {
        this->value_type = rsp->GetColumn(0)->GetValueType();
        nullable = rsp->GetColumn( 0 )->IsNullable();
    }
    else
    {
        this->value_type = BiaodashiValueType::LIST;
        // we do not need the value_type_array as in js version!
    }

    /*we need insert sub-query's referenced columns to here todo*/
    for (size_t oi = 0; oi < new_query_context->outer_column_array.size(); oi++)
    {
        arg_expr_context->referenced_column_array.push_back(
            new_query_context->outer_column_array[oi]);
        arg_expr_context->referenced_column_agg_status_array.push_back(
            new_query_context->outer_column_agg_status_array[oi]);
    }
}

bool CommonBiaodashi::CheckExprTypeMulti(std::vector<BiaodashiValueType> arg_type_array)
{
    if (this->value_type == BiaodashiValueType::UNKNOWN)
    {
        return true; // todo
    }

    bool matching = false;
    for (size_t i = 0; i < arg_type_array.size(); i++)
    {
        if (this->value_type == arg_type_array[i])
        {
            matching = true;
            break;
        }
    }

    if (!matching)
    {
        char ebuff[ERRMSGSIZE];
        (void) snprintf(ebuff, sizeof(ebuff), "%s ERROR: expression [%s] should have a type in multi.",
                        ERRMSG_SYNTAX_ERROR, ToString().data());
        string errMsg(ebuff);
        LOG(INFO) << errMsg;
        // TODO
        // throw AriesFrontendException(ER_SYNTAX_ERROR, errMsg);
    }

    return matching;
}

/**/
std::string CommonBiaodashi::GetName()
{
    /*if I am a Biaoshifu(IT SHOULD NOT HAPPEN!), or columnshell, return the column, or
     * return "nameless"*/

    ColumnShellPointer colsh;
    std::string ret;
    static uint64_t idx = 0;

    switch (this->type)
    {

    case BiaodashiType::Lie:
        colsh = boost::get<ColumnShellPointer>(this->content);
        ret = colsh->GetColumnName();
        break;
    case BiaodashiType::QuestionMark:
        ret = "?" + std::to_string(idx++);
        break;

    default:
        if (!name.empty()) {
            ret = name;
        } else {
            ret = "nameless";
        }
    }

    return ret;
}

std::string CommonBiaodashi::GetTableName()
{
    /*if I am a Biaoshifu(IT SHOULD NOT HAPPEN!), or columnshell, return the column, or
     * return "nameless"*/

    ColumnShellPointer colsh;
    std::string ret;

    switch (this->type)
    {

    case BiaodashiType::Lie:
        colsh = boost::get<ColumnShellPointer>(this->content);
        ret = colsh->GetTableName();
        break;

    default:
        ret = "nameless";
    }

    return ret;
}

bool CommonBiaodashi::IsSelectAlias()
{
    ColumnShellPointer colsh;

    if (this->type == BiaodashiType::Lie)
    {
        colsh = boost::get<ColumnShellPointer>(this->content);

        if (colsh->GetExpr4Alias() != nullptr)
        {
            return true;
        }
    }
    return false;
}

BiaodashiPointer CommonBiaodashi::GetRealExprIfAlias()
{
    ColumnShellPointer colsh;

    if (this->type == BiaodashiType::Lie)
    {
        colsh = boost::get<ColumnShellPointer>(this->content);

        return colsh->GetExpr4Alias();
    }
    return nullptr;
}

LogicType CommonBiaodashi::GetLogicType4AndOr()
{
    assert(this->type == BiaodashiType::Andor);

    int my_content = boost::get<int>(this->GetContent());

    return my_content == ((int)LogicType::AND) ? LogicType::AND : LogicType::OR;
}

void CommonBiaodashi::ObtainReferenceTableInfo()
{
    ClearInvolvedTableList();
    ColumnShellPointer colsh;

    switch (this->type)
    {
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Star:
    case BiaodashiType::Decimal:
        break;

    case BiaodashiType::Lie:
        /*at this moment, the content should already be a ColumnShell*/
        /*todo: what if this is a refereneced column from outer table?*/
        colsh = boost::get<ColumnShellPointer>(this->content);
        if (this->expr_context->GetQueryContext()->query_level == colsh->GetAbsoluteLevel())
        {
            this->involved_table_list.push_back(colsh->GetTable());
        }
        else
        {
            /*this is a referenced column*/
            this->contain_outer = true;
        }

        break;

    case BiaodashiType::Hanshu:
    case BiaodashiType::SQLFunc:
    case BiaodashiType::Shuzu:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Andor:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Kuohao:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::Between:
    case BiaodashiType::Case:
    case BiaodashiType::IsNotNull:
    case BiaodashiType::IsNull:
    case BiaodashiType::ExprList:

        for (size_t i = 0; i < this->children.size(); i++)
        {
            if (this->children[i] == nullptr)
            {
                LOG( INFO ) << "a child is null!";
                continue;
            }

            CommonBiaodashi *a_child = (CommonBiaodashi *)((this->children[i]).get());

            a_child->ObtainReferenceTableInfo();

            if (a_child->GetContainOuter() == true)
            {
                this->SetContainOuter(true);
            }

            if (a_child->GetContainSubquery() == true)
            {
                this->SetContainSubquery(true);
            }

            std::vector<BasicRelPointer> child_involved_table_list =
                a_child->GetInvolvedTableList();

            this->involved_table_list.insert(this->involved_table_list.end(),
                                             child_involved_table_list.begin(),
                                             child_involved_table_list.end());
        }

        break;

    case BiaodashiType::Query:
        /*todo: find out those columns used by this subquery that belongs to this
         * level!!!*/
        if (1 > 0)
        {

            this->SetContainSubquery(true);

            QueryContextPointer my_query_context = this->expr_context->GetQueryContext();

            AbstractQueryPointer agp = boost::get<AbstractQueryPointer>(this->content);
            SelectStructure *raw_pointer = ((SelectStructure *)(agp.get()));
            QueryContextPointer subquery_context = raw_pointer->GetQueryContext();

            //		LOG(INFO) << "ObtainReferenceTableInfo: outer_column_array.size() =
            //"
            //<< subquery_context->outer_column_array.size() << "\n";

            for (size_t i = 0; i < subquery_context->outer_column_array.size(); i++)
            {
                /*a_o_c is a ColumnShell*/

                ColumnShellPointer a_o_c = subquery_context->outer_column_array[i];

                /*for debug*/
                //		    LOG(INFO) << "ObtainReferenceTableInfo: " <<
                // a_o_c->ToString()
                //<< "\n";

                if (a_o_c->GetAbsoluteLevel() == my_query_context->query_level)
                {
                    /*this column belongs to a table in current query level*/
                    this->involved_table_list.push_back(a_o_c->GetTable());
                }
            }
        }
        break;
    case BiaodashiType::IntervalExpression:
        break;

    default: {
        string msg = "ObtainReferenceTableInfo: unknown type: ";
        msg.append(get_name_of_expr_type(type));
        LOG(ERROR) << msg;
    }

    }
}

void CommonBiaodashi::ObtainReferenceTableAndOtherInfo( vector< AbstractQueryPointer >& subqueries, vector< ColumnShellPointer >& columns )
{
    ClearInvolvedTableList();
    ColumnShellPointer colsh;

    switch (this->type)
    {
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Star:
    case BiaodashiType::Decimal:
        break;

    case BiaodashiType::Lie:
        /*at this moment, the content should already be a ColumnShell*/
        /*todo: what if this is a refereneced column from outer table?*/
        colsh = boost::get<ColumnShellPointer>(this->content);
        if (this->expr_context->GetQueryContext()->query_level == colsh->GetAbsoluteLevel())
        {
            columns.push_back( colsh );
            this->involved_table_list.push_back(colsh->GetTable());
        }
        else
        {
            /*this is a referenced column*/
            this->contain_outer = true;
        }

        break;

    case BiaodashiType::Hanshu:
    case BiaodashiType::SQLFunc:
    case BiaodashiType::Shuzu:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Andor:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Kuohao:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::Between:
    case BiaodashiType::Case:
    case BiaodashiType::IsNotNull:
    case BiaodashiType::IsNull:
    case BiaodashiType::ExprList:

        for (size_t i = 0; i < this->children.size(); i++)
        {
            if (this->children[i] == nullptr)
            {
                LOG( INFO ) << "a child is null!";
                continue;
            }

            CommonBiaodashi *a_child = (CommonBiaodashi *)((this->children[i]).get());

            a_child->ObtainReferenceTableAndOtherInfo( subqueries, columns );

            if (a_child->GetContainOuter() == true)
            {
                this->SetContainOuter(true);
            }

            if (a_child->GetContainSubquery() == true)
            {
                this->SetContainSubquery(true);
            }

            std::vector<BasicRelPointer> child_involved_table_list =
                a_child->GetInvolvedTableList();

            this->involved_table_list.insert(this->involved_table_list.end(),
                                             child_involved_table_list.begin(),
                                             child_involved_table_list.end());
        }

        break;

    case BiaodashiType::Query:
        /*todo: find out those columns used by this subquery that belongs to this
         * level!!!*/
        if (1 > 0)
        {

            this->SetContainSubquery(true);

            QueryContextPointer my_query_context = this->expr_context->GetQueryContext();

            AbstractQueryPointer agp = boost::get<AbstractQueryPointer>(this->content);
            SelectStructure *raw_pointer = ((SelectStructure *)(agp.get()));
            QueryContextPointer subquery_context = raw_pointer->GetQueryContext();

            //      LOG(INFO) << "ObtainReferenceTableInfo: outer_column_array.size() =
            //"
            //<< subquery_context->outer_column_array.size() << "\n";

            for (size_t i = 0; i < subquery_context->outer_column_array.size(); i++)
            {
                /*a_o_c is a ColumnShell*/

                ColumnShellPointer a_o_c = subquery_context->outer_column_array[i];

                /*for debug*/
                //          LOG(INFO) << "ObtainReferenceTableInfo: " <<
                // a_o_c->ToString()
                //<< "\n";

                if (a_o_c->GetAbsoluteLevel() == my_query_context->query_level)
                {
                    /*this column belongs to a table in current query level*/
                    this->involved_table_list.push_back(a_o_c->GetTable());
                }
            }
            subqueries.push_back( agp );
        }
        break;
    case BiaodashiType::IntervalExpression:
        break;

    default: {
        string msg = "ObtainReferenceTableInfo: unknown type: ";
        msg.append(get_name_of_expr_type(type));
        LOG(ERROR) << msg;
    }

    }
}

void CommonBiaodashi::ResetReferencedColumnsInfo( const string& oldTableName, BasicRelPointer& newTable, int absoluteLevel )
{
    switch( this->type )
    {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Star:
        case BiaodashiType::Decimal:
            break;
        case BiaodashiType::Lie:
        {
            ColumnShellPointer col = boost::get< ColumnShellPointer >( this->content );
            if( col->GetTableName() == oldTableName )
            {
                ColumnShellPointer ret = std::make_shared< ColumnShell >( newTable->GetMyOutputName(), col->GetColumnName() );
                ret->SetTable( newTable );
                ret->SetColumnStructure( col->GetColumnStructure() );
                ret->SetLocationInTable( col->GetLocationInTable() );
                ret->SetAbsoluteLevel( absoluteLevel );
                ret->SetIsPrimaryKey( col->IsPrimaryKey() );
                ret->SetIsUnique( col->IsUnique() );
                this->content = ret;
            }
            break;
        }
        case BiaodashiType::Hanshu:
        case BiaodashiType::SQLFunc:
        case BiaodashiType::Shuzu:
        case BiaodashiType::Yunsuan:
        case BiaodashiType::Bijiao:
        case BiaodashiType::Andor:
        case BiaodashiType::Qiufan:
        case BiaodashiType::Kuohao:
        case BiaodashiType::Cunzai:
        case BiaodashiType::Likeop:
        case BiaodashiType::Inop:
        case BiaodashiType::NotIn:
        case BiaodashiType::Between:
        case BiaodashiType::Case:
        case BiaodashiType::IsNotNull:
        case BiaodashiType::IsNull:
         case BiaodashiType::ExprList:
        {
            for( size_t i = 0; i < this->children.size(); i++ )
            {
                if( this->children[i] == nullptr )
                {
                    LOG( INFO ) << "a child is null!";
                    continue;
                }

                CommonBiaodashi *a_child = ( CommonBiaodashi * )( ( this->children[i] ).get() );
                a_child->ResetReferencedColumnsInfo( oldTableName, newTable, absoluteLevel );
            }
            break;
        }
        case BiaodashiType::Query:
            break;
        default:
        {
            string msg = "ObtainReferenceTableInfo: unknown type: ";
            msg.append( get_name_of_expr_type( type ) );
            LOG(ERROR)<< msg;
        }
    }
}

void CommonBiaodashi::ReplaceReferencedColumns( const ColumnShellPointer& oldCol, const ColumnShellPointer& newCol )
{
    switch( this->type )
    {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Fudianshu:
        case BiaodashiType::Zhenjia:
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Star:
        case BiaodashiType::Decimal:
            break;
        case BiaodashiType::Lie:
        {
            ColumnShellPointer col = boost::get< ColumnShellPointer >( this->content );
            if( col->GetTableName() == oldCol->GetTableName() && col->GetColumnName() == oldCol->GetColumnName() )
                this->content = newCol;
            break;
        }
        case BiaodashiType::Hanshu:
        case BiaodashiType::SQLFunc:
        case BiaodashiType::Shuzu:
        case BiaodashiType::Yunsuan:
        case BiaodashiType::Bijiao:
        case BiaodashiType::Andor:
        case BiaodashiType::Qiufan:
        case BiaodashiType::Kuohao:
        case BiaodashiType::Cunzai:
        case BiaodashiType::Likeop:
        case BiaodashiType::Inop:
        case BiaodashiType::NotIn:
        case BiaodashiType::Between:
        case BiaodashiType::Case:
        case BiaodashiType::IsNotNull:
        case BiaodashiType::IsNull:
         case BiaodashiType::ExprList:
        {
            for( size_t i = 0; i < this->children.size(); i++ )
            {
                if( this->children[i] == nullptr )
                {
                    LOG( INFO ) << "a child is null!";
                    continue;
                }

                CommonBiaodashi *a_child = ( CommonBiaodashi * )( ( this->children[i] ).get() );
                a_child->ReplaceReferencedColumns( oldCol, newCol );
            }
            break;
        }
        case BiaodashiType::Query:
            break;
        default:
        {
            string msg = "ObtainReferenceTableInfo: unknown type: ";
            msg.append( get_name_of_expr_type( type ) );
            LOG(ERROR)<< msg;
        }
    }
}

void CommonBiaodashi::SetContainOuter(bool arg_value)
{
    this->contain_outer = arg_value;
}

void CommonBiaodashi::SetContainSubquery(bool arg_value)
{
    this->contain_subquery = arg_value;
}

void CommonBiaodashi::ClearInvolvedTableList()
{
    involved_table_list.clear();
}

bool CommonBiaodashi::GetContainOuter() { return this->contain_outer; }

bool CommonBiaodashi::GetContainSubquery() { return this->contain_subquery; }

std::vector<BasicRelPointer> CommonBiaodashi::GetInvolvedTableList()
{
    return this->involved_table_list;
}

/*true of false*/
bool CommonBiaodashi::GetMyBoolValue()
{
    assert(this->type == BiaodashiType::Zhenjia);
    return boost::get<bool>(this->content);
}

std::vector<ColumnShellPointer> CommonBiaodashi::GetAllReferencedColumns()
{
    std::vector<ColumnShellPointer> ret;

    switch (this->type)
    {
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Zifuchuan:
    case BiaodashiType::Null:

    case BiaodashiType::Star:
    case BiaodashiType::Hanshu:
    case BiaodashiType::Shuzu:
    case BiaodashiType::Decimal:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Andor:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Kuohao:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::Between:
    case BiaodashiType::Case:
    case BiaodashiType::SQLFunc:
    case BiaodashiType::IntervalExpression:
    case BiaodashiType::IsNotNull:
    case BiaodashiType::IsNull:
    case BiaodashiType::ExprList:
        if (1 > 0)
        {
            for (size_t ci = 0; ci < this->children.size(); ci++)
            {
                if (this->children[ci] == nullptr)
                {
                    continue;
                }

                CommonBiaodashi *a_child =
                    ((CommonBiaodashi *)((this->children[ci]).get()));

                std::vector<ColumnShellPointer> a_child_columns =
                    a_child->GetAllReferencedColumns();

                ret.insert(ret.end(), a_child_columns.begin(), a_child_columns.end());
            }
        }

        break;

    case BiaodashiType::Lie:
        /*todo: consider level here!*/
        if (1 > 0)
        {
            ColumnShellPointer csp = boost::get<ColumnShellPointer>(this->content);

            if (csp->GetAbsoluteLevel() ==
                this->GetExprContext()->GetQueryContext()->query_level)
            {
                ret.push_back(csp);
            }
        }
        break;

    case BiaodashiType::Query:
        if (1 > 0)
        {
            AbstractQueryPointer agp = boost::get<AbstractQueryPointer>(this->content);
            SelectStructure *the_ss = (SelectStructure *)(agp.get());

            std::vector<ColumnShellPointer> v_csp =
                the_ss->GetQueryContext()->outer_column_array;

            for (size_t i = 0; i < v_csp.size(); i++)
            {
                ColumnShellPointer csp = v_csp[i];
                if (csp->GetAbsoluteLevel() ==
                    this->GetExprContext()->GetQueryContext()->query_level)
                {
                    ret.push_back(csp);
                }
            }
        }
        break;
    default:
        string msg("CommonBiaodashi::GetAllReferencedColumn: unknown type: ");
        msg.append(get_name_of_expr_type(type));
        ARIES_EXCEPTION_SIMPLE(ER_SYNTAX_ERROR, msg);
    }

    return ret;
}

/*this function is only for subquery unnesting!*/
std::vector<ColumnShellPointer> CommonBiaodashi::GetAllReferencedColumns_NoQuery()
{
    std::vector<ColumnShellPointer> ret;

    switch (this->type)
    {
    case BiaodashiType::Zhengshu:
    case BiaodashiType::Fudianshu:
    case BiaodashiType::Zhenjia:
    case BiaodashiType::Zifuchuan:

    case BiaodashiType::Star:
    case BiaodashiType::Hanshu:
    case BiaodashiType::Shuzu:
    case BiaodashiType::Yunsuan:
    case BiaodashiType::Bijiao:
    case BiaodashiType::Andor:
    case BiaodashiType::Qiufan:
    case BiaodashiType::Kuohao:
    case BiaodashiType::Cunzai:
    case BiaodashiType::Likeop:
    case BiaodashiType::Inop:
    case BiaodashiType::NotIn:
    case BiaodashiType::Between:
    case BiaodashiType::Case:
    case BiaodashiType::SQLFunc:
    case BiaodashiType::IntervalExpression:
    case BiaodashiType::ExprList:
        if (1 > 0)
        {
            for (size_t ci = 0; ci < this->children.size(); ci++)
            {
                if (this->children[ci] == nullptr)
                {
                    continue;
                }

                CommonBiaodashi *a_child =
                    ((CommonBiaodashi *)((this->children[ci]).get()));

                std::vector<ColumnShellPointer> a_child_columns =
                    a_child->GetAllReferencedColumns();

                ret.insert(ret.end(), a_child_columns.begin(), a_child_columns.end());
            }
        }

        break;

    case BiaodashiType::Lie:
        /*todo: consider level here!*/
        if (1 > 0)
        {
            ColumnShellPointer csp = boost::get<ColumnShellPointer>(this->content);

            if (csp->GetAbsoluteLevel() ==
                this->GetExprContext()->GetQueryContext()->query_level)
            {
                ret.push_back(csp);
            }
        }
        break;

    case BiaodashiType::Query:
        if (1 > 0)
        {
            LOG(ERROR) << "CommonBiaodashi::GetAllReferencedColumn_NoQuery: WE FOUND QUERY!!!";
        }
        break;

    default:
        LOG(ERROR) << "CommonBiaodashi::GetAllReferencedColumn: unknown type: " +
                      std::to_string(int(this->type));
    }

    return ret;
}

void CommonBiaodashi::ResetChildByIndex(int arg_index, BiaodashiPointer arg_bp)
{
    // to_remove.push_back( this->children[arg_index] );
    SetChild( arg_index, arg_bp );
}

/*
void CommonBiaodashi::ConvertSelfToFloat(double arg_value)
{

    if (this->type == BiaodashiType::Zifuchuan)
    {
        // LOG(INFO) << "I am a String, " << this->ContentToString() << "\n";
    }

    this->type = BiaodashiType::Fudianshu;
    this->content = arg_value;
    this->children.empty();

    this->value_type = BiaodashiValueType::FLOAT;

    // LOG(INFO) << "i have been become a float!\n";
}
*/

void CommonBiaodashi::ConvertSelfToDecimal( const std::string& arg_value )
{
    if (arg_value.empty())
    {
        type = BiaodashiType::Null;
    }
    else
    {
        type = BiaodashiType::Decimal;
    }

    content = arg_value;
    ClearChildren();
    value_type = BiaodashiValueType::DECIMAL;
}

void CommonBiaodashi::ConvertSelfToString( const std::string& arg_value)
{
    type = BiaodashiType::Zifuchuan;

    content = arg_value;
    ClearChildren();
    value_type = BiaodashiValueType::TEXT;
}

void CommonBiaodashi::ConvertSelfToNull()
{
    type = BiaodashiType::Null;

    content = 0;
    ClearChildren();
    value_type = BiaodashiValueType::UNKNOWN;
}

void CommonBiaodashi::ConvertSelfToBuffer(const aries_acc::AriesDataBufferSPtr& buffer) {
    buffer_ptr = buffer;
    type = BiaodashiType::Buffer;

    ClearChildren();
    value_type = BiaodashiValueType::LIST;
}

bool CommonBiaodashi::IsNullable() { return nullable; }

void CommonBiaodashi::SetIsNullable(bool is_nullable) {
    nullable = is_nullable;
}

std::int32_t CommonBiaodashi::GetLength() { return length; }

void CommonBiaodashi::SetLength(std::int32_t length) { this->length = length; }

void CommonBiaodashi::SetAssociatedLength(int length) { this->associated_length = length; }

int CommonBiaodashi::GetAssociatedLength() const { return associated_length; }

BiaodashiValueType CommonBiaodashi::GetPromotedValueType()
{
    if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetDouble)
    {
        return BiaodashiValueType::DOUBLE;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetFloat)
    {
        return BiaodashiValueType::FLOAT;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetDecimal)
    {
        return BiaodashiValueType::DECIMAL;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetInt64)
    {
        return BiaodashiValueType::LONG_INT;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetInt32)
    {
        return BiaodashiValueType::INT;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetInt16)
    {
        return BiaodashiValueType::SMALL_INT;
    }
    else if (numeric_type_value & CommonBiaodashi::NumericTypeOffsetInt8)
    {
        return BiaodashiValueType::TINY_INT;
    }
    else
    {
        return value_type;
    }
}

bool CommonBiaodashi::IsCalculable() {
    return is_calculable;
}

std::shared_ptr<CommonBiaodashi> CommonBiaodashi::Clone() {
    auto new_one = std::make_shared<CommonBiaodashi>(type, content);
    new_one->value_type = value_type;
    new_one->numeric_type_value =  numeric_type_value;
    new_one->nullable = nullable;
    new_one->associated_length = associated_length;
    new_one->SetParent( GetParent() );

    for (const auto& child : children) {
        if (child) {
            auto child_expr = std::dynamic_pointer_cast<CommonBiaodashi>(child);
            new_one->AddChild(child_expr->Clone());
        } else {
            new_one->AddChild(child);
        }
    }

    return new_one;
}

std::shared_ptr<CommonBiaodashi> CommonBiaodashi::CloneUsingNewExprContext( ExprContextPointer context )
{
    auto new_one = std::make_shared< CommonBiaodashi >( type, content );
    new_one->value_type = value_type;
    new_one->numeric_type_value = numeric_type_value;
    new_one->nullable = nullable;
    new_one->associated_length = associated_length;
    new_one->SetExprContext( context );
    new_one->SetParent( GetParent() );
    for( const auto& child : children )
    {
        if( child )
        {
            auto child_expr = std::dynamic_pointer_cast< CommonBiaodashi >( child );
            new_one->AddChild( child_expr->CloneUsingNewExprContext( context ) );
        }
        else
        {
            new_one->AddChild( child );
        }
    }

    return new_one;
}

void CommonBiaodashi::Clone(const std::shared_ptr<CommonBiaodashi>& other) {
    SetContent(other->GetContent());
    value_type = other->value_type;
    type = other->type;
    numeric_type_value =  other->numeric_type_value;
    nullable = other->nullable;
    associated_length = other->associated_length;
    SetParent( other->GetParent() );

    ClearChildren();
    for (const auto& child : other->children) {
        if (child) {
            auto child_expr = std::dynamic_pointer_cast<CommonBiaodashi>(child);
            AddChild(child_expr->Clone());
        } else {
            AddChild(child);
        }
    }
}

void CommonBiaodashi::ClearChildren() {
    for ( auto& child : children )
    {
        if ( child )
            child->SetParent( nullptr );
    }
    children.clear();
}

bool CommonBiaodashi::IsVisibleInResult() {
    return is_visible_in_result;
}

void CommonBiaodashi::SetIsVisibleInResult(bool value) {
    is_visible_in_result = value;
}

void CommonBiaodashi::SetNeedShowAlias(bool value) {
    need_show_alias = value;
}

bool CommonBiaodashi::NeedShowAlias() {
    return need_show_alias;
}

bool CommonBiaodashi::IsSameAs(CommonBiaodashi* other) {
    if (type != other->type) {
        return false;
    }

    if (content.which() != other->content.which()) {
        return false;
    }

    //typedef boost::variant<bool, int, double, std::string, AbstractQueryPointer,
    //                   ColumnShellPointer, SQLFunctionPointer, sys_var*>

    if (CHECK_VARIANT_TYPE(content, bool)) {
        auto left = boost::get<bool>(content);
        auto right = boost::get<bool>(other->content);
        if (left != right) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, int)) {
        auto left = boost::get<int>(content);
        auto right = boost::get<int>(other->content);
        if (left != right) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, double)) {
        auto left = boost::get<double>(content);
        auto right = boost::get<double>(other->content);
        if (left != right) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, std::string)) {
        auto left = boost::get<std::string>(content);
        auto right = boost::get<std::string>(other->content);
        aries_utils::to_upper(left);
        aries_utils::to_upper(right);
        if (left != right) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, ColumnShellPointer)) {
        auto left = boost::get<ColumnShellPointer>(content);
        auto right = boost::get<ColumnShellPointer>(other->content);
        if (left->GetTableName() != right->GetTableName() || left->GetColumnName() != right->GetColumnName()) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, SQLFunctionPointer)) {
        auto left = boost::get<SQLFunctionPointer>(content);
        auto right = boost::get<SQLFunctionPointer>(other->content);
        if (left->GetType() != right->GetType()) {
            return false;
        }
    } else if (CHECK_VARIANT_TYPE(content, SQLIdentPtr)) {
        auto left = boost::get< SQLIdentPtr >( content );
        auto right = boost::get< SQLIdentPtr >( other->content );

        if ( left->db != right->db || left->table != right->table || left->id != right->id )
        {
            return false;
        }
    } else {
        LOG( INFO ) << "unhandled content type: " + std::to_string(content.which());
    }

    if (children.size() != other->children.size()) {
        return false;
    }

    for (size_t i = 0; i < children.size(); i ++) {
        auto child = (CommonBiaodashi*) (children[i].get());
        if (child == nullptr && other->children[i].get() == nullptr) {
            continue;
        } else if (child == nullptr || other->children[i].get() == nullptr) {
            return false;
        }
        if (!child->IsSameAs((CommonBiaodashi*) (other->children[i].get()))) {
            return false;
        }
    }
    return true;
}

void CommonBiaodashi::SetExpectBuffer(bool value) {
    expect_buffer = value;
}

bool CommonBiaodashi::IsExpectBuffer() {
    return expect_buffer;
}

bool CommonBiaodashi::IsAggFunction() const {
    if (type == BiaodashiType::SQLFunc) {
        auto function = boost::get<SQLFunctionPointer>(content);
        return function->GetIsAggFunc();
    } else if (type == BiaodashiType::Hanshu) {
        auto function_name = boost::get<std::string>(content);
        auto function = std::make_shared<SQLFunction>(function_name);
        return function->GetIsAggFunc();
    }
    return false;
}

bool CommonBiaodashi::ContainsAggFunction() const {
    if( IsAggFunction() )
        return true;
    for (const auto& child : children) {
        if (!child) {
            continue;
        }

        auto child_expr = std::dynamic_pointer_cast<CommonBiaodashi>(child);
        if (child_expr->IsAggFunction()) {
            return true;
        }

        if (child_expr->ContainsAggFunction()) {
            return true;
        }
    }
    return false;
}

bool CommonBiaodashi::CouldBeCompactDecimal() const {
    switch (type) {
        case BiaodashiType::Zhengshu:
        case BiaodashiType::Decimal: {
            return true;
        }
        default: {
            switch ( value_type )
            {
                case BiaodashiValueType::DECIMAL:
                    return associated_length != -1;
                case BiaodashiValueType::INT:
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::LONG_INT:
                case BiaodashiValueType::TINY_INT:
                    return true;
                default:
                    break;
            }
            return false;
        }
    }
}

bool CommonBiaodashi::IsEqualCondition() const
{
    if ( type != BiaodashiType::Bijiao )
    {
        return false;
    }

    if ( boost::get< int >( content ) != static_cast< int >( ComparisonType::DengYu ) )
    {
        return false;
    }

    return true;
}

bool CommonBiaodashi::IsTrueConstant() const
{
    if ( type != BiaodashiType::Zhenjia )
    {
        return false;
    }

    return boost::get< bool >( content );
}

void CommonBiaodashi::SwitchChild()
{
    auto left = children[ 0 ];
    auto right = children[ 1 ];

    children[ 0 ] = right;
    children[ 1 ] = left;
}

aries_acc::AriesDataBufferSPtr CommonBiaodashi::GetBuffer() {
    return buffer_ptr;
}

std::shared_ptr<CommonBiaodashi> CommonBiaodashi::Normalized()
{
    auto normalized = Clone();

    if ( children.size() == 1 && children[ 0 ] )
    {
        auto child = std::dynamic_pointer_cast< CommonBiaodashi >( children[ 0 ] )->Normalized();
        normalized->SetChild( 0, child );
    }
    else if ( children.size() > 1 )
    {
        for ( size_t i = 0; i < normalized->children.size(); i++ )
        {
            if ( !normalized->children[ i ] )
            {
                continue;
            }
            normalized->SetChild( i, std::dynamic_pointer_cast< CommonBiaodashi >( normalized->children[ i ] )->Normalized() );
        }
        if ( type == BiaodashiType::Bijiao )
        {
            auto comparison_type = static_cast< ComparisonType >( boost::get< int >( normalized->content ) );
            auto left = std::dynamic_pointer_cast< CommonBiaodashi >( normalized->children[ 0 ] );
            auto right = std::dynamic_pointer_cast< CommonBiaodashi >( normalized->children[ 1 ] );

            switch ( comparison_type )
            {
                case ComparisonType::DengYu:
                case ComparisonType::BuDengYu:
                case ComparisonType::SQLBuDengYu:
                {
                    if ( *left < *right )
                    {
                        normalized->SetChild( 0, right );
                        normalized->SetChild( 1, left );
                    }
                    break;
                }
                case ComparisonType::DaYu:
                {
                    normalized->SetChild( 0, right );
                    normalized->SetChild( 1, left );
                    normalized->content = static_cast< int >( ComparisonType::XiaoYu );
                    break;
                }
                case ComparisonType::DaYuDengYu:
                {
                    normalized->SetChild( 0, right );
                    normalized->SetChild( 1, left );
                    normalized->content = static_cast< int >( ComparisonType::XiaoYuDengYu );
                    break;
                }
                default:
                    break;
            }
        }

        if ( type == BiaodashiType::Andor )
        {
            BiaodashiAuxProcessor processor;
            auto logic_type = static_cast< LogicType >( boost::get< int >( normalized->content ) );

            if ( logic_type == LogicType::AND )
            {
                std::vector< BiaodashiPointer > list = processor.generate_and_list( normalized->Clone() );

                if ( list.size() > 1 )
                {
                    std::sort( list.begin(), list.end(), [=]( const BiaodashiPointer& l, const BiaodashiPointer& r ) {
                        return *( ( CommonBiaodashi* )( l.get() ) ) < *( ( CommonBiaodashi* )( r.get() ) );
                    });

                    auto new_expr = processor.make_biaodashi_from_and_list( list );
                    auto new_children = std::dynamic_pointer_cast< CommonBiaodashi >( new_expr )->children;
                    normalized->SetChildren( new_children );
                }
            }
            else
            {
                std::vector< BiaodashiPointer > list;
                processor.generate_or_list( normalized->Clone(), list );

                if ( list.size() > 1 )
                {
                    std::sort( list.begin(), list.end(), [=]( const BiaodashiPointer& l, const BiaodashiPointer& r ) {
                        return *( ( CommonBiaodashi* )( l.get() ) ) < *( ( CommonBiaodashi* )( r.get() ) );
                    });

                    auto new_expr = processor.make_biaodashi_from_and_list( list );
                    auto new_children = std::dynamic_pointer_cast< CommonBiaodashi >( new_expr )->children;
                    normalized->SetChildren( new_children );
                }
            }

        }
    }

    return normalized;
}

void CommonBiaodashi::ReplaceBiaoshifu( const vector< CommonBiaodashi* >& exprsToCheck )
{
    assert( !exprsToCheck.empty() && type != BiaodashiType::Biaoshifu );
    for( size_t i = 0; i < children.size(); ++i )
    {
        auto child = std::dynamic_pointer_cast< CommonBiaodashi >( children[ i ] );
        if( child )
        {
            if( child->type == BiaodashiType::Biaoshifu )
            {
                for( auto& expr : exprsToCheck )
                {
                    if( child->origName == expr->GetOrigName() )
                    {
                        SetChild( i, expr->Clone() );
                        break;
                    }
                }
            }
            else
            {
                child->ReplaceBiaoshifu( exprsToCheck );
            }
        }
    }
}

int compare( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    if ( l.type < r.type )
    {
        return -1;
    }
    else if ( l.type > r.type )
    {
        return 1;
    }

    if ( l.children.size() < r.children.size() )
    {
        return -1;
    }
    else if ( l.children.size() > r.children.size() )
    {
        return 1;
    }

    switch ( l.type )
    {
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Decimal:
        {
            auto l_value = boost::get< std::string>( l.content );
            auto r_value = boost::get< std::string>( r.content );
            if ( l_value == r_value )
            {
                return 0;
            }
            else if ( l_value < r_value )
            {
                return -1;
            }
            else
            {
                return 1;
            }
        }
        case BiaodashiType::Zhengshu:
        {
            long l_value, r_value;
            switch ( l.value_type )
            {
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::TINY_INT:
                case BiaodashiValueType::INT:
                    l_value = boost::get< int >( l.content );
                    break;
                default:
                    l_value = boost::get< int64_t >( l.content );
                    break;
            }

            switch ( r.value_type )
            {
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::TINY_INT:
                case BiaodashiValueType::INT:
                    r_value = boost::get< int >( r.content );
                    break;
                default:
                    r_value = boost::get< int64_t >( r.content );
                    break;
            }

            if ( l_value < r_value )
            {
                return -1;
            }
            else if ( l_value > r_value )
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        case BiaodashiType::Lie:
        {
            auto l_column = boost::get< ColumnShellPointer >( l.content );
            auto r_column = boost::get< ColumnShellPointer >( r.content );

            auto l_alias = l_column->GetExpr4Alias();
            auto r_alias = r_column->GetExpr4Alias();

            if ( l_alias || r_alias )
            {
                auto l_expr = std::dynamic_pointer_cast< CommonBiaodashi >( l_alias );
                auto r_expr = std::dynamic_pointer_cast< CommonBiaodashi >( r_alias );
                if ( l_expr && r_expr )
                {
                    return compare( *l_expr, *r_expr );
                }
                else if ( l_expr )
                {
                    return compare( *l_expr, r );
                }
                else
                {
                    return compare( l, *r_expr );
                }
            }

            auto l_table = l_column->GetTable();
            auto r_table = r_column->GetTable();

            if ( l_table->GetDb() < r_table->GetDb() )
            {
                return -1;
            }
            else if ( l_table->GetDb() > r_table->GetDb() )
            {
                return 1;
            }

            if ( l_table->GetID() < r_table->GetID() )
            {
                return -1;
            }
            else if( l_table->GetID() > r_table->GetID() )
            {
                return 1;
            }

            if ( l_column->GetColumnName() < r_column->GetColumnName() )
            {
                return -1;
            }
            else if ( l_column->GetColumnName() > r_column->GetColumnName() )
            {
                return 1;
            }

            return 0;
        }
        default:
            if( l.children.size() > 0 )
                break;
            else 
                return -1;
    }

    for ( size_t i = 0; i < l.children.size(); i++ )
    {
        auto l_child = std::dynamic_pointer_cast< CommonBiaodashi >( l.children[ i ] );
        auto r_child = std::dynamic_pointer_cast< CommonBiaodashi >( r.children[ i ] );
        if ( l_child == r_child )
        {
            continue;
        }

        if ( !l_child )
        {
            return -1;
        }

        if ( !r_child )
        {
            return 1;
        }

        auto ret = compare( *l_child, *r_child );
        if ( ret != 0 )
        {
            return ret;
        }
    }

    return 0;
}

bool operator>( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    return compare( l, r ) == 1;
}

bool operator<( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    return compare( l, r ) == -1;
}

bool operator==( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    return compare( l, r ) == 0;
}

bool operator!=( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    return compare( l, r ) != 0;
}

// for container comparator
bool kv_compare( const CommonBiaodashi& l, const CommonBiaodashi& r )
{
    if ( l.type < r.type )
    {
        return true;
    }
    else if ( l.type > r.type )
    {
        return false;
    }

    if ( l.children.size() < r.children.size() )
    {
        return true;
    }
    else if ( l.children.size() > r.children.size() )
    {
        return false;
    }

    switch ( l.type )
    {
        case BiaodashiType::Zifuchuan:
        case BiaodashiType::Decimal:
        {
            auto l_value = boost::get< std::string>( l.content );
            auto r_value = boost::get< std::string>( r.content );
            if ( l_value == r_value )
            {
                return false;
            }
            else if ( l_value < r_value )
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        case BiaodashiType::Zhengshu:
        {
            long l_value, r_value;
            switch ( l.value_type )
            {
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::TINY_INT:
                case BiaodashiValueType::INT:
                    l_value = boost::get< int >( l.content );
                    break;
                default:
                    l_value = boost::get< int64_t >( l.content );
                    break;
            }

            switch ( r.value_type )
            {
                case BiaodashiValueType::SMALL_INT:
                case BiaodashiValueType::TINY_INT:
                case BiaodashiValueType::INT:
                    r_value = boost::get< int >( r.content );
                    break;
                default:
                    r_value = boost::get< int64_t >( r.content );
                    break;
            }

            if ( l_value < r_value )
            {
                return true;
            }
            else if ( l_value > r_value )
            {
                return false;
            }
            else
            {
                return false;
            }
        }
        case BiaodashiType::Lie:
        {
            auto l_column = boost::get< ColumnShellPointer >( l.content );
            auto r_column = boost::get< ColumnShellPointer >( r.content );

            auto l_alias = l_column->GetExpr4Alias();
            auto r_alias = r_column->GetExpr4Alias();

            if ( l_alias || r_alias )
            {
                auto l_expr = std::dynamic_pointer_cast< CommonBiaodashi >( l_alias );
                auto r_expr = std::dynamic_pointer_cast< CommonBiaodashi >( r_alias );
                if ( l_expr && r_expr )
                {
                    return kv_compare( *l_expr, *r_expr );
                }
                else if ( l_expr )
                {
                    return kv_compare( *l_expr, r );
                }
                else
                {
                    return kv_compare( l, *r_expr );
                }
            }

            auto l_table = l_column->GetTable();
            auto r_table = r_column->GetTable();

            if ( l_table->GetDb() < r_table->GetDb() )
            {
                return true;
            }
            else if ( l_table->GetDb() > r_table->GetDb() )
            {
                return false;
            }

            if ( l_table->GetID() < r_table->GetID() )
            {
                return true;
            }
            else if( l_table->GetID() > r_table->GetID() )
            {
                return false;
            }

            if ( l_column->GetColumnName() < r_column->GetColumnName() )
            {
                return true;
            }
            else if ( l_column->GetColumnName() > r_column->GetColumnName() )
            {
                return false;
            }

            return false;
        }
        default:
            if( l.children.size() > 0 )
                break;
            else 
                return true;
    }

    for ( size_t i = 0; i < l.children.size(); i++ )
    {
        auto l_child = std::dynamic_pointer_cast< CommonBiaodashi >( l.children[ i ] );
        auto r_child = std::dynamic_pointer_cast< CommonBiaodashi >( r.children[ i ] );
        if ( l_child == r_child )
        {
            continue;
        }

        if ( !l_child )
        {
            return true;
        }

        if ( !r_child )
        {
            return false;
        }

        auto ret = kv_compare( *l_child, *r_child );
        if ( ret )
        {
            return ret;
        }
    }

    return false;
}

} // namespace aries
