#include <iostream>
#include <memory>

#include "AriesAssert.h"
#include "ColumnEntry.h"
#include "TableEntry.h"
#include "datatypes/decimal.hxx"
#include "datatypes/AriesDatetimeTrans.h"
#include "utils/string_util.h"

#include "Compression/dict/AriesDict.h"

using namespace std;

namespace aries {

namespace schema
{

const std::string DEFAULT_VAL_NOW = "CURRENT_TIMESTAMP";
class TableEntry;

static std::map<ColumnType, string> DataTypeStringMap = {
        {ColumnType::TINY_INT,"tinyint"},
        {ColumnType::SMALL_INT, "smallint"},
        {ColumnType::INT, "int"},
        {ColumnType::LONG_INT, "bigint"},
        {ColumnType::DECIMAL, "decimal"},
        {ColumnType::FLOAT, "float"},
        {ColumnType::DOUBLE, "double"},
        {ColumnType::TEXT, "char"},
        {ColumnType::BOOL, "bool"},
        {ColumnType::DATE, "date"},
        {ColumnType::TIME, "time"},
        {ColumnType::DATE_TIME, "datetime"},
        {ColumnType::TIMESTAMP, "timestamp"},
        {ColumnType::UNKNOWN, "NULL"},
        {ColumnType::YEAR, "year"},
        {ColumnType::LIST ,"list"},
        {ColumnType::BINARY, "binary"},
        {ColumnType::VARBINARY, "varbinary"}
};
string DataTypeString ( ColumnType columnType ) {
    auto it = DataTypeStringMap.find( columnType );
    if (DataTypeStringMap.end() != it) {
        return it->second;
    }
    return "UNKNOWN";
}
string ColumnTypeString (const ColumnEntryPtr& column) {
    string type = DataTypeString( column->GetType() );
    ColumnType columnType = column->GetType();
    switch (columnType) {
        case ColumnType::TINY_INT:
        case ColumnType::SMALL_INT:
        case ColumnType::INT:
        case ColumnType::LONG_INT:
            type.append("(");
            type.append(std::to_string( column->display_width ));
            type.append(")");
            break;
        case ColumnType::DECIMAL:
            type.append("(");
            type.append(std::to_string(column->numeric_precision)).append(",").append(std::to_string(column->numeric_scale));
            type.append(")");
            break;
        case ColumnType::FLOAT:
        case ColumnType::DOUBLE:
            if ( -1 != column->numeric_scale )
            {
                type.append("(");
                type.append(std::to_string(column->display_width)).append(",").append(std::to_string(column->numeric_scale));
                type.append(")");
            }
            break;
        case ColumnType::TEXT:
        case ColumnType::BINARY:
        case ColumnType::VARBINARY:
            type.append("(");
            type.append(std::to_string(column->char_max_len));
            type.append(")");
            break;
        case ColumnType::BOOL:
            break;
        case ColumnType::DATE:
            break;
        case ColumnType::TIME:
            break;
        case ColumnType::DATE_TIME:
            break;
        case ColumnType::TIMESTAMP:
            break;
        case ColumnType::UNKNOWN:
            break;
        case ColumnType::YEAR:
            type.append("(4)");
            break;
        case ColumnType::LIST:
            break;
        default:
            break;
    }
    if ( column->IsUnsigned() )
    {
        type.append( " unsigned" );
    }
    return type;
}

/*
 * mysql server 5.7.26
 * mysql> create table db1.t_int_nake(i1 tinyint, i2 tinyint unsigned, i3 smallint, i4 smallint unsigned, i5 int, i6 int unsigned, i7 bigint, i8 bigint unsigned );
Query OK, 0 rows affected (0.04 sec)

mysql> desc db1.t_int_nake;
+-------+----------------------+------+-----+---------+-------+
| Field | Type                 | Null | Key | Default | Extra |
+-------+----------------------+------+-----+---------+-------+
| i1    | tinyint(4)           | YES  |     | NULL    |       |
| i2    | tinyint(3) unsigned  | YES  |     | NULL    |       |
| i3    | smallint(6)          | YES  |     | NULL    |       |
| i4    | smallint(5) unsigned | YES  |     | NULL    |       |
| i5    | int(11)              | YES  |     | NULL    |       |
| i6    | int(10) unsigned     | YES  |     | NULL    |       |
| i7    | bigint(20)           | YES  |     | NULL    |       |
| i8    | bigint(20) unsigned  | YES  |     | NULL    |       |
+-------+----------------------+------+-----+---------+-------+
8 rows in set (0.01 sec)

mysql> show create table db1.t_int_nake;
+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Table      | Create Table                                                                                                                                                                                                                                                                                                                                                      |
+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| t_int_nake | CREATE TABLE `t_int_nake` (
  `i1` tinyint(4) DEFAULT NULL,
  `i2` tinyint(3) unsigned DEFAULT NULL,
  `i3` smallint(6) DEFAULT NULL,
  `i4` smallint(5) unsigned DEFAULT NULL,
  `i5` int(11) DEFAULT NULL,
  `i6` int(10) unsigned DEFAULT NULL,
  `i7` bigint(20) DEFAULT NULL,
  `i8` bigint(20) unsigned DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 |
+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
1 row in set (0.00 sec)

mysql> select * from columns where table_name='t_int_nake';
+---------------+--------------+------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
| TABLE_CATALOG | TABLE_SCHEMA | TABLE_NAME | COLUMN_NAME | ORDINAL_POSITION | COLUMN_DEFAULT | IS_NULLABLE | DATA_TYPE | CHARACTER_MAXIMUM_LENGTH | CHARACTER_OCTET_LENGTH | NUMERIC_PRECISION | NUMERIC_SCALE | DATETIME_PRECISION | CHARACTER_SET_NAME | COLLATION_NAME | COLUMN_TYPE          | COLUMN_KEY | EXTRA | PRIVILEGES                      | COLUMN_COMMENT | GENERATION_EXPRESSION |
+---------------+--------------+------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
| def           | db1          | t_int_nake | i1          |                1 | NULL           | YES         | tinyint   |                     NULL |                   NULL |                 3 |             0 |               NULL | NULL               | NULL           | tinyint(4)           |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i2          |                2 | NULL           | YES         | tinyint   |                     NULL |                   NULL |                 3 |             0 |               NULL | NULL               | NULL           | tinyint(3) unsigned  |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i3          |                3 | NULL           | YES         | smallint  |                     NULL |                   NULL |                 5 |             0 |               NULL | NULL               | NULL           | smallint(6)          |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i4          |                4 | NULL           | YES         | smallint  |                     NULL |                   NULL |                 5 |             0 |               NULL | NULL               | NULL           | smallint(5) unsigned |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i5          |                5 | NULL           | YES         | int       |                     NULL |                   NULL |                10 |             0 |               NULL | NULL               | NULL           | int(11)              |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i6          |                6 | NULL           | YES         | int       |                     NULL |                   NULL |                10 |             0 |               NULL | NULL               | NULL           | int(10) unsigned     |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i7          |                7 | NULL           | YES         | bigint    |                     NULL |                   NULL |                19 |             0 |               NULL | NULL               | NULL           | bigint(20)           |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_nake | i8          |                8 | NULL           | YES         | bigint    |                     NULL |                   NULL |                20 |             0 |               NULL | NULL               | NULL           | bigint(20) unsigned  |            |       | select,insert,update,references |                |                       |
+---------------+--------------+------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
8 rows in set (0.01 sec)

 mysql> create table db1.t_int_width(i1 tinyint(2), i2 tinyint(2) unsigned, i3 smallint(3), i4 smallint(3) unsigned, i5 int(8), i6 int(8) unsigned, i7 bigint(15), i8 bigint(15) unsigned );
Query OK, 0 rows affected (0.03 sec)

mysql> desc db1.t_int_width;
+-------+----------------------+------+-----+---------+-------+
| Field | Type                 | Null | Key | Default | Extra |
+-------+----------------------+------+-----+---------+-------+
| i1    | tinyint(2)           | YES  |     | NULL    |       |
| i2    | tinyint(2) unsigned  | YES  |     | NULL    |       |
| i3    | smallint(3)          | YES  |     | NULL    |       |
| i4    | smallint(3) unsigned | YES  |     | NULL    |       |
| i5    | int(8)               | YES  |     | NULL    |       |
| i6    | int(8) unsigned      | YES  |     | NULL    |       |
| i7    | bigint(15)           | YES  |     | NULL    |       |
| i8    | bigint(15) unsigned  | YES  |     | NULL    |       |
+-------+----------------------+------+-----+---------+-------+
8 rows in set (0.01 sec)

mysql> show create table db1.t_int_width;
+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Table       | Create Table                                                                                                                                                                                                                                                                                                                                                     |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| t_int_width | CREATE TABLE `t_int_width` (
  `i1` tinyint(2) DEFAULT NULL,
  `i2` tinyint(2) unsigned DEFAULT NULL,
  `i3` smallint(3) DEFAULT NULL,
  `i4` smallint(3) unsigned DEFAULT NULL,
  `i5` int(8) DEFAULT NULL,
  `i6` int(8) unsigned DEFAULT NULL,
  `i7` bigint(15) DEFAULT NULL,
  `i8` bigint(15) unsigned DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
1 row in set (0.00 sec)

mysql> select * from columns where table_name='t_int_width';
+---------------+--------------+-------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
| TABLE_CATALOG | TABLE_SCHEMA | TABLE_NAME  | COLUMN_NAME | ORDINAL_POSITION | COLUMN_DEFAULT | IS_NULLABLE | DATA_TYPE | CHARACTER_MAXIMUM_LENGTH | CHARACTER_OCTET_LENGTH | NUMERIC_PRECISION | NUMERIC_SCALE | DATETIME_PRECISION | CHARACTER_SET_NAME | COLLATION_NAME | COLUMN_TYPE          | COLUMN_KEY | EXTRA | PRIVILEGES                      | COLUMN_COMMENT | GENERATION_EXPRESSION |
+---------------+--------------+-------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
| def           | db1          | t_int_width | i1          |                1 | NULL           | YES         | tinyint   |                     NULL |                   NULL |                 3 |             0 |               NULL | NULL               | NULL           | tinyint(2)           |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i2          |                2 | NULL           | YES         | tinyint   |                     NULL |                   NULL |                 3 |             0 |               NULL | NULL               | NULL           | tinyint(2) unsigned  |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i3          |                3 | NULL           | YES         | smallint  |                     NULL |                   NULL |                 5 |             0 |               NULL | NULL               | NULL           | smallint(3)          |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i4          |                4 | NULL           | YES         | smallint  |                     NULL |                   NULL |                 5 |             0 |               NULL | NULL               | NULL           | smallint(3) unsigned |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i5          |                5 | NULL           | YES         | int       |                     NULL |                   NULL |                10 |             0 |               NULL | NULL               | NULL           | int(8)               |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i6          |                6 | NULL           | YES         | int       |                     NULL |                   NULL |                10 |             0 |               NULL | NULL               | NULL           | int(8) unsigned      |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i7          |                7 | NULL           | YES         | bigint    |                     NULL |                   NULL |                19 |             0 |               NULL | NULL               | NULL           | bigint(15)           |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_int_width | i8          |                8 | NULL           | YES         | bigint    |                     NULL |                   NULL |                20 |             0 |               NULL | NULL               | NULL           | bigint(15) unsigned  |            |       | select,insert,update,references |                |                       |
+---------------+--------------+-------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+----------------------+------------+-------+---------------------------------+----------------+-----------------------+
8 rows in set (0.01 sec)

mysql> create table db1.t_float_width(f1 float, f2 float(10), f3 float(10, 4), f4 double, f6 double precision(11, 6));
Query OK, 0 rows affected (0.03 sec)

mysql> desc db1.t_float_width;
+-------+--------------+------+-----+---------+-------+
| Field | Type         | Null | Key | Default | Extra |
+-------+--------------+------+-----+---------+-------+
| f1    | float        | YES  |     | NULL    |       |
| f2    | float        | YES  |     | NULL    |       |
| f3    | float(10,4)  | YES  |     | NULL    |       |
| f4    | double       | YES  |     | NULL    |       |
| f6    | double(11,6) | YES  |     | NULL    |       |
+-------+--------------+------+-----+---------+-------+
5 rows in set (0.00 sec)

mysql> select * from columns where table_name='t_float_width';
+---------------+--------------+---------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+--------------+------------+-------+---------------------------------+----------------+-----------------------+
| TABLE_CATALOG | TABLE_SCHEMA | TABLE_NAME    | COLUMN_NAME | ORDINAL_POSITION | COLUMN_DEFAULT | IS_NULLABLE | DATA_TYPE | CHARACTER_MAXIMUM_LENGTH | CHARACTER_OCTET_LENGTH | NUMERIC_PRECISION | NUMERIC_SCALE | DATETIME_PRECISION | CHARACTER_SET_NAME | COLLATION_NAME | COLUMN_TYPE  | COLUMN_KEY | EXTRA | PRIVILEGES                      | COLUMN_COMMENT | GENERATION_EXPRESSION |
+---------------+--------------+---------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+--------------+------------+-------+---------------------------------+----------------+-----------------------+
| def           | db1          | t_float_width | f1          |                1 | NULL           | YES         | float     |                     NULL |                   NULL |                12 |          NULL |               NULL | NULL               | NULL           | float        |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_float_width | f2          |                2 | NULL           | YES         | float     |                     NULL |                   NULL |                12 |          NULL |               NULL | NULL               | NULL           | float        |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_float_width | f3          |                3 | NULL           | YES         | float     |                     NULL |                   NULL |                10 |             4 |               NULL | NULL               | NULL           | float(10,4)  |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_float_width | f4          |                4 | NULL           | YES         | double    |                     NULL |                   NULL |                22 |          NULL |               NULL | NULL               | NULL           | double       |            |       | select,insert,update,references |                |                       |
| def           | db1          | t_float_width | f6          |                5 | NULL           | YES         | double    |                     NULL |                   NULL |                11 |             6 |               NULL | NULL               | NULL           | double(11,6) |            |       | select,insert,update,references |                |                       |
+---------------+--------------+---------------+-------------+------------------+----------------+-------------+-----------+--------------------------+------------------------+-------------------+---------------+--------------------+--------------------+----------------+--------------+------------+-------+---------------------------------+----------------+-----------------------+
5 rows in set (0.01 sec)

 */
shared_ptr< ColumnEntry >
ColumnEntry::MakeColumnEntry( std::string arg_name, ColumnType arg_type, int arg_index,
                              bool arg_primary, bool arg_unique, bool arg_multi, bool arg_foreign,
                              bool arg_unsigned,
                              bool arg_nullable, bool arg_has_default, const shared_ptr<string>& arg_default,
                              int64_t arg_char_max_len, int64_t arg_char_oct_len,
                              int64_t arg_precision, int64_t arg_scale, int64_t arg_datetime_precision,
                              const string& arg_cs, const string& arg_collation,
                              bool arg_explicit_nullable,
                              bool arg_explicit_default_null )
{
    auto colEntry = new ColumnEntry ( arg_name,
                                      arg_type,
                                      arg_index,
                                      arg_primary,
                                      arg_unique,
                                      arg_multi,
                                      arg_foreign,
                                      arg_unsigned,
                                      arg_nullable,
                                      arg_has_default,
                                      arg_default,
                                      arg_char_max_len,
                                      arg_char_oct_len,
                                      arg_precision,
                                      arg_scale,
                                      arg_datetime_precision,
                                      arg_cs,
                                      arg_collation,
                                      arg_explicit_nullable,
                                      arg_explicit_default_null );
    colEntry->CheckConvertDefaultValue();
    shared_ptr< ColumnEntry > colEntryPtr;
    colEntryPtr.reset( colEntry );
    return colEntryPtr;
}
ColumnEntry::ColumnEntry(std::string arg_name, ColumnType arg_type, int arg_index,
                         bool arg_primary, bool arg_unique, bool arg_multi, bool arg_foreign,
                         bool arg_unsigned,
                         bool arg_nullable, bool arg_has_default, const std::shared_ptr<string>& arg_default,
                         int64_t arg_char_max_len, int64_t arg_char_oct_len,
                         int64_t arg_precision, int64_t arg_scale, int64_t arg_datetime_precision,
                         const string& arg_cs, const string& arg_collation,
                         bool arg_explicit_nullable,
                         bool arg_explicit_default_null )
        : DBEntry( arg_name ),
          type( arg_type ),
          is_unsigned( arg_unsigned ),
          is_primary( arg_primary ),
          is_unique( arg_unique ),
          is_multi( arg_multi ),
          allow_null( arg_nullable ),
          has_default( arg_has_default ),
          default_val_ptr( arg_default ),
          char_max_len( arg_char_max_len ),
          actual_char_max_len( arg_char_max_len ),
          char_oct_len( arg_char_oct_len ),
          actual_char_oct_len( arg_char_oct_len ),
          numeric_precision( arg_precision ),
          numeric_scale( arg_scale ),
          datetime_precision( arg_datetime_precision ),
          charset_name( arg_cs ),
          collation_name( arg_collation ),
          is_foreign_key( arg_foreign ),
          index ( arg_index ),
          explicit_nullable( arg_explicit_nullable ),
          explicit_default_null( arg_explicit_default_null ) 
{
    if ( IsStringType( type ) )
    {
        if ( -1 == char_max_len )
            char_max_len = actual_char_max_len = 1;
        if ( -1 == char_oct_len )
            char_oct_len = actual_char_oct_len = 1;
    }
    else if ( ColumnType::TINY_INT == type )
    {
        if ( -1 == arg_precision )
        {
            if ( is_unsigned )
                display_width = 3;
            else
                display_width = 4;
        }
        else
            display_width = arg_precision;
        numeric_precision = 3;
    }
    else if ( ColumnType::SMALL_INT == type )
    {
        if ( -1 == arg_precision )
        {
            if ( is_unsigned )
                display_width = 5;
            else
                display_width = 6;
        }
        else
            display_width = arg_precision;
        numeric_precision = 5;
    }
    else if ( ColumnType::INT == type )
    {
        if ( -1 == arg_precision )
        {
            if ( is_unsigned )
                display_width = 10;
            else
                display_width = 11;
        }
        else
            display_width = arg_precision;
        numeric_precision = 10;
    }
    else if ( ColumnType::LONG_INT == type )
    {
        if ( -1 == arg_precision )
            display_width = 20;
        else
            display_width = arg_precision;
        if ( is_unsigned )
            numeric_precision = 20;
        else
            numeric_precision = 19;

    }
    else if( ColumnType::FLOAT == type || ColumnType::DOUBLE == type )
    {
        if ( -1 == numeric_precision )
        {
            numeric_precision = 12;
        }
        else
        {
            display_width = numeric_precision;
        }
    }
    else if( ColumnType::DECIMAL == type)
    {
        if ( -1 == numeric_precision )
            numeric_precision = 10;
        if ( -1 == numeric_scale )
            numeric_scale = 0;
    }

    InitTypeSize();
}

ColumnEntry::~ColumnEntry()
{
#ifdef DEBUG_MEM
    cout<< "here ~ColumnEntry(): " << GetName() << endl;
#endif
}

void ColumnEntry::InitTypeSize()
{
    switch ( type )
    {
        case schema::ColumnType::BOOL:
        case schema::ColumnType::TINY_INT: // 1 byte
            type_size = sizeof(int8_t);
            break;
        case schema::ColumnType::SMALL_INT:  // 2 bytes
            type_size = sizeof(int16_t);
            break;
        case schema::ColumnType::INT: // 4 bytes
            type_size = sizeof(int32_t);
            break;
        case schema::ColumnType::LONG_INT: // 8 bytes
            type_size = sizeof(int64_t);
            break;
        case schema::ColumnType::DECIMAL:
            type_size = aries_acc::GetDecimalRealBytes( numeric_precision, numeric_scale );
            break;
        case schema::ColumnType::FLOAT:
            type_size = sizeof(float);
            break;
        case schema::ColumnType::DOUBLE:
            type_size = sizeof(double);
            break;
        case schema::ColumnType::DATE:
            type_size = sizeof(aries_acc::AriesDate);
            break;
        case schema::ColumnType::DATE_TIME:
            type_size = sizeof(aries_acc::AriesDatetime);
            break;
        case schema::ColumnType::TIMESTAMP:
            type_size = sizeof(aries_acc::AriesTimestamp);
            break;
        case schema::ColumnType::TEXT:
        case schema::ColumnType::VARBINARY:
        case schema::ColumnType::BINARY:
            type_size = actual_char_max_len;
            break;
        case schema::ColumnType::YEAR:
            type_size = sizeof(aries_acc::AriesYear);
            break;
        case schema::ColumnType::TIME:
            type_size = sizeof(aries_acc::AriesTime);
            break;
        case schema::ColumnType::LIST:
        case schema::ColumnType::UNKNOWN:
        {
            string msg = "Get column size for type " + std::to_string((int) type);
            ARIES_EXCEPTION( ER_UNKNOWN_ERROR,  msg.data() );
            break;
        }
    }
}

ColumnType ColumnEntry::GetType() {
    return type;
}

size_t ColumnEntry::GetTypeSize()
{
    switch ( type )
    {
        case schema::ColumnType::TEXT:
        case schema::ColumnType::VARBINARY:
        case schema::ColumnType::BINARY:
            type_size = GetLength();
            break;
        case schema::ColumnType::LIST:
        case schema::ColumnType::UNKNOWN:
        {
            string msg = "Get column size for type " + std::to_string((int) type);
            ARIES_EXCEPTION( ER_UNKNOWN_ERROR,  msg.data() );
            break;
        }
        default:
            break;
    }
    return type_size;
}
size_t ColumnEntry::GetItemStoreSize()
{
    return GetTypeSize() + allow_null;
}

int ColumnEntry::GetLength() {
    if( IsStringType( type ) )
    {
        return actual_char_max_len;
    }
    else
    {
        return 1;
    }
}

bool ColumnEntry::IsPrimary() {
    return is_primary;
}

bool ColumnEntry::IsForeignKey() {
    return is_foreign_key;
}

bool ColumnEntry::IsAllowNull() {
    return allow_null;
}

std::string ColumnEntry::GetReferenceDesc() {
    return references_desc;
}

int ColumnEntry::GetColumnIndex() const {
    return index;
}

void ColumnEntry::FixLength(int length) {
    actual_char_max_len = actual_char_oct_len = length;
}
/**
 * mysql 5.7 reference mannual 11.7 Data Type Default Values
 * 
 * Handling of Implicit Defaults
 *
 * If a data type specification includes no explicit DEFAULT value, MySQL
 * determines the default value as follows: If the column can take NULL as a value,
 * the column is defined with an explicit DEFAULT NULL clause. If the column cannot
 * take NULL as a value, MySQL defines the column with no explicit DEFAULT clause.
 * Exception: If the column is defined as part of a PRIMARY KEY but not explicitly
 * as NOT NULL, MySQL creates it as a NOT NULL column (because PRIMARY KEY columns
 * must be NOT NULL).
 * 
mysql> create table t_default_now2(f1 datetime default "now");
ERROR 1067 (42000): Invalid default value for 'f1'

mysql> create table t_default_int_error(f1 int default "abc");
ERROR 1067 (42000): Invalid default value for 'f1'

mysql> create table t_default_int_error(f1 tinyint unsigned default "255a");
 ERROR 1067 (42000): Invalid default value for 'f1'
mysql> create table t_default_int_error(f1 tinyint unsigned default "255");
 Query OK, 0 rows affected (0.03 sec)


mysql> create table t_default_int_error(f1 tinyint default 128);
ERROR 1067 (42000): Invalid default value for 'f1'

mysql> create table t_default_int_error(f1 tinyint unsigned default 256);
ERROR 1067 (42000): Invalid default value for 'f1'

mysql> create table t_default_int_error(f1 int default 999999999999999999999999);
ERROR 1067 (42000): Invalid default value for 'f1'

mysql> create table t_default_int_error(f1 tinyint unsigned default 0xff);
 Query OK, 0 rows affected (0.02 sec)

mysql> create table t_default(f1 int not null default null);
ERROR 1067 (42000): Invalid default value for 'f1'

signed int range: -128, 127, -32768, 32767, -2147483648, 2147483647, -9223372036854775808, 9223372036854775807
unsigned int range: 255, 65535, 4294967295, 18446744073709551615
printf("signed int range: %d, %d, %d, %d, %d, %d, %lld, %lld\n",
       INT8_MIN, INT8_MAX,
       INT16_MIN, INT16_MAX,
       INT32_MIN, INT32_MAX,
       INT64_MIN, INT64_MAX);
printf("unsigned int range: %llu, %llu, %llu, %llu\n",
       UINT8_MAX,
       UINT16_MAX,
       UINT32_MAX,
       UINT64_MAX);
 */
/*
mysql 5.7.26, primay key field nullable and null value:

mysql> create table t2(f1 int default null, primary key(f1));
ERROR 1171 (42000): All parts of a PRIMARY KEY must be NOT NULL; if you need NULL in a key, use UNIQUE instead
mysql> create table t2(f1 int null, primary key(f1));
ERROR 1171 (42000): All parts of a PRIMARY KEY must be NOT NULL; if you need NULL in a key, use UNIQUE instead

mysql> create table t1(f1 int, f2 int, primary key (f1));
mysql> desc t1;
+-------+---------+------+-----+---------+-------+
| Field | Type    | Null | Key | Default | Extra |
+-------+---------+------+-----+---------+-------+
| f1    | int(11) | NO   | PRI | NULL    |       |
| f2    | int(11) | YES  |     | NULL    |       |
+-------+---------+------+-----+---------+-------+

mysql> insert into t1(f2) values(1);
ERROR 1364 (HY000): Field 'f1' doesn't have a default value
mysql> insert into t1 values(null, 1);
ERROR 1048 (23000): Column 'f1' cannot be null
*/
void ColumnEntry::CheckConvertDefaultValue()
{
    if ( !has_default )
    {
        if ( IsAllowNull() )
        {
            // for nullable column, default to NULL implicitly 
            has_default = true;
            default_val_ptr = nullptr;
        }
        return;
    }
    if ( !default_val_ptr ) // default value is NULL
    {
        // create table t1(id int primary key, name char(64) );
        // 对于id字段， has_default = true; default_val_ptr = nullptr; allow_null = false 
        if ( !allow_null && !IsPrimary() )
            ARIES_EXCEPTION( ER_INVALID_DEFAULT, GetName().data() );
        return;
    }
    string defValStr = *default_val_ptr;
    string defValStrTrimed = aries_utils::trim( defValStr );

    if ( ColumnType::DECIMAL == type )
    {
        defValStr = defValStrTrimed;
    }
    size_t size = 0;
    char *tail;
    long longVal = 0;
    unsigned long longValUnsigned = 0;
    const char* colName = GetName().data();
    switch ( type )
    {
        case schema::ColumnType::BOOL:
        case schema::ColumnType::TINY_INT: // 1 byte
        {
            size = sizeof( int8_t );
            if ( is_unsigned )
            {
                uint8_t value = 0;
                longValUnsigned = std::strtoul(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (longValUnsigned > UINT8_MAX)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                else
                    value = longValUnsigned;
                default_val_converted.assign( (char*) &value, size );
            }
            else
            {
                int8_t value = 0;
                longVal = std::strtol(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (longVal > INT8_MAX || longVal < INT8_MIN)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                else
                    value = longVal; 
                default_val_converted.assign( (char*) &value, size );
            }
            break;
        }
        case schema::ColumnType::SMALL_INT:  // 2 bytes
        {
            size = sizeof( int16_t );
            if ( is_unsigned )
            {
                uint16_t value = 0;
                longValUnsigned = std::strtoul(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (longValUnsigned > UINT16_MAX)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT,  colName );
                else
                    value = longValUnsigned;
                default_val_converted.assign( (char*) &value, size );
            }
            else
            {
                int16_t value = 0;
                longVal = std::strtol(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (longVal > INT16_MAX || longVal < INT16_MIN)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                else
                    value = longVal; 
                default_val_converted.assign( (char*) &value, size );
            }
            break;
        }
        case schema::ColumnType::INT: // 4 bytes
        {
            size = sizeof( int32_t );
            errno = 0;
            if ( is_unsigned )
            {
                uint32_t value = 0;
                longValUnsigned = std::strtoul(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (longValUnsigned > UINT32_MAX || ERANGE == errno)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                else
                    value = longValUnsigned;
                default_val_converted.assign( (char*) &value, size );
            }
            else
            {
                int32_t value = 0;
                longVal = std::strtol(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if ( longVal > INT32_MAX || longVal < INT32_MIN )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                else
                {
                    if (ERANGE == errno)
                        ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                    value = longVal; 
                }
                default_val_converted.assign( (char*) &value, size );
            }
            break;
        }
        case schema::ColumnType::LONG_INT: // 8 bytes
        {
            size = sizeof( int64_t );
            errno = 0;
            if ( is_unsigned )
            {
                uint64_t value = std::strtoull(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (errno == ERANGE /*&& llValUnsigned == ULLONG_MAX*/)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                default_val_converted.assign( (char*) &value, size );
            }
            else
            {
                int64_t value = std::strtoll(defValStr.c_str(), &tail, 10);
                if ( '\0' != *tail )
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

                if (errno == ERANGE /*&& llValUnsigned == ULLONG_MAX*/)
                    ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
                default_val_converted.assign( (char*) &value, size );
            }
            break;
        }
        case schema::ColumnType::DECIMAL:
        {
            aries_acc::Decimal value( numeric_precision,
                                      numeric_scale,
                                      ARIES_MODE_STRICT_ALL_TABLES,
                                      defValStr.c_str() );
            if ( value.GetError() == ERR_STR_2_DEC || value.GetError() == ERR_OVER_FLOW )
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

            char tmpValueBuff[ 1024 ] = { 0 };
            int len = aries_acc::GetDecimalRealBytes( numeric_precision, numeric_scale );
            if ( !value.ToCompactDecimal( tmpValueBuff, len ) )
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            default_val_converted.assign( tmpValueBuff, len ); 
            break;
        }
        case schema::ColumnType::FLOAT:
        {
            // float unsinged max: 3.40282e38, min: 0
            // float signed min: -3.40282e38, max: 3.40282e38
            size = sizeof( float );
            errno = 0;
            float value = std::strtof( defValStr.c_str(), &tail );
            if ( '\0' != *tail || ERANGE == errno || ( is_unsigned && value < 0 ) )
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

            default_val_converted.assign( (char*) &value, size );
            break;
        }
        case schema::ColumnType::DOUBLE:
        {
            // double max: 1.7976931348623157e308
            size = sizeof( double );
            errno = 0;
            double value = std::strtod( defValStr.c_str(), &tail );
            if ( '\0' != *tail || ERANGE == errno || ( is_unsigned && value < 0 ) )
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );

            default_val_converted.assign( (char*) &value, size );
            break;
        }

        // Date values with two-digit years are ambiguous because the century is unknown.
        // For DATETIME, DATE, and TIMESTAMP types,
        // MySQL interprets dates specified with ambiguous year
        // values using these rules:
        // • Year values in the range 00-69 are default_val_converted to 2000-2069.
        // • Year values in the range 70-99 are default_val_converted to 1970-1999.
        // 
        // For YEAR, the rules are the same, with this exception:
        // A numeric 00 inserted into YEAR(4) results in
        // 0000 rather than 2000.
        // To specify zero for YEAR(4) and have it be interpreted as 2000,
        // specify it as a string '0' or '00'.

        // By default, when MySQL encounters a value for a date or time type
        // that is out of range or otherwise invalid for the type,
        // it converts the value to the “zero” value for that type.
        // The exception is that out-of range TIME values are clipped 
        // to the appropriate endpoint of the TIME range.

        // by enabling the ALLOW_INVALID_DATES SQL mode, MySQL verifies
        // only that the month is in the range from 1 to 12 and
        // that the day is in the range from 1 to 31.

        // MySQL permits you to store dates where the day or month and day
        // are zero in a DATE or DATETIME defValStrumn. 
        // To disallow zero month or day parts in
        // dates, enable the NO_ZERO_IN_DATE mode.

        // MySQL permits you to store a “zero” value of '0000-00-00' as a “dummy date.” 
        // To disallow '0000-00-00', enable the NO_ZERO_DATE mode.

        /*
        The following table shows the format of the “zero” value for each type.
        You can also do this using the values '0' or 0, which are easier to write.

        For temporal types that include a date part (DATE, DATETIME, and TIMESTAMP),
        use of these values produces warnings if the NO_ZERO_DATE SQL
        mode is enabled.
        ====================================
        | Data Type | “Zero” Value         |
        |-----------|----------------------|
        | DATE      | '0000-00-00'         |
        | TIME      | '00:00:00'           |
        | DATETIME  | '0000-00-00 00:00:00'|
        | TIMESTAMP | '0000-00-00 00:00:00'|
        | YEAR      |  0000                |
        ================================== |
        */
        // The supported range is '1000-01-01' to '9999-12-31'.
        case schema::ColumnType::DATE:
        {
            int mode = ARIES_DATE_STRICT_MODE;
            aries_acc::AriesDate date;
            try
            {
                date = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDate( defValStr, mode );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            }
            size = sizeof( aries_acc::AriesDate);
            default_val_converted.assign( (char*) &date, size );
            break;
        }
        // The supported range is '1000-01-01 00:00:00' to '9999-12-31 23:59:59'
        case schema::ColumnType::DATE_TIME:
        {
            int mode = ARIES_DATE_NOT_STRICT_MODE;
            aries_acc::AriesDatetime datetime;
            try
            {
                datetime = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesDatetime( defValStr, mode );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            }
            size = sizeof( aries_acc::AriesDatetime );
            default_val_converted.assign( (char*) &datetime, size );
            break;
        }
        // TIMESTAMP has a range of '1970-01-01 00:00:01' UTC to '2038-01-19 03:14:07' UTC.
        // MySQL does not accept TIMESTAMP values that include a zero in the day or month defValStrumn or values
        // that are not a valid date. The sole exception to this rule is the special “zero” value '0000-00-00
        // 00:00:00'.
        case schema::ColumnType::TIMESTAMP:
        {
            int mode = ARIES_DATE_STRICT_MODE;
            aries_acc::AriesTimestamp timestamp;
            try
            {
                timestamp = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTimestamp( defValStr, mode );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            }
            size = sizeof( aries_acc::AriesTimestamp );
            default_val_converted.assign( (char*) &timestamp, size );
            break;
        }
        case schema::ColumnType::TEXT:
        case schema::ColumnType::VARBINARY:
        case schema::ColumnType::BINARY:
        {
            // mysql non-strict mode:
            // Warning | 1265 | Data truncated for defValStrumn 'f1' at row 1
            size = defValStr.size();
            // mysql> create table t3 (f1 char(5) default "123456");
            // ERROR 1067 (42000): Invalid default value for 'f1'
            if ( size > ( size_t )GetLength() )
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            default_val_converted = defValStr;
            break;
        }
        case schema::ColumnType::YEAR:
        {
            int mode = ARIES_DATE_STRICT_MODE;
            aries_acc::AriesYear year(0);
            try
            {
                year = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesYear( defValStr, mode );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            }
            size = sizeof( aries_acc::AriesYear );
            default_val_converted.assign( (char*) &year, size );
            break;
        }
        case schema::ColumnType::TIME:
        {
            int mode = ARIES_DATE_STRICT_MODE;
            aries_acc::AriesTime time;
            try
            {
                time = aries_acc::AriesDatetimeTrans::GetInstance().ToAriesTime( defValStr, mode );
            }
            catch ( ... )
            {
                ARIES_EXCEPTION( ER_INVALID_DEFAULT, colName );
            }
            size = sizeof( aries_acc::AriesTime );
            default_val_converted.assign( (char*) &time, size );
            break;
        }
        case schema::ColumnType::LIST:
        case schema::ColumnType::UNKNOWN:
        {
            string errMsg =  "not supported column type " + std::to_string( (int) type );
            ARIES_EXCEPTION_SIMPLE( ER_UNKNOWN_ERROR, errMsg.data() );
            break;
        }
    }
}

int64_t ColumnEntry::GetDictId() const { return m_dict->GetId(); }
string ColumnEntry::GetDictName() const { return m_dict->GetName(); }

ColumnType ColumnEntry::GetDictIndexDataType() const
{
    if ( m_dict )
        return m_dict->GetIndexDataType();
    else
        return ColumnType::UNKNOWN;
}

AriesColumnType ColumnEntry::GetDictIndexColumnType() const
{
    AriesColumnType resultDataType{ { AriesValueType::INT32, 1 }, allow_null };
    switch ( m_dict->GetIndexDataType() )
    {
        case aries::schema::ColumnType::TINY_INT:
        {
            resultDataType.DataType.ValueType = AriesValueType::INT8;
            break;
        }
        case aries::schema::ColumnType::SMALL_INT:
        {
            resultDataType.DataType.ValueType = AriesValueType::INT16;
            break;
        }
        case aries::schema::ColumnType::INT:
        {
            resultDataType.DataType.ValueType = AriesValueType::INT32;
            break;
        }
        default:
            aries::ThrowNotSupportedException("dict encoding type: " + std::to_string( (int)m_dict->GetIndexDataType() ) );
            break;
    }
    return resultDataType;
}

size_t ColumnEntry::GetDictIndexItemSize() const
{
    return m_dict->getDictIndexItemSize();
}

size_t ColumnEntry::GetDictCapacity() const
{
    return m_dict->getDictCapacity();
}

} // namespace schema
} // namespace aries
