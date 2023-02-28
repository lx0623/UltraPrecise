#pragma once

#include <memory>
#include "AriesDefinition.h"
#include "AriesColumnType.h"
#include "DBEntry.h"
#include "AriesException.h"

namespace aries {
class AriesDict;
using AriesDictSPtr = shared_ptr< AriesDict >;
namespace schema
{

extern const string RATEUP_NULL_VALUE_STRING;
extern const std::string DEFAULT_VAL_NOW;
enum class ColumnType : int
{
    TINY_INT, // 1 byte
    SMALL_INT, // 2 bytes
	INT, // 4 bytes
    LONG_INT, // 8 bytes
    DECIMAL,
	FLOAT,
    DOUBLE,
	TEXT,
	BOOL,
	DATE,
    TIME,
    DATE_TIME,
    TIMESTAMP,
	UNKNOWN,
    YEAR,
    VARBINARY,
    BINARY,
	LIST
};

class TableEntry;
class ColumnEntry;

class ColumnEntry: public DBEntry
{
private:
    ColumnEntry(std::string arg_name, ColumnType arg_type, int arg_position,
                bool arg_primary, bool arg_unique, bool arg_multi, bool arg_foreign,
                bool arg_unsigned,
                bool arg_nullable, bool arg_has_default, const shared_ptr<string>& arg_default,
                int64_t arg_char_max_len, int64_t arg_char_oct_len,
                int64_t arg_precision, int64_t arg_scale, int64_t arg_datetime_precision,
                const string& arg_cs, const string& arg_collation,
                bool arg_explicit_nullable = false,
                bool arg_explicit_default_null = false );
    void CheckConvertDefaultValue();

public:
    ColumnType type;
    bool is_unsigned;
    bool is_primary;
    bool is_unique;
    // refman-5.7.26-en.a4 24.5 The INFORMATION_SCHEMA COLUMNS Table
    // If COLUMN_KEY is MUL, the column is the first column of a
    // nonunique index in which multiple occurrences of a given value
    // are permitted within the column.
    bool is_multi;
    bool allow_null;
    bool has_default;
    std::shared_ptr<std::string> default_val_ptr;
    string default_val_converted = RATEUP_NULL_VALUE_STRING;
    // For string columns, the maximum length in characters.
    // not include nullable flag
    int64_t char_max_len = -1;
    int64_t actual_char_max_len = -1;
    // For string columns, the maximum length in bytes.
    // CHARACTER_OCTET_LENGTH should be the same as CHARACTER_MAXIMUM_LENGTH,
    // except for multibyte character sets.
    int64_t char_oct_len = -1;
    int64_t actual_char_oct_len = -1;
    // For numeric columns, the numeric precision.
    int64_t numeric_precision = -1;
    // For numeric columns, the numeric scale.
    int64_t numeric_scale = -1;
    // only for integer types
    int display_width = -1;
    // For temporal columns, the fractional seconds precision.
    int64_t datetime_precision = -1;
    string charset_name;
    string collation_name;
    bool is_foreign_key;
    std::string references_desc;

    EncodeType encode_type = EncodeType::NONE;

private:
    // start from 0;
    // ordinal_position in information_schema.columns start from 1
    int index = -1;
    size_t type_size;
    bool explicit_nullable = false;
    bool explicit_default_null = false;

    AriesDictSPtr m_dict;

public:
    static shared_ptr< ColumnEntry > MakeColumnEntry(std::string arg_name, ColumnType arg_type, int arg_position,
                bool arg_primary, bool arg_unique, bool arg_multi, bool arg_foreign,
                bool arg_unsigned,
                bool arg_nullable, bool arg_has_default, const shared_ptr<string>& arg_default,
                int64_t arg_char_max_len, int64_t arg_char_oct_len,
                int64_t arg_precision, int64_t arg_scale, int64_t arg_datetime_precision,
                const string& arg_cs, const string& arg_collation,
                bool arg_explicit_nullable = false,
                bool arg_explicit_default_null = false );
    ~ColumnEntry();

    bool IsExplicitNullable() const { return  explicit_nullable; }
    bool IsExplicitDefaultNull() const { return  explicit_default_null; }

    void SetDict( const AriesDictSPtr& dict )
    {
        encode_type = EncodeType::DICT;
        m_dict = dict;
    }

    AriesDictSPtr GetDict() const
    {
        return m_dict;
    }

    int64_t GetDictId() const;
    string GetDictName() const;

    ColumnType GetDictIndexDataType() const;

    AriesColumnType GetDictIndexColumnType() const;

    size_t GetDictIndexItemSize() const;

    size_t GetDictCapacity() const;

    string GetConvertedDefaultValue() const
    {
        return default_val_converted;
    }

    static bool IsStringType( ColumnType type )
    {
        return ( ColumnType::TEXT == type ||
                 ColumnType::VARBINARY == type ||
                 ColumnType::BINARY == type );
    }
    static bool IsIntegerType( ColumnType type )
    {
        return ( ColumnType::TINY_INT == type ||
                 ColumnType::SMALL_INT == type ||
                ColumnType::INT == type ||
                ColumnType::LONG_INT == type );
    }
    static bool IsNumberType( ColumnType type )
    {
        return ( ColumnType::TINY_INT == type ||
                 ColumnType::SMALL_INT == type ||
                ColumnType::INT == type ||
                ColumnType::LONG_INT == type ||
                ColumnType::FLOAT == type ||
                ColumnType::DOUBLE == type ||
                ColumnType::DECIMAL == type );
    }
    // getters
    ColumnType GetType();
    int GetLength();
    size_t GetTypeSize();
    size_t GetItemStoreSize();
    bool IsUnsigned() { return is_unsigned; }
    bool IsPrimary();
    bool IsAllowNull();
    bool IsForeignKey();
    std::string GetFilePath();

    int GetDecimal() {
        return numeric_scale;
    }
    std::string GetReferenceDesc();


    int GetColumnIndex() const;

    void FixLength(int length);

    void SetColumnId(int column_id);

    const std::shared_ptr<string> GetDefault() const { return default_val_ptr; }
    bool HasDefault() const { return has_default; }

    std::string ToString() {
        std::string ret = "";
        ret += GetName();
        ret += " ";
        // ret += column_type;
        /*
        if (this->column_major_len > 0) {
            ret += "(";
            ret += std::to_string(this->column_major_len);
            if (this->column_minor_len >= 0) {
                ret += " , ";
                ret += std::to_string(this->column_minor_len);
            }
            ret += ")";
        }
        ret += " ";
        if (this->not_null) {
            ret += " NOT NULL ";
        }
        if (this->primary_key) {
            ret += " PRIMARY KEY ";
        }
        if (this->foreign_key) {
            ret += " FOREIGN KEY REFERENCES ";
            ret += this->fk_table_name;
            ret += "(";
            ret += this->fk_column_name;
            ret += ")";
        }

        if (!default_value.empty()) {
            ret += " DEFAULT " + default_value;
        }
         */

        return ret;
    }
private:
    void InitTypeSize();
};
using ColumnEntryPtr = std::shared_ptr<ColumnEntry>;

string DataTypeString (ColumnType columnType);
string ColumnTypeString (const ColumnEntryPtr& column);


} // namespace schema
} // namespace aries
