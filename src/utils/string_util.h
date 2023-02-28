//
// Created by tengjp on 19-7-16.
//

#ifndef AIRES_STRING_UTIL_H
#define AIRES_STRING_UTIL_H

#include <string>
#include <algorithm>
#include <vector>
#include "server/mysql/include/my_global.h"

using std::string;
class THD;

const string WHITESPACE = " \n\r\t\v\f";
namespace aries_utils {
#define CHECK_VARIANT_TYPE(value, type) (boost::get<type>(&(value)) != nullptr)
#define IS_STRING_VARIANT(value) (boost::get<std::string>(&(value)) != nullptr)
#define IS_ARIESDATE_VARIANT(value) (boost::get<aries_acc::AriesDate>(&(value)) != nullptr)
#define IS_ARIESDATETIME_VARIANT(value) (boost::get<aries_acc::AriesDatetime>(&(value)) != nullptr)

string ltrim(const string& s, const string& delimiters = WHITESPACE);
string rtrim(const string& s, const string& delimiters = WHITESPACE);
std::string trim(const std::string& s, const string& delimiters = WHITESPACE);
std::string strip_quotes(const std::string& value);

string& to_upper(std::string& value);
string& to_lower(std::string& value);
char *llstr(longlong value,char *buff);
char *ullstr(longlong value,char *buff);

uchar* set_to_string(THD *thd, string* result, ulonglong set, const char *lib[]);
uchar* flagset_to_string(THD *thd, string* result, ulonglong set, const char *lib[]);

std::string convert_to_upper(const std::string& value);
std::string convert_to_lower(const std::string& value);

std::vector< std::string > split( std::string str, const string& delimiter );
std::string get_current_work_directory();

std::string escape_string( const std::string& old );
std::string unescape_string( const std::string& old );

}// namespace aries_utils

#endif //AIRES_STRING_UTIL_H
