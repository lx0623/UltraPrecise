//
// Created by tengjp on 19-7-16.
//
#include <string>
#include <regex>
#include <mutex>
#include <server/mysql/include/sql_const.h>
#include <server/mysql/include/sql_class.h>

using std::string;

namespace aries_utils {

string ltrim(const string& s, const string& delimiters) {
    size_t idx = s.find_first_not_of(delimiters);
    return string::npos == idx ? "" : s.substr(idx);
}
string rtrim(const string& s, const string& delimiters) {
    size_t idx = s.find_last_not_of(delimiters);
    return string::npos == idx ? "" : s.substr(0, idx + 1);
}
std::string trim(const std::string& s, const string& delimiters) {
    return rtrim(ltrim(s, delimiters), delimiters);
}

std::string strip_quotes(const std::string& value) {
    if (value.size() < 3) {
        return value;
    }

    auto first = value[0];
    auto end = value[value.size() - 1];

    if ((first == '`' && end == '`') || (first == '"' && end == '"') || (first == '\'' && end == '\'')) {
        std::string tmp;
        tmp.assign(value.data() + 1, value.size() - 2);
        return tmp;
    }

    return value;
}

string& to_upper(std::string& value) {
    transform(value.begin(), value.end(), value.begin(), ::toupper);
    return value;
}
string& to_lower(std::string& value) {
    transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value;
}

char *llstr(longlong value,char *buff)
{
    longlong10_to_str(value,buff,-10);
    return buff;
}

char *ullstr(longlong value,char *buff)
{
    longlong10_to_str(value,buff,10);
    return buff;
}

std::string convert_to_upper(const std::string& value) {
    std::string upper_stirng = value;
    to_upper(upper_stirng);
    return upper_stirng;
}

std::string convert_to_lower(const std::string& value) {
    std::string upper_stirng = value;
    to_lower(upper_stirng);
    return upper_stirng;
}

uchar* set_to_string(THD *thd, string* result, ulonglong set, const char *lib[])
{
    string unused;
    if (!result)
        result = &unused;

    for (uint i= 0; set; i++, set >>= 1)
        if (set & 1) {
            result->append(lib[i]);
            result->append(",");
        }
    return (uchar*)result->data();
}

uchar* flagset_to_string(THD *thd, string* result, ulonglong set, const char *lib[])
{
    string unused;
    if (!result)
        result = &unused;

    // note that the last element is always "default", and it's ignored below
    for (uint i= 0; lib[i+1]; i++, set >>= 1)
    {
        result->append(lib[i]);
        result->append(set & 1 ? "=on," : "=off,");
    }

    return (uchar*)result->data();
}

std::vector< std::string > split( std::string str, const string& delimiter )
{
    std::vector< std::string > result;
    size_t start = 0;
    size_t end = str.find( delimiter );
    while ( end != std::string::npos )
    {
        result.emplace_back( str.substr( start, end - start ) );
        start = end + 1;
        end = str.find( delimiter, start );
    }
    result.emplace_back( str.substr( start, end ) );
    return result;
}

std::string get_current_work_directory()
{
    static std::string cwd;
    static std::mutex mutex;

    if ( cwd.empty() )
    {
        mutex.lock();
        if ( cwd.empty() )
        {
            char* buf = ( char* )malloc( PATH_MAX );
            auto n = readlink( "/proc/self/exe", buf, PATH_MAX );
            ( void )n;
            assert ( n != -1 );
            char* p = strrchr( buf, '/' );
            if ( NULL != p )
            {
                *p = '\0';
            }

            cwd.assign( buf, strlen( buf ) );
            free( buf );
        }
        mutex.unlock();
    }

    assert( !cwd.empty() );
    return cwd;
}

std::string unescape_string( const std::string& old )
{
    std::string new_str;
    const auto& data = old.data();
    const int size = old.size();

    const char tab = 0x09;

    for ( int i = 0; i < size; i++ )
    {
        if ( data[ i ] == '\\' )
        {
            if ( ( i + 1 ) == size )
            {
                break;
            }

            switch ( data[ i + 1 ] )
            {
                case '\'':
                case '\"':
                case '\?':
                case '\\':
                    new_str.append( data + i + 1, 1 );
                    break;
                case 't':
                    new_str.append( &tab, 1 );
                    break;
            }
            i++;
        }
        else
        {
            new_str.append( data + i, 1 );
        }
    }

    return new_str;
}

std::string escape_string( const std::string& old )
{
    std::string new_str;
    const auto& data = old.data();
    const int size = old.size();

    // const char tab = 0x09;

    const char line = '\\';

    for ( int i = 0; i < size; i++ )
    {
        switch ( data[ i ] )
        {
            case '\'':
            case '\\':
            case '"':
                new_str.append( &line, 1 );
                break;
            default: break;
        }
        new_str.append( data + i , 1 );
    }

    return new_str;
}

} // namespace aries_utils
