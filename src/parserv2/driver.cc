// $Id$
/** \file driver.cc Implementation of the example::Driver class. */

#include <fstream>
#include <sstream>

#ifndef DEBUG
#   ifndef NDEBUG
#       define DEBUG
#   endif
#endif

#ifdef DEBUG
#include <ctime>
#endif

#include <glog/logging.h>
#include <server/mysql/include/derror.h>
#include <server/mysql/include/mysqld_error.h>
#include "AriesAssert.h"

#include "driver.h"
#include "scanner.h"

namespace aries_parser {

Driver::Driver()
    : trace_scanning(false),
      trace_parsing(true)
{
}

bool Driver::parse_stream(std::istream &in,
                          const std::string &sname) {
    streamname = sname;

    in.seekg(0, in.end);
    long length = in.tellg();
    if (input_buf) {
        delete[] input_buf;
        input_buf = nullptr;
    }
    input_buf = new char[length + 1];
    memset(input_buf, 0, length + 1);

    in.seekg(0, in.beg);
    in.read(input_buf, length);
    in.seekg(0, in.beg);

    bool result = parse_stream_internal(in, length);
    delete[] input_buf;
    input_buf = nullptr;
    return result;
}

bool Driver::parse_stdin() {
    streamname = "stdin";
    Scanner scanner(nullptr);
    scanner.set_debug(trace_parsing);
    lexer = &scanner;

    Parser parser(*this);

    return (parser.parse() == 0);
}

bool Driver::parse_file(const std::string &filename)
{
    std::ifstream in(filename.c_str());
    if (!in.good()) return false;

    return parse_stream(in, filename);
}

bool Driver::parse_stream_internal(std::istream &in, long length)
{
    static const int buffLen = 2048;
    char tmpBuff[buffLen];
    memset(tmpBuff, 0, buffLen);
    long len = length < buffLen ? length : buffLen - 1;
    long curPos = in.tellg();
    in.read(tmpBuff, len);
    in.seekg(curPos, in.beg);
#ifdef DEBUG
    LOG(INFO) << "\n===============================";
    LOG(INFO) << "length is: " << length << ", string: " << tmpBuff;
    auto start = clock();
#endif

    Scanner scanner(&in);
    scanner.set_debug(trace_scanning);
    this->lexer = &scanner;

    Parser parser(*this);
#if YYDEBUG
    parser.set_debug_level(trace_parsing);
#endif

    auto ret = (parser.parse() == 0);

#ifdef DEBUG
    auto elpased = clock() - start;
    double elpased_d = ((double)elpased) / CLOCKS_PER_SEC;
    LOG(INFO) << "******* DEBUG ******* >> PARSE ELPASED: " << elpased_d;
#endif

    return ret;
}

bool Driver::parse_string(const std::string &input)
{
    const std::string& sname = "string stream";
    std::istringstream iss(input);
    auto ret = parse_stream(iss, sname);
    return ret;
}

void Driver::error(const class location& l,
		   const std::string& m)
{
    // ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your Rateup server version for the right syntax to use near 'sedsg' at line 1
    // ER_PARSE_ERROR
    static const std::string prefix = "You have an error in your SQL syntax; check the manual that corresponds to your Rateup server version for the right syntax to use";
    std::string errSym = get_string_at_location(l);
    ARIES_EXCEPTION(ER_PARSE_ERROR, prefix.data(), errSym.data(), l.begin.line);
}

void Driver::error(const std::string& m)
{
    ARIES_EXCEPTION_SIMPLE(ER_PARSE_ERROR, m);
}

std::string Driver::get_string_at_location(location l) {
    auto begin = l.begin;
    auto end = l.end;

    if (input_buf == nullptr) {
        return std::string();
    }

#ifdef DEBUG
    LOG(INFO) << "location: " << l << std::endl;
#endif

    int cur = 0;

    auto line = begin.line - 1;
    while (line > 0) {
        if (input_buf[cur] == '\n') {
            line --;
        }
        cur ++;
    }

    auto start = cur + begin.column - 1;

    line =  end.line - begin.line;
    auto end_column = end.column;
    if (line > 0) {
        do {
            if (input_buf[cur] == '\n') {
                line --;
            }
            cur ++;
        } while (line > 0);
    }

    auto end_pos = cur + end_column - 1;

    auto length = end_pos - start;
    std::string ret(input_buf + start, length);

#ifndef NDEBUG
    LOG(INFO) << "GOT STRING: line " << begin.line << ":" << begin.column << " ~ " << end.line << ":" << end.column << endl;
    LOG(INFO) << "[" << ret << "]" << ", length: " << length << std::endl;
#endif
    return ret;
}

} // namespace aries_parser
