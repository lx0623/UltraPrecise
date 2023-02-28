//
// Created by david.shen on 2019/10/8.
//

#ifndef ARIESASSERT_H
#define ARIESASSERT_H

#include <cstring>
#include <glog/logging.h>
#include "AriesException.h"
#include "server/mysql/include/mysqld_error.h"
#include "server/mysql/include/derror.h"

using namespace aries;
namespace aries
{
std::string GetVersionInfo();
}

#define __ARIES_LOG_ERROR(errorCode, file, line, function, message) \
do {                                                                \
    LOG(ERROR) << "ERROR "                                          \
    << errorCode << " " << file                                     \
    << ":" << std::to_string(line)                                  \
    << ":" << function                                              \
    << " INFO: " << message                                         \
    << ", build info: " << GetVersionInfo();                        \
} while (0)

#define ARIES_EXCEPTION(errorCode, ...)                                       \
do {                                                                          \
    auto message = format_mysql_err_msg((errorCode), ##__VA_ARGS__);          \
    __ARIES_LOG_ERROR(#errorCode, __FILE__, __LINE__, __FUNCTION__, message); \
    throw AriesException((errorCode), message);                               \
} while (0)

#define ARIES_EXCEPTION_SIMPLE(errorCode, errorMessage)                            \
do {                                                                               \
    __ARIES_LOG_ERROR(errorCode, __FILE__, __LINE__, __FUNCTION__, errorMessage); \
    throw AriesException((errorCode), (errorMessage));                             \
} while (0)

#ifndef NDEBUG
#include <cassert>
#define ARIES_ASSERT(e, msg)       \
do {                               \
    DLOG_IF(ERROR, !(e)) << msg;   \
    assert(e);                     \
} while (0)

#else //#ifndef NDEBUG
#include "server/mysql/include/derror.h"

#define ARIES_ASSERT(e, msg)                                                                                     \
do {                                                                                                             \
    if (!(e)) {                                                                                                  \
        std::string message = msg;                                                                               \
        message = "error on `" + string((#e)) + "`" + (message.empty() ? "" : ", debug info `" + message + "`"); \
        __ARIES_LOG_ERROR("ER_UNKNOWN_ERROR", __FILE__, __LINE__, __FUNCTION__, message);                        \
        throw AriesException(ER_UNKNOWN_ERROR, format_mysql_err_msg(ER_UNKNOWN_ERROR));                          \
    }                                                                                                            \
} while (0)

#endif // #ifndef NDEBUG

#endif //ARIESASSERT_H
