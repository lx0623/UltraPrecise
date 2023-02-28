//
// Created by tengjp on 19-8-14.
//

#include "server/mysql/include/mysqld_error.h"
#include "server/mysql/include/derror.h"
#include "AriesException.h"

extern bool STRICT_MODE;

namespace aries {
void ThrowNotSupportedException(const string& msg) {
    string errMsg = format_mysql_err_msg(ER_NOT_SUPPORTED_YET, msg.data());
    throw AriesException(ER_NOT_SUPPORTED_YET, errMsg);
}
void ThrowFakeImplException( int errCode, const string& msg )
{
    throw AriesFakeImplException( errCode, msg );
}

int FormatOutOfRangeValueError(const string& colName,
                               int64_t lineIdx,
                               string &errorMsg)
{
    int errorCode = ER_WARN_DATA_OUT_OF_RANGE;
    errorMsg = format_mysql_err_msg(errorCode, colName.data(), lineIdx + 1);
    return errorCode;
}
int FormatDataTruncError(const string& colName,
                         int64_t lineIdx,
                         string &errorMsg)
{
    int errorCode = WARN_DATA_TRUNCATED;
    errorMsg = format_mysql_err_msg(WARN_DATA_TRUNCATED,
                                    colName.data(), lineIdx + 1);
    return errorCode;
}

int FormatDataTooLongError( const string& colName,
                            int64_t lineIdx,
                            string &errorMsg)
{
    int errorCode = ER_DATA_TOO_LONG;
    errorMsg = format_mysql_err_msg( errorCode,
                                     colName.data(),
                                     lineIdx + 1);
    return errorCode;
}
int FormatTruncWrongValueError(const string &colName,
                               const string &colValue,
                               int64_t lineIdx,
                               const char *valueTypeStr,
                               string &errorMsg)
{
    int errorCode = ER_TRUNCATED_WRONG_VALUE_FOR_FIELD;
    errorMsg = format_mysql_err_msg( ER_TRUNCATED_WRONG_VALUE_FOR_FIELD, valueTypeStr,
                                    colValue.data(), colName.data(), lineIdx + 1 );
    return errorCode;
}

}
