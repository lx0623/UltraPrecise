//
// Created by tengjp on 19-6-26.
//

#include <server/mysql/include/mysqld_error.h>
#include "SQLResult.h"
namespace aries {
    SQLResult::SQLResult() : mSuccess(false), mErrorCode(ER_UNKNOWN_ERROR) {}
void SQLResult::SetSuccess(bool b)
{
    mSuccess = b;
    if (b) {
        mErrorCode = 0;
    }
}
bool SQLResult::IsSuccess()
{
    return mSuccess;
}
void SQLResult::SetResults(const std::vector<AbstractMemTablePointer>& results)
{
    mResults = results;
}
const std::vector<AbstractMemTablePointer>& SQLResult::GetResults()
{
    return mResults;
}
}
