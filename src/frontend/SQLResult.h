//
// Created by tengjp on 19-6-26.
//

#ifndef AIRES_SQLRESULT_H
#define AIRES_SQLRESULT_H

#include <vector>
#include "AriesEngineWrapper/AbstractMemTable.h"
using std::string;
using std::vector;

namespace aries {
class SQLResult {
private:
    bool mSuccess;
    int mErrorCode;
    string mErrMsg;
    std::vector<string> mRowKeys;
    std::vector<AbstractMemTablePointer> mResults;

public:
    SQLResult();
    void SetResults( const std::vector<AbstractMemTablePointer>& results);
    const std::vector<AbstractMemTablePointer>& GetResults();
    void SetSuccess(bool b);
    void AddRowKeys(const vector<string>& rowKeys) { mRowKeys.insert(std::end(mRowKeys), std::begin(rowKeys), std::end(rowKeys)); };
    void SetRowKeys(const vector<string>& rowKeys) { mRowKeys = rowKeys; };
    bool IsSuccess();
    void SetError( int errCode, const string& errMsg )
    {
        mSuccess = false;
        mErrorCode = errCode;
        mErrMsg.assign( errMsg );
    }
    void SetErrorCode(int errCode) {
        mSuccess = false;
        mErrorCode = errCode;
    }

    int GetErrorCode() { return mErrorCode; }
    const string& GetErrorMessage() { return mErrMsg; }
    const std::vector<string>& GetAffectedRowKeys() { return mRowKeys; }
};
typedef std::shared_ptr<SQLResult> SQLResultPtr;
}


#endif //AIRES_SQLRESULT_H
