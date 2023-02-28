//
// Created by david.shen on 2019/9/30.
//

#include <iostream>
#include "server/mysql/include/mysqld_error.h"
#include "AriesEngineException.h"

BEGIN_ARIES_ACC_NAMESPACE

    AriesEngineException NotSupportedException(const string& msg) {
        return AriesEngineException(ER_NOT_SUPPORTED_YET, msg);
    }

    AriesEngineException UnknownException(const string& msg) {
        return AriesEngineException(ER_UNKNOWN_ERROR, msg);
    }

END_ARIES_ACC_NAMESPACE