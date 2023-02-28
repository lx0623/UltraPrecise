//
// Created by david.shen on 2019/9/30.
//

#pragma once

#include <cstring>
#include "AriesDefinition.h"
#include "AriesAssert.h"

using namespace aries;

BEGIN_ARIES_ACC_NAMESPACE

    class AriesEngineException : public AriesException{
    public:
        AriesEngineException(int errCodeArg, const string &errMsgArg = string()): AriesException(errCodeArg, errMsgArg, std::string(), 0) {}
    };

    AriesEngineException NotSupportedException(const string& msg);
    AriesEngineException UnknownException(const string& msg);
    AriesEngineException InnerException(const string& msg);

#define ARIES_ENGINE_EXCEPTION(errorCode, msg) \
do {                                           \
    std::string err = msg;                     \
    ARIES_EXCEPTION(errorCode, err.data());    \
} while (0)

END_ARIES_ACC_NAMESPACE

