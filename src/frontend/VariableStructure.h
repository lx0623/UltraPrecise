//
// Created by tengjp on 19-8-20.
//

#ifndef AIRES_VARIABLESTRUCTURE_H
#define AIRES_VARIABLESTRUCTURE_H


#include <string>
#include <memory>
#include "AriesDefinition.h"

using std::string;
enum enum_var_type
{
    OPT_DEFAULT= 0, OPT_SESSION, OPT_GLOBAL
};

NAMESPACE_ARIES_START
enum VAR_TYPE : std::uint8_t {
    SYS_VAR,
    USER_VAR
};
class VariableStructure {
public:
    VariableStructure() = default;
    virtual ~VariableStructure() = default;
    VAR_TYPE varType;
    string varName;
};
using VariableStructurePtr = std::shared_ptr<VariableStructure>;

class UserVarStructure : public VariableStructure {
public:
    UserVarStructure() {
        varType = USER_VAR;
    }
    string origVarName;
};
using UserVarStructurePtr = std::shared_ptr<UserVarStructure>;

class SysVarStructure : public VariableStructure {
public:
    SysVarStructure() {
        varType = SYS_VAR;
    }
    enum_var_type varScope; // 0: OPT_DEFAULT, 1: OPT_SESSION, 2: OPT_GLOBAL set_var.h
    string varFullName; // @@global.varname
};

using SysVarStructurePtr = std::shared_ptr<SysVarStructure>;
NAMESPACE_ARIES_END // namespace aries

#endif //AIRES_VARIABLESTRUCTURE_H
