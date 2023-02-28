//
// Created by tengjp on 19-8-12.
//

#ifndef AIRES_SYS_VAR_H
#define AIRES_SYS_VAR_H

#include <unordered_map>
#include <string>
#include "AriesDefinition.h"

using namespace aries;
using std::string;
using std::unordered_map;
class aries_sys_var {
public:
    aries_sys_var(const string& nameArg, const string& valueArg) :
            name(nameArg), value(valueArg)
    {
    }
    std::string name;
    std::string value;
    AriesValueType valueType;

public:
    virtual int IntValue() const = 0;
    virtual string StringValue() const = 0;

    AriesValueType GetValueType() const {
        return valueType;
    }
};

class aries_sys_var_int : public aries_sys_var {
public:
    aries_sys_var_int(const string& nameArg, const string& valueArg) :
            aries_sys_var(nameArg, valueArg) {
        valueType = AriesValueType::INT32;
    }
    virtual int IntValue() const override {
        return *((int*)value.data());
    }
    virtual string StringValue() const override {
        return std::to_string(IntValue());
    }
};
class aries_sys_var_string : public aries_sys_var {
public:
    aries_sys_var_string(const string& nameArg, const string& valueArg) :
            aries_sys_var(nameArg, valueArg) {
        valueType = AriesValueType::CHAR;
    }
    virtual int IntValue() const override {
        return 0;
    }
    virtual string StringValue() const override {
        return value;
    }
};



#endif //AIRES_SYS_VAR_H
