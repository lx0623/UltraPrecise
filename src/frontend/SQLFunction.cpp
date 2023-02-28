#include "AriesAssert.h"
#include "SQLFunction.h"

#include "utils/string_util.h"


// using namespace aries_acc;

namespace aries {

// SUBSTRING, EXTRACT, ST_VOLUMN, DATE, NOW, DATE_SUB, DATE_ADD, DATE_DIFF, TIME_DIFF, ABS, COUNT, SUM, AVG, MAX, MIN
static std::map<std::string, AriesSqlFunctionType> name_type_maps = {
    {"SUBSTRING", AriesSqlFunctionType::SUBSTRING},
    {"EXTRACT", AriesSqlFunctionType::EXTRACT},
    {"ST_VOLUMN", AriesSqlFunctionType::ST_VOLUMN},
    {"DATE", AriesSqlFunctionType::DATE},
    {"NOW", AriesSqlFunctionType::NOW},
    {"DATE_ADD", AriesSqlFunctionType::DATE_ADD},
    {"ADDDATE", AriesSqlFunctionType::DATE_ADD},
    {"DATE_SUB", AriesSqlFunctionType::DATE_SUB},
    {"SUBDATE", AriesSqlFunctionType::DATE_SUB},
    {"DATEDIFF", AriesSqlFunctionType::DATE_DIFF},
    {"TIMEDIFF", AriesSqlFunctionType::TIME_DIFF},
    {"ABS", AriesSqlFunctionType::ABS},
    {"COUNT", AriesSqlFunctionType::COUNT},
    {"SUM", AriesSqlFunctionType::SUM},
    {"AVG", AriesSqlFunctionType::AVG},
    {"MAX", AriesSqlFunctionType::MAX},
    {"MIN", AriesSqlFunctionType::MIN},
    {"UNIX_TIMESTAMP", AriesSqlFunctionType::UNIX_TIMESTAMP},
    {"CAST", AriesSqlFunctionType::CAST},
    {"CONVERT", AriesSqlFunctionType::CONVERT},
    {"MONTH", AriesSqlFunctionType::MONTH},
    {"DATE_FORMAT", AriesSqlFunctionType::DATE_FORMAT},
    {"COALESCE", AriesSqlFunctionType::COALESCE},
    {"CONCAT", AriesSqlFunctionType::CONCAT},
    {"TRUNCATE", AriesSqlFunctionType::TRUNCATE},
    {"DICT_INDEX", AriesSqlFunctionType::DICT_INDEX},
    {"ANY_VALUE", AriesSqlFunctionType::ANY_VALUE}
};

SQLFunction::SQLFunction(std::string arg_name) {
    ARIES_ASSERT(!arg_name.empty(), "sql functions's name should not be empty");

    this->name = aries_utils::to_upper(arg_name);

    for (size_t i = 0; i < this->agg_func_names.size(); i++) {
        if (arg_name == this->agg_func_names[i]) {
            this->isAggFunc = true;
            break;
        }
    }
    if( name_type_maps.find(arg_name) == name_type_maps.end() )
        ThrowNotSupportedException( arg_name );
    type = name_type_maps[arg_name];

    /*we don't set valuetype now cause we don't know!*/
}

std::string SQLFunction::GetName() {
    return this->name;
}

bool SQLFunction::GetIsAggFunc() {
    return this->isAggFunc;
}

BiaodashiValueType SQLFunction::GetValueType() {
    return this->value_type;
}

void SQLFunction::SetValueType(BiaodashiValueType arg_value_type) {
    this->value_type = arg_value_type;
}

AriesSqlFunctionType SQLFunction::GetType() {
    return type;
}

std::string SQLFunction::ToString() {
    return "[SQLFunction:(" + this->name + ")]";
}

}
