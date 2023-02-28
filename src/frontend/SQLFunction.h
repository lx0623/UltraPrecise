#ifndef ARIES_SQL_FUNCTION
#define ARIES_SQL_FUNCTION

#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>


#include "TroubleHandler.h"
#include "VariousEnum.h"

#include "ColumnStructure.h"
#include "AbstractBiaodashi.h"


namespace aries {

class SQLFunction {
private:

    SQLFunction(const SQLFunction &arg);

    SQLFunction &operator=(const SQLFunction &arg);

    std::vector<std::string> agg_func_names{"COUNT", "AVG", "SUM", "MAX", "MIN", "ANY_VALUE"};

    std::string name;
    BiaodashiValueType value_type = BiaodashiValueType::UNKNOWN;
    bool isAggFunc = false;

    AriesSqlFunctionType type;

public:

    SQLFunction(std::string arg_name);

    std::string GetName();

    bool GetIsAggFunc();

    BiaodashiValueType GetValueType();

    void SetValueType(BiaodashiValueType arg_value_type);

    AriesSqlFunctionType GetType();

    std::string ToString();
};

typedef std::shared_ptr<SQLFunction> SQLFunctionPointer;

}//namespace









#endif
