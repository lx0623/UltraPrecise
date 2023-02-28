#include "ShowUtility.h"

namespace aries {


ShowUtility *ShowUtility::instance = nullptr;

ShowUtility *ShowUtility::GetInstance() {
    if (instance == nullptr) {
        instance = new ShowUtility();
    }

    return instance;
}

ShowUtility::ShowUtility() {
}

std::string ShowUtility::GetTextFromBiaodashiType(BiaodashiType arg_bt) {
    return "";//unused function!
}

std::string ShowUtility::GetTextFromLogicType(LogicType arg_lt) {
    return this->global_logictype_map[arg_lt];
}

std::string ShowUtility::GetTextFromCalcType(CalcType arg_ct) {
    return this->global_calctype_map[arg_ct];
}

std::string ShowUtility::GetTextFromComparisonType(ComparisonType arg_compt) {
    return this->global_comptype_map[arg_compt];
}

std::string ShowUtility::GetTextFromJoinType(JoinType arg_jt) {
    return this->global_jointype_map[arg_jt];
}

std::string ShowUtility::GetTextFromSetOperationType(SetOperationType arg_bt) {
    return this->global_setoperationtype_map[arg_bt];
}


}
