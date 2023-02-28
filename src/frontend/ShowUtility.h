#ifndef ARIES_SHOW_UTILITY
#define ARIES_SHOW_UTILITY


#include <map>
#include <memory>
#include <string>

#include "VariousEnum.h"

namespace aries {


class ShowUtility {
private:
    ShowUtility();


    ShowUtility(const ShowUtility &arg);

    ShowUtility &operator=(const ShowUtility &arg);

    std::map<LogicType, std::string> global_logictype_map =
            {
                    {LogicType::AND, "and"},
                    {LogicType::OR,  "or"}
            };


    std::map<CalcType, std::string> global_calctype_map =
            {
                    {CalcType::ADD, "+"},
                    {CalcType::SUB, "-"},
                    {CalcType::MUL, "x"},
                    {CalcType::DIV, "/"}
            };


    std::map<ComparisonType, std::string> global_comptype_map =
            {
                    {ComparisonType::DengYu,       "="},
                    {ComparisonType::BuDengYu,     "!="},
                    {ComparisonType::SQLBuDengYu,  "<>"},
                    {ComparisonType::XiaoYuDengYu, "<="},
                    {ComparisonType::DaYuDengYu,   ">="},
                    {ComparisonType::XiaoYu,       "<"},
                    {ComparisonType::DaYu,         ">"},
            };


    std::map<JoinType, std::string> global_jointype_map =
            {
                    {JoinType::InnerJoin,      "Inner Join"},
                    {JoinType::LeftJoin,       "Left Join"},
                    {JoinType::LeftOuterJoin,  "Left Outer Join"},
                    {JoinType::RightJoin,      "Right Join"},
                    {JoinType::RightOuterJoin, "Right Outer Join"},
                    {JoinType::FullJoin,       "Full Join"},
                    {JoinType::FullOuterJoin,  "Full Outer Join"},
                    {JoinType::SemiJoin,       "Semi Join"},
                    {JoinType::AntiJoin,       "AntiJoin"}
            };

    std::map<SetOperationType, std::string> global_setoperationtype_map =
            {
                    {SetOperationType::UNION,         "Union"},
                    {SetOperationType::UNION_ALL,     "Union All"},
                    {SetOperationType::INTERSECT,     "Intersect"},
                    {SetOperationType::INTERSECT_ALL, "Intersect All"},
                    {SetOperationType::EXCEPT,        "Except"},
                    {SetOperationType::EXCEPT_ALL,    "Except All"}

            };


//	static std::shared_ptr<ShowUtility> instance;// = nullptr;

    static ShowUtility *instance;

public:


//	static std::shared_ptr<ShowUtility> GetInstance();
    static ShowUtility *GetInstance();

    std::string GetTextFromBiaodashiType(BiaodashiType arg_bt);

    std::string GetTextFromLogicType(LogicType arg_bt);

    std::string GetTextFromCalcType(CalcType arg_bt);

    std::string GetTextFromComparisonType(ComparisonType arg_bt);

    std::string GetTextFromJoinType(JoinType arg_bt);

    std::string GetTextFromSetOperationType(SetOperationType arg_bt);


};

}//namespace 

#endif
