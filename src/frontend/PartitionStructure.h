#pragma once

#include <memory>
#include <vector>

#include "frontend/CommonBiaodashi.h"

namespace aries {

using namespace schema;

struct PartTypeDef
{
    bool m_liner;
    string m_method;
    vector< BiaodashiPointer > m_partitionExprs;
    string m_exprStr;
};

struct PartValueItem
{
    bool m_isMaxValue = false;
    BiaodashiPointer m_expr;
};
struct PartValueItems
{
    bool m_isMaxValue = false;
    vector< PartValueItem > m_valueItems;
};
using PartValueItemsSPtr = std::shared_ptr< PartValueItems >;

struct PartValues
{
    bool m_isRange;
    bool m_isMaxValue = false;
    vector< PartValueItemsSPtr > m_valueItemsList;
};
using PartValuesSPtr = std::shared_ptr< PartValues >;

struct PartDef
{
    string m_partitionName;
    string m_partExpr;
    PartValuesSPtr m_partValues;
};
using PartDefList = std::shared_ptr< vector< PartDef > >;

class PartitionStructure
{
public:
    bool m_liner;
    std::string m_partMethod;
    vector< BiaodashiPointer > m_partitionExprs;
    string m_partitionExprStr;
    uint32_t m_partitionCount = 0;
    PartDefList m_partitionDefList;
};
using PartitionStructureSPtr = std::shared_ptr< PartitionStructure >;

}