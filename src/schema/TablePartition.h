#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "DBEntry.h"

#include "datatypes/AriesDate.hxx"

namespace aries
{

namespace schema
{
struct TablePartition
{
public:
    int64_t m_tableId;
    //  RANGE, LIST, HASH, LINEAR HASH, KEY, or LINEAR KEY
    uint32_t m_partOrdPos;
    std::string m_partitionName;
    // for RANGE and LIST partitions;
    // For a RANGE partition, it contains the value set
    // in the partition's VALUES LESS THAN clause,
    // which can be either an integer or MAXVALUE.
    // For a LIST partition, this column contains the values
    // defined in the partition's VALUES IN clause, which is
    // a list of comma-separated integer values.
    std::string m_partDesc;
    bool m_isMaxValue = false;
    int64_t m_value;
};
using TablePartitionSPtr = std::shared_ptr< TablePartition >;
}
}
