#pragma once

#include <memory>
#include <AriesDefinition.h>

NAMESPACE_ARIES_START
enum TX_CMD : int8_t
{
    TX_START = 0,
    TX_COMMIT,
    TX_ROLLBACK
};
enum enum_yes_no_unknown {
  TVL_YES,
  TVL_NO,
  TVL_UNKNOWN
};

const enum_yes_no_unknown DEFAULT_CHAIN = TVL_NO;
const enum_yes_no_unknown DEFAULT_RELEASE = TVL_NO;
struct TransactionStructure {
  TX_CMD txCmd;
  enum_yes_no_unknown txChain = DEFAULT_CHAIN;;
  enum_yes_no_unknown txRelease = DEFAULT_RELEASE;
};
using TransactionStructurePtr = std::shared_ptr< TransactionStructure >;

NAMESPACE_ARIES_END // namespace aries
