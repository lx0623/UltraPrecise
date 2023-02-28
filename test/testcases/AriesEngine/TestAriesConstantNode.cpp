#include <gtest/gtest.h>

#include "../../TestUtils.h"
#include "AriesConstantGenerator.h"

using namespace aries_engine;
using namespace aries_acc;

/**
 * 1, "abc", "2019-10-10"
 * 2, "efg", "2019-10-12"
 */
//ARIES_UNIT_TEST( AriesConstantNode, GetNext )
//{
//    auto node = GenerateConstNode();
//
//    ASSERT_TRUE( node->Open() );
//
//    auto result = node->GetNext();
//    ASSERT_EQ( result.Status, AriesOpNodeStatus::END );
//    ASSERT_TRUE( result.TableBlock );
//
//    auto& table = result.TableBlock;
//
//    ASSERT_EQ( table->GetRowCount(), 2 );
//    ASSERT_EQ( table->GetColumnCount(), 3 );
//
//    result = node->GetNext();
//    ASSERT_EQ( result.Status, AriesOpNodeStatus::END );
//    ASSERT_FALSE( result.TableBlock );
//}
