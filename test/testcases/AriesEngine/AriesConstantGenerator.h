#pragma once

#include "AriesEngine/AriesConstantNode.h"

/**
 * 1, "abc", "2019-10-10"
 * 2, "efg", "2019-10-12"
 */
std::shared_ptr< aries_engine::AriesConstantNode > GenerateConstNode( const string& dbName, const string& tableName, const vector< int >& ids );
