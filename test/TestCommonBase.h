//
// Created by david.shen on 2019/9/25.
//

#ifndef AIRES_TESTCOMMONBASE_H
#define AIRES_TESTCOMMONBASE_H

#include <cstring>
#include "AriesEngineWrapper/AriesMemTable.h"

using namespace aries_engine;
using namespace aries_acc;
using namespace std;

void CheckQueryResult( AriesTableBlockUPtr& queryResultTable, string expectResultFile );
void doQueryAndCheckResult(string dbName, string sqlFile, string resultFile, int query_number, bool check_query_result=true);
void TestTpch( int scale, int arg_query_number );
void TestTpch218( int scale, int arg_query_number, bool check_query_result=true );


#endif //AIRES_TESTCOMMONBASE_H
