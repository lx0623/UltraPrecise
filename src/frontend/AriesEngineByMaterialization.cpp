/*
 * AriesEngineByMaterialization.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: lichi
 */

#include <frontend/AriesEngineByMaterialization.h>
#include "schema/SchemaManager.h"
#include "schema/DatabaseEntry.h"
#include "schema/TableEntry.h"
#include "SelectStructure.h"
#include "../CudaAcc/AriesSqlOperator.h"
#include "datatypes/AriesDatetimeTrans.h"
#include "AriesAssert.h"
#include "BiaodashiAuxProcessor.h"
#include "utils/datatypes.h"
#include "AriesEngine/AriesConstantNode.h"
#include "AriesEngineWrapper/AriesExprBridge.h"
#include "AriesEngine/AriesSpoolCacheManager.h"
#include "CudaAcc/DynamicKernel.h"
#include "CpuTimer.h"
#include "AriesEngineWrapper/ExpressionSimplifier.h"

namespace aries
{

    AriesEngineByMaterialization::AriesEngineByMaterialization( bool isUpdate )
        : m_isUpdate( isUpdate )
    {
        //m_ctx = InitAriesEngine();
    }

    AriesEngineByMaterialization::~AriesEngineByMaterialization()
    {
        //UninitAriesEngine(m_ctx);
    }

    string AriesEngineByMaterialization::ToString()
    {
        return "AriesEngineByMaterialization";
    }

    AbstractMemTablePointer AriesEngineByMaterialization::ExecuteQueryTree( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName )
    {
        assert( arg_query_tree );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        ExecuteUncorrelatedSubQueries( tx, arg_query_tree, dbName );
        auto result = ExecuteQuery( tx, arg_query_tree, dbName );
#ifdef ARIES_PROFILE
        JSON perfStat = {
            {"plans", {m_perfStat}},
            {"total", t.end()},
        };
        SelectStructure *ss = ( SelectStructure * )( ( arg_query_tree->GetMyQuery() ).get() );
        if ( ss->GetInitQueryCount() > 0 )
            perfStat["plans"].push_back(m_perfStat);
        LOG( INFO ) << "\n" << perfStat;
#endif
        return result;
    }

    AriesMemTableSPtr AriesEngineByMaterialization::ExecuteQueryPipeline( AriesOpNodeSPtr pipeline )
    {
        AriesMemTableSPtr table;
        AriesOutputNodeSPtr outputNode = CreateOutputNode( pipeline );
        if( outputNode )
        {
        #ifdef ARIES_PROFILE
            aries::CPU_Timer t;
            t.begin();
        #endif
            string code = outputNode->GetPipelineKernelCode();
            if( !code.empty() )
            {
                AriesDynamicCodeInfo codeInfo;
                codeInfo.KernelCode = code;
                auto modules = AriesDynamicKernelManager::GetInstance().FindModule( codeInfo.KernelCode );
                if( !modules )
                {
                    LOG(INFO) << "===============dynamic kernel start=================" << endl;
                    LOG(INFO) << code << endl;
                    LOG(INFO) << "===============dynamic kernel end================" << endl;
                    modules = AriesDynamicKernelManager::GetInstance().CompileKernels( codeInfo );
                    if ( !modules || modules->Modules.empty() )
                        ARIES_EXCEPTION_SIMPLE(ER_UNKNOWN_ERROR, "dyn compile error");
                }
                outputNode->AttachCuModuleToPipeline( modules->Modules );
            }
        #ifdef ARIES_PROFILE
            long elapsed = t.end();
            m_perfStat["compileKernel"] = elapsed;
        #endif

            auto result = outputNode->GetResult();

        #ifdef ARIES_PROFILE
            m_perfStat["plan"] = {pipeline->GetProfile()};
        #endif
            table = std::make_shared< AriesMemTable >();
            table->SetContent( std::move( result ) );
        }

        return table;
    }

    AbstractMemTablePointer
    AriesEngineByMaterialization::ExecuteInsert( const aries_engine::AriesTransactionPtr& tx,
                                                 const string& dbName,
                                                 const string& tableName,
                                                 vector< int >& insertColumnIds,
                                                 VALUES_LIST& insertValuesList,
                                                 vector< BiaodashiPointer >& optUpdateColumnExprs,
                                                 vector< BiaodashiPointer >& optUpdateValueExprs,
                                                 SQLTreeNodePointer queryPlanTree )
    {
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesMemTableSPtr table;
        auto dbEntry =
           SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
        auto tableEntry = dbEntry->GetTableByName( tableName );
        auto insertNode = make_shared< AriesInsertNode >( tx, dbName, tableName );
        insertNode->SetNodeId( 0 );
        insertNode->SetColumnIds( insertColumnIds );
        if ( queryPlanTree )
        {
            ExecuteUncorrelatedSubQueries( tx, queryPlanTree, dbName );
            int nodeIndex = 1;
            auto queryPipeline = CreateOpNode( tx, queryPlanTree, dbName, nodeIndex );
            insertNode->SetSourceNode( queryPipeline );
        }
        else
        {
#ifdef ARIES_PROFILE
            aries::CPU_Timer t2;
            t2.begin();
#endif
            vector< vector< AriesCommonExprUPtr > > insertData;
            AriesExprBridge bridge;
            auto columnCnt = insertValuesList[ 0 ]->size();
            auto thd = current_thd;
            for ( auto& valueExprLine : insertValuesList )
            {
                vector< AriesCommonExprUPtr > lineData;
                lineData.reserve( columnCnt );
                for ( auto& valueExpr : *valueExprLine )
                {
                    auto commonExpr = ( CommonBiaodashi* )( valueExpr.get() );
                    aries_engine::ExpressionSimplifier exprSimplifier;
                    auto result = exprSimplifier.Simplify( commonExpr, thd );
                    if ( result )
                    {
                        lineData.push_back( std::move( result ) );
                    }
                    else
                    {
                        ARIES_EXCEPTION_SIMPLE( ER_BAD_FIELD_ERROR, "Unknown column in 'field list'" );
                    }
                }
                insertData.emplace_back( std::move( lineData ) );
            }
#ifdef ARIES_PROFILE
            LOG( INFO ) << "Expr bridge time: " << t2.end() << "ms";
#endif
            auto constantNode = make_shared< AriesConstantNode >( dbName, tableName );
            string errorMsg;
            int errorCode = constantNode->SetColumnData( insertData, insertColumnIds, errorMsg );
            if ( 0 != errorCode )
                ARIES_EXCEPTION_SIMPLE( errorCode, errorMsg.data() );

            constantNode->SetNodeId( 1 );
            insertNode->SetSourceNode( constantNode );
        }

        table = ExecuteQueryPipeline( insertNode );
#ifdef ARIES_PROFILE
        m_perfStat["info"] = std::to_string( t.end() ) + "ms total time of insert";
        LOG( INFO ) << "\n" << m_perfStat;
#endif
        return table;
    }

    void CheckUpdateTables( const vector< BiaodashiPointer >& selectExprs,
                            const string& targetDbName,
                            string& tableName, // out param
                            vector< int >& updateColumnIds ) // out param
    {
        map< TableMapKey, int > tableMap;

        BiaodashiPointer expr = selectExprs[ 0 ];
        auto col = boost::get< ColumnShellPointer >( ( ( CommonBiaodashi* )expr.get() )->GetContent() );
        tableName = col->GetTableName();
        auto dbEntry =
           SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( targetDbName );
        auto tableEntry = dbEntry->GetTableByName( tableName );

        // for update t set f1 = expr1, f2 = expr2,
        // the select statement is:
        // select __rateup_rowid__, f1, f2, expr1, expr2 from ...
        size_t count = selectExprs.size() >> 1;
        for( size_t i = 1; i <= count; i++ )
        {
            expr = selectExprs[ i ];
            col = boost::get< ColumnShellPointer >( ( ( CommonBiaodashi* )expr.get() )->GetContent() );
            tableName = col->GetTableName();
            auto key = TableMapKey( targetDbName, tableName );
            auto it = tableMap.find( key );
            if ( tableMap.end() == it )
                tableMap[ key ] = 1;

            if ( tableMap.size() > 1 )
                ThrowNotSupportedException( "update multi tables" );

            auto columnName = col->GetColumnName();
            auto columnEntry = tableEntry->GetColumnByName( columnName );
            updateColumnIds.push_back( columnEntry->GetColumnIndex() + 1 );
        }
    }

    AbstractMemTablePointer
    AriesEngineByMaterialization::ExecuteUpdateTree( const aries_engine::AriesTransactionPtr& tx,
                                                     SQLTreeNodePointer arg_query_tree,
                                                     const string& targetDbName,
                                                     const std::string& defaultDbName )
    {
        assert( arg_query_tree );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesMemTableSPtr table;
        ExecuteUncorrelatedSubQueries( tx, arg_query_tree, defaultDbName );

        m_targetDbName = targetDbName;
        if ( m_targetDbName.empty() )
            m_targetDbName = defaultDbName;
        int nodeIndex = 1;
        auto pipeline = CreateOpNode( tx, arg_query_tree, defaultDbName, nodeIndex );
        if( pipeline )
        {
            std::vector< BiaodashiPointer > the_select_exprs = arg_query_tree->GetExprs4ColumnNode();
            vector< int > updateColumnIds;
            CheckUpdateTables( the_select_exprs,
                               m_targetDbName,
                               m_targetTableName,
                               updateColumnIds );
            auto updateNode = make_shared< AriesUpdateNode >( tx, m_targetDbName, m_targetTableName );
            updateNode->SetNodeId( 0 );
            updateNode->SetUpdateColumnIds( updateColumnIds );
            // see function CreateUpdateStructure in parserv2/common.cc
            updateNode->SetColumnId4RowPos( 1 );
            updateNode->SetSourceNode( pipeline );
            table = ExecuteQueryPipeline( updateNode );
        }
#ifdef ARIES_PROFILE
        m_perfStat["info"] = std::to_string( t.end() ) + "ms total time of update";
        LOG( INFO ) << "\n" << m_perfStat;
#endif
        return table;
    }

    AbstractMemTablePointer
    AriesEngineByMaterialization::ExecuteDeleteTree( const aries_engine::AriesTransactionPtr& tx,
                                                     const string& dbName,
                                                     const string& tableName,
                                                     SQLTreeNodePointer arg_query_tree,
                                                     const string& defaultDbName )
    {
        assert( arg_query_tree );
#ifdef ARIES_PROFILE
        aries::CPU_Timer t;
        t.begin();
#endif
        AriesMemTableSPtr table;
        ExecuteUncorrelatedSubQueries( tx, arg_query_tree, defaultDbName );
        int nodeIndex = 1;
        auto pipeline = CreateOpNode( tx, arg_query_tree, defaultDbName, nodeIndex );
        if( pipeline )
        {
            auto deleteNode = make_shared< AriesDeleteNode >( tx, dbName, tableName );
            deleteNode->SetNodeId( 0 );
            deleteNode->SetSourceNode( pipeline );
            deleteNode->SetColumnId4RowPos( 1 );
            table = ExecuteQueryPipeline( deleteNode );
        }
#ifdef ARIES_PROFILE
        m_perfStat["info"] = std::to_string( t.end() ) + "ms total time of delete";
        LOG( INFO ) << "\n" << m_perfStat;
#endif
        return table;
    }

    AbstractMemTablePointer AriesEngineByMaterialization::ExecuteQuery( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName )
    {
        AriesMemTableSPtr table;
        spool_cache_manager = std::make_shared< AriesSpoolCacheManager >();
        int nodeIndex = 0;
        auto pipeline = CreateOpNode( tx, arg_query_tree, dbName, nodeIndex );
        if( pipeline )
        {
            table = ExecuteQueryPipeline( pipeline );
        }
        if( table == nullptr )
        {
            table = std::make_shared< AriesMemTable >();
            table->SetContent( std::make_unique< AriesTableBlock >() );
        }
        return table;
    }

    void AriesEngineByMaterialization::ExecuteUncorrelatedSubQueries( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_query_tree, const std::string& dbName )
    {
        SelectStructure *ss = ( SelectStructure * )( ( arg_query_tree->GetMyQuery() ).get() );

        // aries::CPU_Timer t;
        // // string perfStat;
        // if ( ss->GetInitQueryCount() > 0 )
        // {
        //     // perfStat = "-------------------- sub queries begin\n\n";
        //     t.begin();
        // }

        for( int i = ss->GetInitQueryCount() - 1; i >=0; --i )
        {
            SQLTreeNodePointer stnp = ss->GetInitQueryByIndex( i )->GetQueryPlanTree();
            //std::LOG(INFO) << "\n Now We execute init queries!\n";
            auto result = std::dynamic_pointer_cast< AriesMemTable >( ExecuteQuery( tx, stnp, dbName ) );

            // perfStat += "sub query " + std::to_string( i ) + ":\n" + m_perfStat + "\n\n";

            auto content = result->GetContent();

            /*reset the expr to use the execution result!*/
            BiaodashiPointer bp = ss->GetReplaceExprByIndex( i );
            CommonBiaodashi *expr_p = ( CommonBiaodashi * )( bp.get() );

            if ( expr_p->IsExpectBuffer() )
            {
                // TODO:here only process the buffer of column 1, 
                // if subquery's result has more than one column, other columns are not processed correctly
                if(content->GetColumnCount() > 1)
                    ARIES_EXCEPTION( ER_OPERAND_COLUMNS, content->GetColumnCount() );
                LOG(INFO) << "here convert uncorrelated query's result to buffer, expr: " << expr_p->ToString();
                expr_p->ConvertSelfToBuffer(content->GetColumnBuffer(1));
                continue;
            }

            if ( content->GetRowCount() != 1 || content->GetColumnCount() != 1 ) {
                ARIES_EXCEPTION_SIMPLE( ER_SYNTAX_ERROR, "Sub query with more than one result is not supported yet" );
            }

            auto ret_value = content->GetLiteralValue();

            expr_p->SetIsNullable( true );
            //expr_p->ConvertSelfToFloat( ret_value );
            char decimal_value[64];

            if( ret_value.IsNull )
            {
                LOG(INFO) << "here uncorrelated query's result is null, expr: " << expr_p->ToString();
                expr_p->ConvertSelfToNull(); // empty string for null value
            }
            else
            {
                if ( boost::get< aries_acc::Decimal >( &( ret_value.Value ) ) )
                {
                    expr_p->ConvertSelfToDecimal( boost::get< aries_acc::Decimal >( ret_value.Value ).GetDecimal( decimal_value ) );
                }
                else if ( boost::get< int64_t >( &( ret_value.Value ) ) )
                {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< int64_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< int32_t >( &( ret_value.Value ) ) )
                {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< int32_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< int16_t >( &( ret_value.Value ) ) )
                {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< int16_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< int8_t >( &( ret_value.Value ) ) )
                {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< int8_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< double >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< double >( ret_value.Value ) ) );
                }
                else if ( boost::get< float >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToDecimal( std::to_string( boost::get< float >( ret_value.Value ) ) );
                }
                else if ( boost::get< uint8_t >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( std::to_string( boost::get< uint8_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< uint16_t >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( std::to_string( boost::get< uint16_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< uint32_t >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( std::to_string( boost::get< uint32_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< uint64_t >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( std::to_string( boost::get< uint64_t >( ret_value.Value ) ) );
                }
                else if ( boost::get< std::string >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( boost::get< std::string >( ret_value.Value ) );
                }
                else if ( boost::get< AriesDate >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get< AriesDate >( ret_value.Value ) ) );
                }
                else if ( boost::get< AriesDatetime >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get< AriesDatetime >( ret_value.Value ) ) );
                }
                else if ( boost::get< AriesTime >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( aries_acc::AriesDatetimeTrans::GetInstance().ToString( boost::get< AriesTime >( ret_value.Value ) ) );
                }
                else if ( boost::get< AriesTimestamp >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( aries_acc::AriesDatetimeTrans::GetInstance().ToString( AriesDatetime( boost::get< AriesTimestamp >( ret_value.Value ) ) ) );
                }
                else if ( boost::get< AriesYear >( &( ret_value.Value ) ) ) {
                    expr_p->ConvertSelfToString( std::to_string( boost::get< AriesYear >( ret_value.Value ).getYear() ) );
                }
                else
                {
                    ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, "Sub query is not supported yet" );
                }
            }

        }
        // if ( ss->GetInitQueryCount() > 0 )
        // {
        //     // perfStat = "\n" + std::to_string( t.end() ) + "ms total time of sub queries\n" + perfStat;
        //     // perfStat += "-------------------- sub queries end\n";
        //     // m_perfStat = perfStat;
        // }
    }

    AriesOpNodeSPtr AriesEngineByMaterialization::CreateOpNode( const aries_engine::AriesTransactionPtr& tx,
                                                                SQLTreeNodePointer arg_input_node,
                                                                const std::string& dbName,
                                                                int& nodeId )
    {
        AriesOpNodeSPtr result;
        if( arg_input_node )
        {
            switch( arg_input_node->GetType() )
            {
                case SQLTreeNodeType::Column_NODE:
                {
                    AriesOpNodeSPtr node;
                    if ( m_isUpdate )
                        node = CreateUpdateCalcNode( arg_input_node, nodeId );
                    else
                        node = CreateColumnNode( arg_input_node, nodeId );
                    if( node )
                    {
                        if (arg_input_node->GetTheChild())
                            node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                        result = node;
                    }
                    else
                    {
                        LOG(INFO) << "ColumnNode is nullptr, use child";
                        result = CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId );
                    }
                    break;
                }
                case SQLTreeNodeType::Group_NODE:
                {
                    auto node = CreateGroupNode( arg_input_node, nodeId );
                    node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::Sort_NODE:
                {
                    auto node = CreateSortNode( arg_input_node, nodeId );
                    node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::Filter_NODE:
                {
                    auto node = CreateFilterNode( arg_input_node, nodeId );
                    node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::Table_NODE:
                {
                    result = CreateScanNode( tx, arg_input_node, dbName, nodeId );
                    break;
                }
                case SQLTreeNodeType::BinaryJoin_NODE:
                {
                    auto node = CreateJoinNode( arg_input_node, nodeId );
                    auto left = CreateOpNode( tx, arg_input_node->GetLeftChild(), dbName, ++nodeId );
                    auto right = CreateOpNode( tx, arg_input_node->GetRightChild(), dbName, ++nodeId );
                    node->SetSourceNode( left, right );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::Limit_NODE:
                {
                    auto node = CreateLimitNode( arg_input_node, nodeId );
                    node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                    result = node;
                    if( !node->IsValidSize() )
                    {
                        LOG(ERROR) << "LimitNode's size is invalid!";
                    }
                    break;
                }
                case SQLTreeNodeType::SetOp_NODE:
                {
                    auto node = CreateSetOpNode( arg_input_node, nodeId );
                    auto left = CreateOpNode( tx, arg_input_node->GetLeftChild(), dbName, ++nodeId );
                    auto right = CreateOpNode( tx, arg_input_node->GetRightChild(), dbName, ++nodeId );
                    node->SetSourceNode( left, right );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::SELFJOIN_NODE:
                {
                    auto node = CreateSelfJoinNode( arg_input_node, nodeId );
                    node->SetSourceNode( CreateOpNode( tx, arg_input_node->GetTheChild(), dbName, ++nodeId ) );
                    result = node;
                    break;
                }
                // case SQLTreeNodeType::Insert_NODE:
                // {
                //     ARIES_ASSERT( 0, "should not occur, InsertNode should be handled by CommandExecutor");
                // }
                case SQLTreeNodeType::StarJoin_NODE:
                {
                    auto node = CreateStarJoinNode( tx, arg_input_node, dbName, nodeId );
                    result = node;
                    break;
                }
                case SQLTreeNodeType::Exchange_NODE:
                {
                    // auto node = Create
                    auto node = CreateExchangeNode( tx, arg_input_node, dbName, nodeId );
                    result = node;
                    break;
                }
                default:
                    ARIES_ASSERT( 0, "UnSupported node type: " + std::to_string( ( int )arg_input_node->GetType() ) );

            }

            auto nodeSpoolId = arg_input_node->GetSpoolId();
            if (nodeSpoolId != -1)  // 不等于-1表示query tree有相同的node，需要cache node data
            {
                result->SetSpoolCache( nodeSpoolId, spool_cache_manager );
                spool_cache_manager->increaseSpoolReferenceCount(nodeSpoolId);
                spool_cache_manager->buildSpoolChildrenMap(arg_input_node);
            }
        }

        result->SetUniqueColumnsId( arg_input_node->GetUniqueKeys() );
        return result;
    }

    AriesOpNodeSPtr AriesEngineByMaterialization::CreateScanNode( const aries_engine::AriesTransactionPtr& tx, SQLTreeNodePointer arg_input_node, const std::string& defaultDbName, int nodeId )
    {
        if (arg_input_node->GetTheChild())
        {
            std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
            std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
            for ( auto referencedColumn : referencedColumnArray )
            {
                for ( size_t i = 0; i < childRequiredColumnArray.size(); ++i){
                    if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
                    referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
                    {
                        arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                        break;
                    }
                }
            }
        }
        BasicRelPointer the_basic_rel = arg_input_node->GetBasicRel();
        PhysicalTablePointer the_table = the_basic_rel->GetPhysicalTable();
        vector< int > the_column_output_sequence = arg_input_node->GetColumnOutputSequence();
        string dbName = the_basic_rel->GetDb();
        if( dbName.empty() )
        {
            dbName = defaultDbName;
        }
        string tableName = the_table->GetName();
        auto dbEntry = schema::SchemaManager::GetInstance()->GetSchema()->GetDatabaseByName( dbName );
        auto tableEntry = dbEntry->GetTableByName( tableName );
        {
            auto node =  AriesEngineShell::MakeMvccScanNode( nodeId,
                                                       tx,
                                                       dbName,
                                                       tableName,
                                                       the_column_output_sequence );
            //return AriesEngineShell::MakeOrcScanNode( the_table, the_column_output_sequence );
            if ( arg_input_node->GetSliceCount() > 0 )
            {
                node->SetRange( arg_input_node->GetSliceCount(), arg_input_node->GetSliceIndex() );
            }

            AriesExprBridge bridge;
            for ( const auto& expr : arg_input_node->GetPartitionCondition() )
            {
                ProcessExprByUsingColumnIndex( expr, arg_input_node );
                node->AddPartitionCondition( bridge.Bridge( expr ) );
            }
            return node;
        }
        //return AriesEngineShell::MakeScanNode(the_table, the_column_output_sequence);
    }

    AriesFilterNodeSPtr AriesEngineByMaterialization::CreateFilterNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
        for ( auto referencedColumn : referencedColumnArray )
        {
            for ( size_t i = 0; i < childRequiredColumnArray.size(); ++i){
                if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
                   referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
                {
                    arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                    break;
                }
            }
        }
        BiaodashiPointer the_filter_expr = arg_input_node->GetFilterStructure();
        ProcessExprByUsingColumnIndex( the_filter_expr, arg_input_node );
        vector< int > the_column_output_sequence = arg_input_node->GetColumnOutputSequence();

        return AriesEngineShell::MakeFilterNode( nodeId, the_filter_expr, the_column_output_sequence );
    }

    AriesGroupNodeSPtr AriesEngineByMaterialization::CreateGroupNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        vector< BiaodashiPointer > group_by_exprs;
        vector< BiaodashiPointer > select_exprs;

        // auto query = std::dynamic_pointer_cast< SelectStructure >( arg_input_node->GetMyQuery() );
        // if ( query->GetQueryContext()->type != QueryContextType::TheTopQuery && query->GetQueryContext()->GetParent() )
        // {
        //     query = std::dynamic_pointer_cast< SelectStructure>( query->GetQueryContext()->GetParent()->select_structure );
        // }
        // const auto& alias_map = query->GetSpoolAlias();

        // std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        // std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
        // for ( auto referencedColumn : referencedColumnArray )
        // {
        //     auto it = alias_map.find( referencedColumn->GetTable()->GetMyOutputName() );
        //     std::string alias_table_name;
        //     if ( it != alias_map.cend() )
        //     {
        //         alias_table_name = it->second;
        //     }

        //     for ( int i = 0; i < childRequiredColumnArray.size(); ++i){
        //         if(( *referencedColumn->GetTable() == *childRequiredColumnArray[ i ]->GetTable() || childRequiredColumnArray[ i ]->GetTable()->GetMyOutputName() == alias_table_name ) &&
        //            referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
        //         {
        //             arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
        //             break;
        //         }
        //     }
        // }

        GroupbyStructurePointer gbsp = arg_input_node->GetMyGroupbyStructure();
        for( size_t gi = 0; gi < gbsp->GetGroupbyExprCount(); gi++ )
        {
            BiaodashiPointer a_expr = gbsp->GetGroupbyExpr( gi );
            auto expr = std::dynamic_pointer_cast<CommonBiaodashi>(a_expr);
            auto real = expr->GetRealExprIfAlias();
            if ( real )
            {
                ProcessExprByUsingColumnIndex( real, arg_input_node );
                group_by_exprs.push_back( real );
            }
            else
            {
                ProcessExprByUsingColumnIndex( a_expr, arg_input_node );
                group_by_exprs.push_back( a_expr );
            }
        }

        SelectPartStructurePointer spsp = arg_input_node->GetMySelectPartStructure();
        std::vector< BiaodashiPointer > vbp = spsp->GetAllExprs();
        for( size_t vbpi = 0; vbpi < vbp.size(); vbpi++ )
        {
            BiaodashiPointer a_expr = vbp[vbpi];
            ProcessExprByUsingColumnIndex( a_expr, arg_input_node );
            select_exprs.push_back( a_expr );
        }

        /*we could have more select exprs*/
        std::vector< BiaodashiPointer > additional_agg_expr_array = gbsp->GetAdditionalExprsForSelect();
        for( size_t aaea_i = 0; aaea_i < additional_agg_expr_array.size(); aaea_i++ )
        {
            BiaodashiPointer a_expr = additional_agg_expr_array[aaea_i];
            ProcessExprByUsingColumnIndex( a_expr, arg_input_node );
            select_exprs.push_back( a_expr );
        }

        return AriesEngineShell::MakeGroupNode( nodeId, group_by_exprs, select_exprs );
    }

    AriesSortNodeSPtr AriesEngineByMaterialization::CreateSortNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
        for ( auto referencedColumn : referencedColumnArray )
        {
            for ( size_t i = 0; i < childRequiredColumnArray.size(); ++i){
                if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
                   referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
                {
                    arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                    break;
                }
            }
        }
        vector< BiaodashiPointer > the_order_by_exprs;
        vector< OrderbyDirection > the_order_by_directions;

        OrderbyStructurePointer osp = arg_input_node->GetMyOrderbyStructure();
        for( size_t oi = 0; oi < osp->GetOrderbyItemCount(); oi++ )
        {
            BiaodashiPointer a_expr = osp->GetOrderbyItem( oi );
            OrderbyDirection a_ord = osp->GetOrderbyDirection( oi );

            ProcessExprByUsingColumnIndex( a_expr, arg_input_node );

            the_order_by_exprs.push_back( a_expr );
            the_order_by_directions.push_back( a_ord );
        }

        return AriesEngineShell::MakeSortNode( nodeId, the_order_by_exprs, the_order_by_directions, arg_input_node->GetColumnOutputSequence() );
    }

    void AriesEngineByMaterialization::checkJoinConditions( BiaodashiPointer expr, SQLTreeNodePointer arg_input_node, bool b_check_equal_condition )
    {
        ProcessExprByUsingColumnIndex( expr, arg_input_node );
        BiaodashiAuxProcessor processor;
        auto conditions = processor.generate_and_list( expr );
        for ( const auto& condition : conditions )
        {
            auto expression = ( CommonBiaodashi* )( condition.get() );

            while ( expression->GetType() == BiaodashiType::Kuohao )
            {
                expression = ( CommonBiaodashi* )( expression->GetChildByIndex( 0 ).get() );
            }

            if ( expression->GetType() != BiaodashiType::Bijiao )
            {
                continue;
            }

            auto comparison_type = static_cast< ComparisonType >( boost::get< int >( expression->GetContent() ) );
            if ( comparison_type != ComparisonType::DengYu )
            {
                continue;
            }

            auto left = ( CommonBiaodashi* )( expression->GetChildByIndex( 0 ).get() );
            auto right = ( CommonBiaodashi* )( expression->GetChildByIndex( 1 ).get() );

            if ( left->GetValueType() != right->GetValueType() )
            {
                ThrowNotSupportedException( "compare columns with different value types: " +
                                            get_name_of_value_type( left->GetValueType() ) + " and " +
                                            get_name_of_value_type( right->GetValueType() ) );
            }

            if( b_check_equal_condition )
            {
                switch( left->GetValueType() )
                {
                    case BiaodashiValueType::DECIMAL:
                    {
                        ThrowNotSupportedException( "join on decimal type" );
                        break;
                    }
                    default:
                        break;
                }
            }

            break;
        }
    }

    AriesJoinNodeSPtr AriesEngineByMaterialization::CreateJoinNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        std::vector< ColumnShellPointer > leftChildRequiredColumnArray = arg_input_node->GetLeftChild()->GetRequiredColumnArray();
        std::vector< ColumnShellPointer > rightChildRequiredColumnArray = arg_input_node->GetRightChild()->GetRequiredColumnArray();

        for ( auto referencedColumn : referencedColumnArray )
        {
            bool finded = false;
            for ( size_t i = 0; i < leftChildRequiredColumnArray.size(); ++i)
            {
                if(referencedColumn->GetTableName()==leftChildRequiredColumnArray[i]->GetTableName() &&
                   referencedColumn->GetColumnName()==leftChildRequiredColumnArray[i]->GetColumnName())
                {
                    arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                    finded = true;
                }
                if ( finded )
                    break;
            }
            if ( finded )
                continue;
            for ( size_t i = 0; i < rightChildRequiredColumnArray.size(); ++i)
            {
                if(referencedColumn->GetTableName()==rightChildRequiredColumnArray[i]->GetTableName() &&
                   referencedColumn->GetColumnName()==rightChildRequiredColumnArray[i]->GetColumnName())
                {
                    arg_input_node->SetPositionForReferencedColumn( referencedColumn, -(i+1) );
                    finded = true;
                }
                if ( finded )
                    break;
            }

        }
        AriesJoinNodeSPtr node;
        BiaodashiPointer equal_condition = arg_input_node->GetJoinCondition();
        if (equal_condition)
        {
            checkJoinConditions( equal_condition, arg_input_node, true );

            auto equal_expression = std::dynamic_pointer_cast< CommonBiaodashi >( equal_condition );
            if ( equal_expression->IsEqualCondition() )
            {
                auto left = arg_input_node->GetLeftChild();
                auto right = arg_input_node->GetRightChild();

                auto left_tables = left->GetInvolvedTableList();
                auto right_tables = right->GetInvolvedTableList();

                auto left_condition = std::dynamic_pointer_cast< CommonBiaodashi >( equal_expression->GetChildByIndex( 0 ) );
                auto right_condition = std::dynamic_pointer_cast< CommonBiaodashi >( equal_expression->GetChildByIndex( 1 ) );

                bool need_swap = true;
                for ( const auto& table : left_condition->GetInvolvedTableList() )
                {
                    for ( const auto& table2 : left_tables )
                    {
                        if ( table == table2 )
                        {
                            need_swap = false;
                            break;
                        }
                    }

                    if ( need_swap )
                    {
                        break;
                    }
                }
                if ( need_swap )
                {
                    equal_expression->SwitchChild();
                }
            }
        }
        BiaodashiPointer other_condition = arg_input_node->GetJoinOtherCondition();
        if (other_condition) {
            checkJoinConditions( other_condition, arg_input_node, false );
        }

        JoinType the_join_type = arg_input_node->GetJoinType();
        vector< int > the_column_output_sequence = arg_input_node->GetColumnOutputSequence();
        bool left_as_hash = false;
        bool right_as_hash = false;
        auto can_use_hash = arg_input_node->CanUseHashJoin( left_as_hash, right_as_hash );

        if( the_join_type == JoinType::InnerJoin )
        {
            int join_hint = arg_input_node->GetPrimaryForeignJoinForm();
            bool join_hint_2 = arg_input_node->GetPrimaryTableIntact();
            node = AriesEngineShell::MakeJoinNode( nodeId, equal_condition, other_condition, join_hint, join_hint_2, the_column_output_sequence );
        }
        else
        {
            node = AriesEngineShell::MakeJoinNodeComplex( nodeId, equal_condition, other_condition, the_join_type, the_column_output_sequence );
        }



        if ( can_use_hash )
        {
            HashJoinInfo hash_join_info;
            int hash_key_child_index;
            if ( left_as_hash )
            {
                node->SetHashJoinType( AriesJoinNode::HashJoinType::LeftAsHash );
                node->SetHashJoinInfo( arg_input_node->GetLeftHashJoinInfo() );
                hash_join_info = arg_input_node->GetLeftHashJoinInfo();
                hash_key_child_index = 0;
            }
            else
            {
                node->SetHashJoinType( AriesJoinNode::HashJoinType::RightAsHash );
                node->SetHashJoinInfo( arg_input_node->GetRightHashJoinInfo() );
                hash_join_info = arg_input_node->GetRightHashJoinInfo();
                hash_key_child_index = 1;
            }

            auto hash_child_node = arg_input_node->GetChildByIndex( hash_key_child_index );
            auto value_child_node = arg_input_node->GetChildByIndex( 1 - hash_key_child_index );

            std::vector< int > unique_keys;
            std::vector< int > hash_value_keys;
            for ( const auto& condition : hash_join_info.EqualConditions )
            {
                auto hash_child = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( hash_key_child_index ) );
                auto value_child = std::dynamic_pointer_cast< CommonBiaodashi >( condition->GetChildByIndex( 1 - hash_key_child_index ) );
                ARIES_ASSERT( hash_child->GetType() == BiaodashiType::Lie, "invalid child type" );
                ARIES_ASSERT( value_child->GetType() == BiaodashiType::Lie, "invalid child type" );
                auto column = boost::get< ColumnShellPointer >( hash_child->GetContent() );
                unique_keys.emplace_back( abs( arg_input_node->GetPositionForReferencedColumn( column ) ) );

                column = boost::get< ColumnShellPointer >( value_child->GetContent() );
                hash_value_keys.emplace_back( abs( arg_input_node->GetPositionForReferencedColumn( column ) ) );
            }

            node->SetUniqueKeys( unique_keys );
            node->SetHashValueKeys( hash_value_keys );
        }

        node->SetJoinConditionConstraintType( arg_input_node->GetJoinConditionConstraintType() );
        node->SetIsNotIn( arg_input_node->GetIsNotInFlag() );
        return node;
    }

    AriesUpdateCalcNodeSPtr AriesEngineByMaterialization::CreateUpdateCalcNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
        for ( auto referencedColumn : referencedColumnArray )
        {
            for ( size_t i = 0; i < childRequiredColumnArray.size(); ++i){
                if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
                   referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
                {
                    arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                    break;
                }
            }
        }
        std::vector< BiaodashiPointer > the_select_exprs = arg_input_node->GetExprs4ColumnNode();
        std::vector< BiaodashiPointer > real_exprs;

        for( size_t i = 0; i < the_select_exprs.size(); i++ )
        {
            BiaodashiPointer a_expr = the_select_exprs[i];
            this->ProcessExprByUsingColumnIndex( a_expr, arg_input_node );
            real_exprs.push_back( a_expr );
        }
        vector< BiaodashiPointer > updateValueExprs;
        // for update t set f1 = expr1, f2 = expr2,
        // the select statement is:
        // select __rateup_rowid__, f1, f2, expr1, expr2 from ...
        // rowId column
        updateValueExprs.push_back( real_exprs[ 0 ] );
        for( size_t i = ( ( real_exprs.size() + 1 ) >> 1 ); i < real_exprs.size(); i++ )
        {
            updateValueExprs.push_back( real_exprs[ i ] );
        }
        std::vector< int > the_columns_id( updateValueExprs.size() );
        std::iota( the_columns_id.begin(), the_columns_id.end(), 1 );
        auto node = AriesEngineShell::MakeUpdateCalcNode( nodeId, updateValueExprs, the_columns_id );
        return node;

    }
    AriesColumnNodeSPtr AriesEngineByMaterialization::CreateColumnNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        if(arg_input_node->GetTheChild())
        {
            std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
            std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
            for ( auto referencedColumn : referencedColumnArray )
            {
                for ( size_t i = 0; i < childRequiredColumnArray.size(); ++i){
                    if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
                    referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
                    {
                        arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
                        break;
                    }
                }
            }
        }
        AriesColumnNodeSPtr node;

        if( !arg_input_node->IsColumnNodeRemovable() )
        {
            std::vector< BiaodashiPointer > the_select_exprs;
            std::vector< int > the_columns_id;

            if( arg_input_node->GetForwardMode4ColumnNode() )
            {
                the_columns_id = arg_input_node->GetColumnOutputSequence();
                node = AriesEngineShell::MakeColumnNode( nodeId, the_select_exprs, the_columns_id, 1 ); //1 means that we only do a forward by the_columns_id;
                LOG(INFO) << "here created a forward ColumnNode";
            }
            else
            {
                the_select_exprs = arg_input_node->GetExprs4ColumnNode();
                std::vector< BiaodashiPointer > real_exprs;

                for( size_t i = 0; i < the_select_exprs.size(); i++ )
                {
                    BiaodashiPointer a_expr = the_select_exprs[i];
                    this->ProcessExprByUsingColumnIndex( a_expr, arg_input_node );
                    real_exprs.push_back( a_expr );
                }
                node = AriesEngineShell::MakeColumnNode( nodeId, real_exprs, the_columns_id, 0 ); //0 means that we need to output by the_select_exprs;
            }
        }
        else
        {
            //assert( 0 ); //FIXME: need handle this
        }
        return node;
    }

    AriesOutputNodeSPtr AriesEngineByMaterialization::CreateOutputNode( AriesOpNodeSPtr dataSource )
    {
        assert( dataSource );
        AriesOutputNodeSPtr node = AriesEngineShell::MakeOutputNode();
        node->SetSourceOpNode( dataSource );
        return node;
    }

    AriesLimitNodeSPtr AriesEngineByMaterialization::CreateLimitNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        auto node = AriesEngineShell::MakeLimitNode( nodeId, arg_input_node->GetLimitOffset(), arg_input_node->GetLimitSize() );
        return node;
    }

    AriesStarJoinNodeSPtr AriesEngineByMaterialization::CreateStarJoinNode( const aries_engine::AriesTransactionPtr& tx,
        SQLTreeNodePointer arg_input_node,
        const std::string& dbName,
        int& nodeId )
    {
        auto node = std::make_shared< AriesStarJoinNode >();

        std::vector< ColumnShellPointer > referencedColumnArray = arg_input_node->GetReferencedColumnArray();
        // std::vector< ColumnShellPointer > childRequiredColumnArray = arg_input_node->GetTheChild()->GetRequiredColumnArray();
        for ( auto referencedColumn : referencedColumnArray )
        {
        //     for ( int i = 0; i < childRequiredColumnArray.size(); ++i){
        //         if(referencedColumn->GetTableName()==childRequiredColumnArray[i]->GetTableName() &&
        //            referencedColumn->GetColumnName()==childRequiredColumnArray[i]->GetColumnName())
        //         {
        //             arg_input_node->SetPositionForReferencedColumn( referencedColumn, i+1 );
        //             break;
        //         }
        //     }
        }

        auto required_columns = arg_input_node->GetRequiredColumnArray();
        auto referenced_columns = arg_input_node->GetReferencedColumnArray();

        auto center_required_columns = arg_input_node->GetChildByIndex( 0 )->GetRequiredColumnArray();

        auto conditions = arg_input_node->GetStarJoinConditions();

        std::vector< std::vector< int > > fact_key_ids( conditions.size() );
        std::vector< std::vector< int > > dimension_key_ids( conditions.size() );

        for ( size_t i = 0; i < conditions.size(); i++ )
        {
            auto& condition_array = conditions[ i ];
            for ( const auto& condition : condition_array )
            {
                auto* left_condition = ( CommonBiaodashi* )( condition->GetChildByIndex( 0 ).get() );
                auto* right_condition = ( CommonBiaodashi* )( condition->GetChildByIndex( 1 ).get() );

                auto left_condition_columns = left_condition->GetAllReferencedColumns();
                auto right_condition_columns = right_condition->GetAllReferencedColumns();
                ARIES_ASSERT( left_condition_columns.size() == 1, "two many columns in left condition" );
                ARIES_ASSERT( right_condition_columns.size() == 1, "two many columns in right condition" );

                bool is_left = false;
                bool is_right = false;
                int index = -1;
                for ( size_t j = 0; j < center_required_columns.size(); j++ )
                {
                    const auto& required_column = center_required_columns[ j ];
                    if ( *required_column->GetTable() == *left_condition_columns[ 0 ]->GetTable() &&
                        required_column->GetColumnName() == left_condition_columns[ 0 ]->GetColumnName() )
                    {
                        is_left = true;
                        index = j;
                        break;
                    }
                    else if ( *required_column->GetTable() == *right_condition_columns[ 0 ]->GetTable() &&
                        required_column->GetColumnName() == right_condition_columns[ 0 ]->GetColumnName() )
                    {
                        is_right = true;
                        index = j;
                        break;
                    }
                }

                ARIES_ASSERT( is_left || is_right, "invalid condition for star join node" );
                fact_key_ids[ i ].emplace_back( index + 1 );

                ColumnShellPointer hash_column;
                /**
                 * 条件左边的 column 来自 center 节点
                 */
                if ( is_left )
                {
                    hash_column = right_condition_columns[ 0 ];
                }
                else
                {
                    hash_column = left_condition_columns[ 0 ];
                }

                auto hash_node = arg_input_node->GetChildByIndex( i + 1 );
                auto hash_required_columns = hash_node->GetRequiredColumnArray();

                index = -1;
                for ( size_t j = 0; j < hash_required_columns.size(); j++ )
                {
                    const auto& required_column = hash_required_columns[ j ];
                    if ( *required_column->GetTable() == *hash_column->GetTable() &&
                        required_column->GetColumnName() == hash_column->GetColumnName() )
                    {
                        index = j;
                        break;
                    }
                }

                ARIES_ASSERT( index != -1, "cannot find column in hash node" );
                dimension_key_ids[ i ].emplace_back( index + 1 );
            }
        }

        node->SetFactKeyIds( fact_key_ids );
        node->SetDimensionKeyIds( dimension_key_ids );

        std::vector< int > fact_output_ids;
        std::vector< std::vector< int > > dimension_output_ids( arg_input_node->GetChildCount() - 1 );

        std::vector< int > size_of_required;
        for ( size_t i = 0; i < arg_input_node->GetChildCount(); i++ )
        {
            if ( i == 0 )
            {
                size_of_required.emplace_back( arg_input_node->GetChildByIndex( i )->GetRequiredColumnArray().size() );
            }
            else
            {
                size_of_required.emplace_back( arg_input_node->GetChildByIndex( i )->GetRequiredColumnArray().size() + size_of_required[ i - 1 ] );
            }
        }

        std::map< int, int > ids_map;
        std::vector< std::map< int, int > > dimension_maps( arg_input_node->GetChildCount() - 1 );

        for ( size_t id = 0; id < required_columns.size(); id++ )
        {
            auto& column = required_columns[ id ];
            auto pos = arg_input_node->GetPositionForReferencedColumn( column );
            size_t i = 0;
            for ( ; i < size_of_required.size(); i++ )
            {
                if ( size_of_required[ i ] >= pos )
                {
                    break;
                }
            }

            auto child = arg_input_node->GetChildByIndex( i );

            pos = 1;
            for ( const auto& col : child->GetRequiredColumnArray() )
            {
                if ( *column->GetTable() == *col->GetTable() && column->GetColumnName() == col->GetColumnName() )
                {
                    break;
                }
                pos ++;
            }

            if ( i > 0 )
            {
                dimension_output_ids[ i - 1 ].emplace_back( pos );
                dimension_maps[ i - 1 ][ id + 1 ] = pos;
            }
            else
            {
                fact_output_ids.emplace_back( pos );
                ids_map[ id + 1 ] = pos;
            }

        }

        node->SetFactOutputColumnsId( fact_output_ids );
        node->SetDimensionOutputColumnsId( dimension_output_ids );
        node->SetOuputIdsMap( ids_map );
        node->SetDimensionIdsMaps( dimension_maps );

        node->SetNodeId( nodeId );

        auto fact_source_node = CreateOpNode( tx, arg_input_node->GetChildByIndex( 0 ), dbName, ++nodeId );
        node->SetFactSourceNode( fact_source_node );
        std::vector< std::vector< int > > dimension_columns_ids;
        for ( size_t i = 1; i < arg_input_node->GetChildCount(); i++ )
        {
            auto child = arg_input_node->GetChildByIndex( i );
            auto dimension_node = CreateOpNode( tx, child, dbName, ++nodeId );
            node->AddDimensionSourceNode( dimension_node );
        }
        return node;
    }

    AriesExchangeNodeSPtr
    AriesEngineByMaterialization::CreateExchangeNode( const aries_engine::AriesTransactionPtr& tx,
                                                      SQLTreeNodePointer arg_input_node,
                                                      const std::string& db_name,
                                                      int nodeId )
    {
        auto node = AriesEngineShell::MakeExchangeNode( nodeId, arg_input_node->GetTargetDeviceId(), arg_input_node->GetSourceDevicesId() );
        for ( size_t i = 0; i < arg_input_node->GetChildCount();i ++ )
        {
            auto child = arg_input_node->GetChildByIndex( i );
            node->AddSourceNode( CreateOpNode( tx, child, db_name, nodeId ) );
        }
        return node;
    }

    AriesSetOperationNodeSPtr AriesEngineByMaterialization::CreateSetOpNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        return AriesEngineShell::MakeSetOpNode( nodeId, arg_input_node->GetSetOperationType() );
    }

    AriesSelfJoinNodeSPtr AriesEngineByMaterialization::CreateSelfJoinNode( SQLTreeNodePointer arg_input_node, int nodeId )
    {
        return AriesEngineShell::MakeSelfJoinNode( nodeId, arg_input_node->GetSelfJoinColumnId(), arg_input_node->GetSelfJoinMainFilter(),
                arg_input_node->GetSelfJoinInfo(), arg_input_node->GetColumnOutputSequence() );
    }

    void AriesEngineByMaterialization::ProcessColumn( CommonBiaodashi *p_expr, SQLTreeNodePointer arg_input_node )
    {
        ColumnShellPointer csp;

        try
        {
            csp = boost::get< ColumnShellPointer >( p_expr->GetContent() );
            int my_position = arg_input_node->GetPositionForReferencedColumn( csp );
            csp->SetPositionInChildTables( my_position );

            if( csp->GetExpr4Alias() != nullptr )
            {
                BiaodashiPointer the_real_expr = csp->GetExpr4Alias();
                CommonBiaodashi *the_real_expr_rawpointer = ( CommonBiaodashi * )( the_real_expr.get() );
                p_expr->SetValueType( the_real_expr_rawpointer->GetValueType() );
            }
        }
        catch( ... )
        {
            ARIES_ASSERT( 0, "I cannot find the required ColumnShell -->" + csp->ToString() );
        }
    }

    void AriesEngineByMaterialization::ProcessExprByUsingColumnIndex( BiaodashiPointer arg_expr, SQLTreeNodePointer arg_input_node )
    {
        if( arg_expr == nullptr )
            return;

        CommonBiaodashi *p_expr = ( CommonBiaodashi * )( arg_expr.get() );
        switch( p_expr->GetType() )
        {
            case BiaodashiType::Zhengshu:
            case BiaodashiType::Fudianshu:
            case BiaodashiType::Decimal:
            case BiaodashiType::Zhenjia:
            case BiaodashiType::Zifuchuan:
            case BiaodashiType::Star:
            case BiaodashiType::Null:
            case BiaodashiType::IntervalExpression:
            case BiaodashiType::Distinct:
                /*do nothing*/
                break;
            case BiaodashiType::Hanshu:
            case BiaodashiType::Shuzu:
            case BiaodashiType::Yunsuan:
            case BiaodashiType::Bijiao:
            case BiaodashiType::Andor:
            case BiaodashiType::Qiufan:
            case BiaodashiType::Kuohao:
            case BiaodashiType::Cunzai:
            case BiaodashiType::Likeop:
            case BiaodashiType::Inop:
            case BiaodashiType::NotIn:
            case BiaodashiType::Between:
            case BiaodashiType::Case:
            case BiaodashiType::IfCondition:
            case BiaodashiType::IsNotNull:
            case BiaodashiType::IsNull:
            case BiaodashiType::SQLFunc:
            case BiaodashiType::ExprList:
                for( size_t ci = 0; ci < p_expr->GetChildrenCount(); ci++ )
                {
                    BiaodashiPointer a_child = p_expr->GetChildByIndex( ci );
                    this->ProcessExprByUsingColumnIndex( a_child, arg_input_node );
                }
                break;
            case BiaodashiType::Lie:
                this->ProcessColumn( p_expr, arg_input_node );
                break;
            case BiaodashiType::Buffer:
                break;
            case BiaodashiType::Query:
            {
                string errMsg( "correlated sub-query" );
                auto parentExpr = p_expr->GetParent();
                if ( parentExpr )
                {
                    CommonBiaodashi *parentCommonExpr = ( CommonBiaodashi* )parentExpr;
                    auto exprName = get_name_of_expr_type( parentCommonExpr->GetType() );
                    errMsg.append( " in '" ).append( exprName ).append( "' expression");
                }
                ARIES_EXCEPTION( ER_NOT_SUPPORTED_YET, errMsg.data() );
                break;
            }
            default: {
                string msg = "EngineByMaterialization::ProcessExprByUsingColumnIndex---> Unsupported type: ";
                msg.append(get_name_of_expr_type(p_expr->GetType()));
                ARIES_ASSERT( 0, msg );
            }

        }

    }
} /* namespace aries */
