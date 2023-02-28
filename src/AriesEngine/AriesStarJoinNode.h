#pragma once

#include <deque>
#include "AriesOpNode.h"
#include "CudaAcc/AriesEngineException.h"
#include "frontend/SQLTreeNode.h"

using namespace aries_acc;

BEGIN_ARIES_ENGINE_NAMESPACE

class AriesStarJoinNode : public AriesOpNode
{

private:
    AriesOpNodeSPtr factSourceNode;
    std::vector< AriesOpNodeSPtr > dimensionSourceNodes;
    std::vector< int > fact_output_ids;
    std::vector< std::vector< int > > output_columns_ids;
    std::vector< AriesTableBlockUPtr > dimension_tables;
    std::vector< AriesHashTableUPtr > dimension_hash_tables;
    std::vector< AriesHashTableMultiKeysUPtr > multi_key_hash_tables;

    std::vector< std::vector< int > > fact_key_ids;
    std::vector< std::vector< int > > dimension_key_ids;

    std::map< int, int > output_ids_map;
    std::vector< std::map< int, int > > dimension_ids_map;

    std::vector< size_t > rows_count;

    std::vector< AriesTableBlockStats > dimensionStats;
    AriesTableBlockStats factStats;

public:
    AriesStarJoinNode();
    void SetFactSourceNode( const AriesOpNodeSPtr& node );
    void AddDimensionSourceNode( const AriesOpNodeSPtr& node );

    void SetFactOutputColumnsId( const std::vector< int >& ids );
    void SetDimensionOutputColumnsId( const std::vector< std::vector< int > >& ids );

    virtual bool Open();
    virtual void Close();

    virtual void SetSourceNode( AriesOpNodeSPtr leftSource, AriesOpNodeSPtr rightSource = nullptr ) override
    {
        ARIES_ASSERT( 0, "should use SetFactSourceNode or AddDimensionSourceNode" );
    }

    virtual void SetCuModule( const std::vector<aries_acc::CUmoduleSPtr>& modules )
    {
        factSourceNode->SetCuModule( modules );
        for ( auto& source : dimensionSourceNodes )
        {
            source->SetCuModule( modules );
        }
    }

    virtual string GetCudaKernelCode() const
    {
        auto code = factSourceNode->GetCudaKernelCode();
        for ( auto& source : dimensionSourceNodes )
        {
            code += source->GetCudaKernelCode();
        }

        return code;
    }

    virtual AriesOpResult GetNext() override;

    AriesTableBlockUPtr ReadAllData( AriesOpNodeSPtr dataSource );

    void SetFactKeyIds( const std::vector< std::vector< int > >& ids );

    void SetDimensionKeyIds( const std::vector< std::vector< int > >& ids );

    virtual JSON GetProfile() const override;

    void SetOuputIdsMap( const std::map< int, int >& map );

    void SetDimensionIdsMaps( const std::vector< std::map< int, int > >& maps );

    virtual AriesTableBlockUPtr GetEmptyTable() const override final;
};

using AriesStarJoinNodeSPtr = std::shared_ptr< AriesStarJoinNode >;

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */
