#pragma once
#include <vector>
#include <memory>
#include "BasicRel.h"
#include "AbstractQuery.h"
#include "SQLIdent.h"
#include "AbstractBiaodashi.h"
using namespace std;
namespace aries {
class DeleteStructure
{
public:
    DeleteStructure(){}
    ~DeleteStructure();

    void AddTargetTable( const BasicRelPointer& table )
    {
        m_targetTables.push_back( table );
    }

    void SetTargetTables( const vector< BasicRelPointer >& tables )
    {
        m_targetTables = tables;
    }

    void SetSelectStructure( const AbstractQueryPointer& selectStructure )
    {
        m_selectStructure = selectStructure;
    }
    AbstractQueryPointer GetSelectStructure() const
    {
        return m_selectStructure;
    }
    vector< BasicRelPointer > GetTargetTables() const
    {
        return m_targetTables;
    }
private:
    vector< BasicRelPointer > m_targetTables;
    AbstractQueryPointer m_selectStructure;
};
using DeleteStructurePtr = shared_ptr< DeleteStructure >;
}