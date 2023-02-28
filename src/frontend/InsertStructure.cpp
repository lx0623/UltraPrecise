#include "InsertStructure.h"
#include "SelectStructure.h"
namespace aries {
InsertStructure::InsertStructure( const BasicRelPointer& tableIdent )
{
    m_dbName = tableIdent->GetDb();
    m_tableName = tableIdent->GetID();

}
InsertStructure::~InsertStructure()
{
    for ( auto& v : m_insertValuesList )
    {
        v->clear();
        v = nullptr;
    }
    m_insertValuesList.clear();

    if ( m_insertRowsSelectStructure )
    {
        m_insertRowsSelectStructure = nullptr;
    }

    if ( m_selectStructure )
    {
        m_selectStructure = nullptr;
    }
}
void InsertStructure::SetInsertColumns( const vector< BiaodashiPointer >& insertColumns )
{
    m_insertColumns = insertColumns;
}
void InsertStructure::SetInsertColumnValues( const VALUES_LIST& insertValuesList )
{
    m_insertValuesList = insertValuesList;
}
void InsertStructure::SetInsertColumnValues( VALUES_LIST&& insertValuesList )
{
    m_insertValuesList = std::move( insertValuesList );
}
void InsertStructure::SetOptUpdateColumnValues( const vector< BiaodashiPointer >& updateColumns,
                                                const vector< BiaodashiPointer >& updateValues )
{
    m_optUpdateColumns = updateColumns;
    m_optUpdateValues = updateValues; 
}
}
