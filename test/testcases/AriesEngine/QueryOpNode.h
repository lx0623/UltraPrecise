#include "AriesEngine/AriesOpNode.h"

namespace aries_test
{

class QueryOpNode : public aries_engine::AriesOpNode
{
public:
    QueryOpNode( const std::string& sql, const std::string& dbName );
    ~QueryOpNode() = default;


    virtual bool Open() override;
    virtual aries_engine::AriesOpResult GetNext() override;
    virtual void Close() override;

private:
    std::string m_sql;
    std::string m_dbName;
    aries_engine::AriesTableBlockUPtr m_table;
};

} // namespace aries_test