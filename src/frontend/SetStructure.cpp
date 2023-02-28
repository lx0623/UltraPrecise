//
// Created by tengjp on 19-9-4.
//

#include "server/mysql/include/set_var.h"
#include "SetStructure.h"
#include "AriesEngineWrapper/AriesExprBridge.h"
using aries_engine::AriesExprBridge;

NAMESPACE_ARIES_START
void CheckExprForSetStatements( CommonBiaodashiPtr& expr, const string& defaultDb );
string ToString(const SET_CMD& showCmd) {
    switch (showCmd) {
        case SET_CMD::SET_USER_VAR:
            return "set user variable";
            break;
        case SET_CMD::SET_SYS_VAR:
            return "show system variable";
            break;
        case SET_CMD::SET_CHAR_SET:
            return "set character set";
            break;
        case SET_CMD::SET_NAMES:
            return "set names";
            break;
        case SET_CMD::SET_PASSWORD:
            return "set password";
            break;
        case SET_CMD::SET_TX:
            return "set transaction";
            break;
        default:
            return "unknown set command";
            break;
    }
}

void SetSysVarStructure::Check( THD* thd )
{
    sys_var* sysVar = find_sys_var( m_sysVarStructurePtr->varName.data() );
    if (!sysVar) {
        ARIES_EXCEPTION( ER_UNKNOWN_SYSTEM_VARIABLE, m_sysVarStructurePtr->varName.data() );
    }
    m_sysVar = sysVar;

    if ( m_sysVar->is_readonly() ) {
        ARIES_EXCEPTION( ER_INCORRECT_GLOBAL_LOCAL_VAR,
                         m_sysVar->name.data(),
                         "read only");
    }
    if ( !m_sysVar->check_scope( m_sysVarStructurePtr->varScope ) )
    {
        int err =
            ( m_sysVarStructurePtr->varScope == OPT_GLOBAL ? ER_LOCAL_VARIABLE
                                                           : ER_GLOBAL_VARIABLE );
        ARIES_EXCEPTION( err, m_sysVar->name.data() );
    }
    /* value is a NULL pointer if we are using SET ... = DEFAULT */
    if ( !m_valueExpr )
        return;

    if ( BiaodashiType::Biaoshifu == m_valueExpr->GetType() )
    {
        auto content = m_valueExpr->GetContent();
        SQLIdentPtr ident = boost::get<SQLIdentPtr>( content );
        if ( !ident->table.empty() )
        {
            ARIES_EXCEPTION( ER_WRONG_TYPE_FOR_VAR, m_sysVar->name.data() );
        }
        m_valueExpr = std::make_shared< CommonBiaodashi >( BiaodashiType::Zifuchuan, ident->id );
        m_valueExpr->SetValueType( BiaodashiValueType::TEXT );
    }

    AriesExprBridge bridge;
    CheckExprForSetStatements( m_valueExpr, "" );
    m_valueCommonExpr = bridge.Bridge( m_valueExpr );
    if ( m_sysVar->check_update_type( m_valueExpr->GetValueType() ) )
    {
        ARIES_EXCEPTION(ER_WRONG_TYPE_FOR_VAR, m_sysVar->name.data() );
    }
    m_sysVar->check( thd, this );
}
void SetSysVarStructure::Update( THD* thd )
{
    m_valueExpr ? ( void )m_sysVar->update( thd, this ) : m_sysVar->set_default( thd, this );
}
void SetUserVarStructure::Check( THD* thd )
{

}
void SetPasswordStructure::Check( THD* thd )
{

}
NAMESPACE_ARIES_END // namespace aries
