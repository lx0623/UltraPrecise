//
// Created by tengjp on 19-11-25.
//

#ifndef AIRES_ADMINSTMTSTRUCTURE_H
#define AIRES_ADMINSTMTSTRUCTURE_H

NAMESPACE_ARIES_START
enum ADMIN_STMT {
    // BINLOG,
    // CACHE_INDEX,
    // FLUSH,
    KILL,
    // LOAD_INDEX,
    // RESET,
    SHUTDOWN,

};
class AdminStmtStructure {
public:
    AdminStmtStructure(ADMIN_STMT stmt) : adminStmt(stmt)
    {
    }
    virtual ~AdminStmtStructure() {}
    ADMIN_STMT adminStmt;
};
using AdminStmtStructurePtr = std::shared_ptr<AdminStmtStructure>;

class KillStructure : public AdminStmtStructure {
public:
    KillStructure() : AdminStmtStructure(ADMIN_STMT::KILL)
    {
    }
    int killOpt;
    BiaodashiPointer procIdExpr;

};
using KillStructurePtr = std::shared_ptr<KillStructure>;
NAMESPACE_ARIES_END // namespace aries
#endif //AIRES_ADMINSTMTSTRUCTURE_H
