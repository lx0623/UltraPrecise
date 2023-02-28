#ifndef ARIES_VARIOUS_ENUM
#define ARIES_VARIOUS_ENUM

#include "schema/Schema.h"

namespace aries {

enum CommandType {
    NullCommand = 100000,
    ChangeDatabase,
    CreateDatabase,
    DropDatabase,
    CreateTable,
    DropTable,
    CopyTable,
    InsertQuery,
    CreateView,
    DropView,
    CreateUser,
    DropUser
};

enum QueryContextType {
    TheTopQuery,

    FromSubQuery,

    ExprSubQuery,

    GroupbySubQuery,

    OrderbySubQuery,

    SetQueryHalf

};


enum ExprContextType {

    JoinOnExpr,

    WhereExpr,

    GroupbyExpr,

    HavingExpr,

    SelectExpr,

    OrderbyExpr,

    VirtualExpr

};


enum ComparisonType {
    DengYu,
    BuDengYu,
    SQLBuDengYu,
    XiaoYuDengYu,
    DaYuDengYu,
    XiaoYu,
    DaYu

};

enum CalcType {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD
};

enum LogicType {
    AND,
    OR
};

enum BiaodashiType {
    Zhengshu = 3000,
    Fudianshu,
    Zifuchuan,
    Decimal,

    Biaoshifu,
    Lie, /* it is a ColumnShell -- but not use that name!! Dangerous for name pollution!*/
    ExprList,

    Star,

    Hanshu,
    SQLFunc, /*a SQLFunction: after checkexpr, a hanshu will become a sqlfunc*/

    Shuzu,
    Yunsuan,
    Qiufan,
    Bijiao,
    Likeop,
    Inop,
    NotIn,
    Between,
    IsNull,
    IsNotNull,

    Cunzai,
    Case,
    Andor,

    Kuohao,

    Zhenjia,
    Distinct,

    Query,

    IfCondition,
    Null,
    IntervalExpression,

    QuestionMark, // for prepared statements
    Default, // for update set a = DEFAULT
    Buffer,
};

typedef schema::ColumnType BiaodashiValueType;

//enum BiaodashiValueType {
//    INT_VALUE = 37000,
//    FLOAT_VALUE,
//    TEXT_VALUE,
//    BOOL_VALUE,
//    DATE_VALUE,
//    UNKNOWN,
//    LIST
//};


enum JoinType {

    InnerJoin = 9000,

    LeftJoin,
    LeftOuterJoin,

    RightJoin,
    RightOuterJoin,

    FullJoin,
    FullOuterJoin,

    SemiJoin,
    AntiJoin

};

enum OrderbyDirection {
    ASC,
    DESC
};


enum SetOperationType {
    UNION,
    UNION_ALL,
    INTERSECT,
    INTERSECT_ALL,
    EXCEPT,
    EXCEPT_ALL
};


enum SQLTreeNodeType {
    Table_NODE = 99900,
    Filter_NODE,
    Column_NODE,
    /*the things between SELECT and FROM*/
    //        MultiJoin_NODE,
    /*unused now*/
            BinaryJoin_NODE,
    Group_NODE,
    Sort_NODE,
    Limit_NODE,
    SetOp_NODE,
    // Insert_NODE,
    // Update_NODE,
    // Delete_NODE,
    SELFJOIN_NODE,
    StarJoin_NODE,
    InnerJoin_NODE,
    Exchange_NODE,
    // Calculation_NODE
};

}//namespace

#endif
