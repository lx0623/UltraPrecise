/*
 * File: ExpressionSimplifier.h
 * File Created: 2019-08-23
 * Author: Jerry Hu (shenggang.hu@rateup.com.cn)
 */

#pragma once

#include "AriesEngine/AriesCommonExpr.h"
#include "frontend/CommonBiaodashi.h"
#include "datatypes/AriesDate.hxx"
#include "datatypes/AriesDatetime.hxx"
#include "datatypes/AriesTime.hxx"
#include "datatypes/AriesYear.hxx"

using namespace aries_acc;
using namespace aries;

BEGIN_ARIES_ENGINE_NAMESPACE

using SimplifiedResult = boost::variant<int, // 0
                                      double,
                                      aries_acc::Decimal,
                                      aries_acc::AriesDate,
                                      aries_acc::AriesDatetime,
                                      aries_acc::AriesTimestamp, // 5
                                      std::string,
                                      aries_acc::AriesTime,
                                      bool,
                                      aries_acc::AriesNull,
                                      aries_acc::AriesYear, // 10
                                      int64_t,
                                      float>;

class ExpressionSimplifier {
public:
    ExpressionSimplifier( bool simplifyOnlyConst = false )
        : m_simplifyOnlyConst( simplifyOnlyConst )
    {
    }
private:
    bool m_simplifyOnlyConst;
private:
    SimplifiedResult simplifyConstant(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyFunction(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyCalc(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyCompare(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult SimplifyDictEncodedColumnComparison( CommonBiaodashi* origin, THD* thd, bool& success );
    SimplifiedResult simplifyDateFunction(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyIsNull(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyIsNotNull(CommonBiaodashi*, THD* thd, bool& success);
    SimplifiedResult simplifyAndOr(CommonBiaodashi *, THD* thd, bool &success);
    SimplifiedResult simplifyLike(CommonBiaodashi *, THD* thd, bool &success);
    SimplifiedResult doSimplify(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleDateSubOrAdd(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleDate(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleDateDiff(CommonBiaodashi *origin, THD* thd, bool& success);
    SimplifiedResult handleTimeDiff(CommonBiaodashi *origin, THD* thd, bool& success);
    SimplifiedResult handleNow(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleABS(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleUnixTimestamp(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleCast(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleDateFormat(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleCoalesce(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleCase(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleSubstring(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleConcat(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleIfCondition(CommonBiaodashi* origin, THD* thd, bool &success);
    SimplifiedResult handleTruncate(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleMonth(CommonBiaodashi *origin, THD* thd, bool &success);
    SimplifiedResult handleExtract(CommonBiaodashi *origin, THD* thd, bool &success);

    /**!
     * @brief 将常量 between 表达式简化为 bool
     * @param[in] origin 要简化的表达式
     * @param[in] thd 当前线程
     * @param[out] success 简化是否成功
     * @return 如果简化成功，返回简化之后的结果，如果失败，则忽略该返回值
     */
    SimplifiedResult handleBetween(CommonBiaodashi *origin, THD* thd, bool &success);

    /**!
     * @brief 将 not 表达式化简为 bool
     * @param[in] origin 要化简的表达式
     * @param[in] thd 当前线程
     * @param[out] success 化简是否成功
     * @return 如果化简成功，返回化简后的结果，如果失败，则忽略返回值
     */
    SimplifiedResult handleQiufan(CommonBiaodashi *origin, THD* thd, bool &success);

    /**!
     * @brief 将 in 表达式化简为 bool
     * @param[in] origin 要化简的表达式
     * @param[in] thd 当前线程
     * @param[out] success 化简是否成功
     * @return 如果化简成功，返回化简后的结果，如果失败，则忽略返回值
     * @details 候选列表中的 null 可以直接忽略
     */
    SimplifiedResult handleIn( CommonBiaodashi *origin, THD* thd, bool &success );

    /**!
     * @brief 将 not in 表达式化简为 bool
     * @param[in] origin 要化简的表达式
     * @param[in] thd 当前线程
     * @param[out] success 化简是否成功
     * @return 如果化简成功，返回化简后的结果，如果失败，则忽略返回值
     * @details 如果 not in 的候选列表中有 null，则表达式可以直接化简为 false
     */
    SimplifiedResult handleNotIn( CommonBiaodashi *origin, THD* thd, bool &success );

    /**!
     * @brief 将 Exists 表达式化简为 bool
     * @param[in] origin 要化简的表达式
     * @param[in] thd 当前线程
     * @param[out] success 化简是否成功
     * @return 如果化简成功，返回化简后的结果，如果失败，则忽略返回值
     */
    SimplifiedResult handleExists(CommonBiaodashi *origin, THD* thd, bool &success);

    void convertTo(SimplifiedResult& value, BiaodashiValueType type, bool &success);
    AriesCommonExprUPtr resultToAriesCommonExpr(SimplifiedResult& result);
    BiaodashiPointer resultToBiaodashi(SimplifiedResult& result);
    aries_acc::AriesBool conditionValue(const SimplifiedResult& condition, bool &success);

    bool isEqual(SimplifiedResult& left, SimplifiedResult& right);
    void ConvertString(const std::vector<AriesValueType>& expect_types, AriesCommonExpr* constant_child);

public:
    AriesCommonExprUPtr Simplify(CommonBiaodashi* origin, THD* thd);
    BiaodashiPointer SimplifyAsCommonBiaodashi(CommonBiaodashi *origin, THD* thd);
    void ConvertConstantChildrenIfNeed(const AriesCommonExprUPtr& expr);
    bool ConvertExpression(AriesCommonExprUPtr& expr, const AriesColumnType& type);
};

END_ARIES_ENGINE_NAMESPACE
