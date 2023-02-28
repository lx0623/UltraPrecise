#include "AbstractBiaodashi.h"
#include "CommonBiaodashi.h"

namespace aries {
class BiaodashiAuxProcessor {

private:

    BiaodashiAuxProcessor(const BiaodashiAuxProcessor &arg);

    BiaodashiAuxProcessor &operator=(const BiaodashiAuxProcessor &arg);

public:

    BiaodashiAuxProcessor();

    std::vector<BiaodashiPointer> generate_and_list(BiaodashiPointer arg_exp);
    void generate_or_list( BiaodashiPointer arg_exp, std::vector< BiaodashiPointer >& list );

    BiaodashiPointer make_biaodashi_boolean(bool arg_value);

    BiaodashiPointer make_biaodashi_from_and_list(std::vector<BiaodashiPointer> arg_list);
    BiaodashiPointer make_biaodashi_from_and_list( std::vector< CommonBiaodashiPtr > arg_list );

    BiaodashiPointer make_biaodashi_from_or_list(std::vector<BiaodashiPointer> arg_list);


    BiaodashiPointer make_biaodashi_compare_equal(BiaodashiPointer arg_expr_left, BiaodashiPointer arg_expr_right);

    BiaodashiPointer
    make_biaodashi_compare(BiaodashiPointer arg_expr_left, BiaodashiPointer arg_expr_right, ComparisonType arg_type);

    BiaodashiPointer make_biaodashi_float(float arg_value);

    BiaodashiPointer make_biaodashi_zifuchuan(std::string arg_value);

    BiaodashiPointer make_biaodashi_lie(ColumnShellPointer arg_value);

    ColumnShellPointer make_column_shell(BasicRelPointer arg_table, int arg_column_index, int arg_absolute_level);

    ColumnShellPointer
    make_column_shell_only_placeholder(std::string table_name, std::string column_name, BiaodashiValueType value_type,
                                       int level);

    //copy a biaodashi: but not clone! Don't use this unless REALLY necessary!
    BiaodashiPointer shallow_copy_biaodashi(BiaodashiPointer arg_expr);
};

}//namespace
