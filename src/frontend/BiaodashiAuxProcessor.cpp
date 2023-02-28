#include "BiaodashiAuxProcessor.h"

namespace aries {

BiaodashiAuxProcessor::BiaodashiAuxProcessor() {
}


std::vector<BiaodashiPointer> BiaodashiAuxProcessor::generate_and_list(BiaodashiPointer arg_expr) {

    //std::cout << "BiaodashiAuxProcessor::generate_and_list: " << arg_expr->ToString() << "\n";

    std::shared_ptr<CommonBiaodashi> cbp = std::dynamic_pointer_cast<CommonBiaodashi>(arg_expr);

    std::vector<BiaodashiPointer> the_ret_and_list;

    if (cbp->GetType() == BiaodashiType::Kuohao) {
        //std::cout << "cbp->GetType() == BiaodashiType::Kuohao\n";
        the_ret_and_list = this->generate_and_list(cbp->GetChildByIndex(0));
    } else if (cbp->GetType() == BiaodashiType::Andor) {
        assert(cbp->GetChildrenCount() == 2);
        std::vector<BiaodashiPointer> left_and_list = this->generate_and_list(cbp->GetChildByIndex(0));

        std::vector<BiaodashiPointer> right_and_list = this->generate_and_list(cbp->GetChildByIndex(1));



        /*for debug*/
        // for (int lali = 0; lali < left_and_list.size(); lali++) {
        //     //std::cout << "left_and_list: " << lali << ":" << left_and_list.size() << "::" << left_and_list[lali]->ToString() << "\n";
        // }

        // for (int rali = 0; rali < right_and_list.size(); rali++) {
        //     //std::cout << "right_and_list: " << rali << ":" << right_and_list.size() << "::" << right_and_list[rali]->ToString() << "\n";
        // }

        /*for debug*/

        if (cbp->GetLogicType4AndOr() == LogicType::AND) {
            the_ret_and_list.insert(
                    the_ret_and_list.end(),
                    left_and_list.begin(),
                    left_and_list.end());

            the_ret_and_list.insert(
                    the_ret_and_list.end(),
                    right_and_list.begin(),
                    right_and_list.end());

        } else {
            /*now we do good things!*/
            /* we need to convert ((A and B and C) or (A and D and E)) to ((A) and ((B and C) or (D and E)))*/

            std::vector<BiaodashiPointer> both_list;
            std::vector<BiaodashiPointer> left_unequal_list;
            std::vector<bool> right_checked_list;
            std::vector<BiaodashiPointer> right_unequal_list;

            /*now we get the final "A and (B or C)"*/
            for (size_t ri = 0; ri < right_and_list.size(); ri++) {
                right_checked_list.push_back(false);
            }

            for (size_t li = 0; li < left_and_list.size(); li++) {
                bool found_matching = false;

                for (size_t ri = 0; ri < right_and_list.size(); ri++) {
                    //std::cout << "left_and_list[li]: " << li << "\t" << left_and_list[li]->ToString() << "\n";
                    //std::cout << "right_and_list[ri]: " << ri << "\t" << right_and_list[ri]->ToString() << "\n";

                    if (CommonBiaodashi::__compareTwoExprs(
                            (CommonBiaodashi *) (left_and_list[li].get()),
                            (CommonBiaodashi *) (right_and_list[ri].get())) == true) {

                        both_list.push_back(left_and_list[li]);
                        right_checked_list[ri] = true;
                        found_matching = true;
                        break;
                    }

                }

                if (found_matching == false) {
                    left_unequal_list.push_back(left_and_list[li]);
                }

            }

            for (size_t ri = 0; ri < right_and_list.size(); ri++) {
                if (right_checked_list[ri] == false) {
                    right_unequal_list.push_back(right_and_list[ri]);
                }
            }

            /*A and (B or C)
             * A is both_list
             * B is left_unequal_list
             * C is right_unequal_list
             */

            /*For A*/
            if (both_list.size() == 0) {
                /*nothing changed! so we just return this*/
                the_ret_and_list.push_back(arg_expr);
            } else {
                /*For A*/
                std::vector<BiaodashiPointer> final_A_list = both_list;

                /*For B*/
                BiaodashiPointer final_B_expr = nullptr;
                if (left_unequal_list.size() == 0) {
                    /* "(a) or (a and c)" = "(a) and (true or (c))" -- in this case, B = true*/
                    final_B_expr = this->make_biaodashi_boolean(true);
                } else {
                    final_B_expr = this->make_biaodashi_from_and_list(left_unequal_list);

                }


                /*For C*/
                BiaodashiPointer final_C_expr = nullptr;
                if (right_unequal_list.size() == 0) {
                    /* "(a) or (a and c)" = "(a) and (true or (c))" -- in this case, B = true*/
                    final_C_expr = this->make_biaodashi_boolean(true);
                } else {
                    final_C_expr = this->make_biaodashi_from_and_list(right_unequal_list);
                }

                std::vector<BiaodashiPointer> a_tmp_expr_array;
                a_tmp_expr_array.push_back(final_B_expr);
                a_tmp_expr_array.push_back(final_C_expr);
                BiaodashiPointer final_B_or_C_expr = this->make_biaodashi_from_or_list(a_tmp_expr_array);

                //the_ret_and_list = the_ret_and_list.concat(final_A_list, [final_B_or_C_expr]);
                the_ret_and_list.insert(
                        the_ret_and_list.end(),
                        final_A_list.begin(),
                        final_A_list.end());

                the_ret_and_list.push_back(final_B_or_C_expr);
            }


        }


    } else {
        /*the default*/
        //std::cout << "we just append: " << arg_expr->ToString() << "\n";
        the_ret_and_list.push_back(arg_expr);
    }


    /*debug: show before return*/

    for (size_t ti = 0; ti < the_ret_and_list.size(); ti++) {
        //std::cout << "the_ret_and_list[" << ti << "]:  " + the_ret_and_list[ti]->ToString() << "\n";
    }

    return the_ret_and_list;
}

void BiaodashiAuxProcessor::generate_or_list( BiaodashiPointer arg_exp, std::vector< BiaodashiPointer >& list  )
{
    auto expr = std::dynamic_pointer_cast< CommonBiaodashi >( arg_exp );
    if ( expr->GetType() == BiaodashiType::Kuohao )
    {
        generate_or_list( expr->GetChildByIndex( 0 ), list );
    }
    else if ( expr->GetType() == BiaodashiType::Andor )
    {
        auto logic_type = static_cast< LogicType >( boost::get< int >( expr->GetContent() ) );
        if ( logic_type == LogicType::OR )
        {
            generate_or_list( expr->GetChildByIndex( 0 ), list );
            generate_or_list( expr->GetChildByIndex( 1 ), list );
            return;
        }
    }

    list.emplace_back( expr );
}


BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_boolean(bool arg_value) {

    std::shared_ptr<CommonBiaodashi> ret = std::make_shared<CommonBiaodashi>(BiaodashiType::Zhenjia, arg_value);

    ret->SetValueType(BiaodashiValueType::BOOL);

    return ret;

}

BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_from_and_list( std::vector< CommonBiaodashiPtr > arg_list) {
    auto ret_expr = arg_list[0];

    for (size_t i = 1; i < arg_list.size(); i++) {
        std::shared_ptr<CommonBiaodashi> tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::Andor,
                                                                                 (int) LogicType::AND);

        tmp->AddChild(ret_expr);
        tmp->AddChild(arg_list[i]);

        tmp->SetValueType(BiaodashiValueType::BOOL);

        ret_expr = tmp;
    }

    return ret_expr;
}

BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_from_and_list(std::vector<BiaodashiPointer> arg_list) {
    BiaodashiPointer ret_expr = arg_list[0];;

    for (size_t i = 1; i < arg_list.size(); i++) {
        std::shared_ptr<CommonBiaodashi> tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::Andor,
                                                                                 (int) LogicType::AND);

        tmp->AddChild(ret_expr);
        tmp->AddChild(arg_list[i]);

        tmp->SetValueType(BiaodashiValueType::BOOL);

        ret_expr = tmp;
    }

    return ret_expr;
}


BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_from_or_list(std::vector<BiaodashiPointer> arg_list) {

    BiaodashiPointer ret_expr = arg_list[0];;

    for (size_t i = 1; i < arg_list.size(); i++) {
        std::shared_ptr<CommonBiaodashi> tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::Andor,
                                                                                 (int) LogicType::OR);

        tmp->AddChild(ret_expr);
        tmp->AddChild(arg_list[i]);

        tmp->SetValueType(BiaodashiValueType::BOOL);

        ret_expr = tmp;
    }

    return ret_expr;

}


BiaodashiPointer
BiaodashiAuxProcessor::make_biaodashi_compare_equal(BiaodashiPointer arg_expr_left, BiaodashiPointer arg_expr_right) {

    std::shared_ptr<CommonBiaodashi> tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::Bijiao,
                                                                             (int) ComparisonType::DengYu);

    tmp->AddChild(arg_expr_left);
    tmp->AddChild(arg_expr_right);

    tmp->SetValueType(BiaodashiValueType::BOOL);

    return tmp;
}


BiaodashiPointer
BiaodashiAuxProcessor::make_biaodashi_compare(BiaodashiPointer arg_expr_left, BiaodashiPointer arg_expr_right,
                                              ComparisonType arg_type) {

    std::shared_ptr<CommonBiaodashi> tmp = std::make_shared<CommonBiaodashi>(BiaodashiType::Bijiao, (int) (arg_type));

    tmp->AddChild(arg_expr_left);
    tmp->AddChild(arg_expr_right);

    tmp->SetValueType(BiaodashiValueType::BOOL);

    return tmp;
}


BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_float(float arg_value) {

    std::shared_ptr<CommonBiaodashi> ret = std::make_shared<CommonBiaodashi>(BiaodashiType::Fudianshu, arg_value);

    ret->SetValueType(BiaodashiValueType::FLOAT);

    return ret;

}


BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_zifuchuan(std::string arg_value) {

    std::shared_ptr<CommonBiaodashi> ret = std::make_shared<CommonBiaodashi>(BiaodashiType::Zifuchuan, arg_value);

    ret->SetValueType(BiaodashiValueType::TEXT);

    return ret;

}


BiaodashiPointer BiaodashiAuxProcessor::make_biaodashi_lie(ColumnShellPointer arg_value) {
    std::shared_ptr<CommonBiaodashi> ret = std::make_shared<CommonBiaodashi>(BiaodashiType::Lie, arg_value);

    ret->SetValueType(arg_value->GetValueType());

    return ret;
}

ColumnShellPointer
BiaodashiAuxProcessor::make_column_shell(BasicRelPointer arg_table, int arg_column_index, int arg_absolute_level) {

    RelationStructurePointer rsp = arg_table->GetRelationStructure();
    std::string table_name = rsp->GetName();

    ColumnStructurePointer csp = rsp->GetColumn(arg_column_index);
    std::string column_name = csp->GetName();

    ColumnShellPointer ret = std::make_shared<ColumnShell>(table_name, column_name);
    ret->SetTable(arg_table);
    ret->SetColumnStructure(csp);
    ret->SetLocationInTable(arg_column_index);
    ret->SetAbsoluteLevel(arg_absolute_level);

    return ret;
}

ColumnShellPointer
BiaodashiAuxProcessor::make_column_shell_only_placeholder(std::string table_name, std::string column_name,
                                                          BiaodashiValueType value_type, int level) {

    ColumnShellPointer ret = std::make_shared<ColumnShell>(table_name, column_name);
    ret->SetTable(nullptr);
    ret->SetColumnStructure(nullptr);
    ret->SetMyOwnValueType(value_type);
    ret->SetAbsoluteLevel(level);

    ret->SetPlaceholderMark(true);

    return ret;
}


BiaodashiPointer BiaodashiAuxProcessor::shallow_copy_biaodashi(BiaodashiPointer arg_expr) {
    if (arg_expr == nullptr) {
        return nullptr;
    }

    CommonBiaodashi *old_expr_p = (CommonBiaodashi *) (arg_expr.get());

    std::shared_ptr<CommonBiaodashi> ret = std::make_shared<CommonBiaodashi>(
            old_expr_p->GetType(),
            old_expr_p->GetContent()
    );


    for (size_t i = 0; i < old_expr_p->GetChildrenCount(); i++) {
        ret->AddChild(old_expr_p->GetChildByIndex(i));
    }

    ret->SetValueType(old_expr_p->GetValueType());
    ret->SetExprContext(old_expr_p->GetExprContext());

    return ret;
}


}
