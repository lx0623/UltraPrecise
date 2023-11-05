/*
 * AriesEngineShell.cpp
 *
 *  Created on: Mar 13, 2019
 *      Author: lichi
 */

#include "AriesEngineShell.h"
#include "AriesExprBridge.h"
#include "datatypes/AriesDatetimeTrans.h"
#include <queue>
#include <vector>

BEGIN_ARIES_ENGINE_NAMESPACE

    static uint32_t __PREC_WORD_ARRAY[] = {0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,23,24,24,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,25,25,26,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,31,31,31,31,31,31,31,31,31,32,32};

    // 二叉树结构
    struct BinaryTree{
        AriesExprType type;
        bool useDictIndex = false; // 对于字典压缩的列，是否使用字典索引数据进行比较计算
        AriesExpressionContent content; // content 中存放了 column 或者 常量的值
        AriesColumnType value_type;
        vector<BinaryTree *> children;
        bool reverse = false;
        int varIndexInKernel = 0;

        BinaryTree( AriesCommonExprUPtr &expr){
            type = expr->GetType();
            useDictIndex = expr->IsUseDictIndex();
            content = expr->GetContent();
            value_type = expr->GetValueType();
        };

        BinaryTree(AriesExprType ObjType,  bool ObjUseDictIndex, AriesExpressionContent ObjContent, AriesColumnType ObjValue_Type, bool ObjReverse, int varIndex){
            type = ObjType;
            useDictIndex = ObjUseDictIndex;
            content = ObjContent;
            value_type = ObjValue_Type;
            reverse = ObjReverse;
            varIndexInKernel = varIndex;
        };

        BinaryTree( AriesCalculatorOpType symbol){
            type = AriesExprType::CALC;
            content =  static_cast< int >( symbol );
        };
    };

    // 多叉树结构
    struct MultipleTree{
        AriesExprType type;
        bool useDictIndex = false; // 对于字典压缩的列，是否使用字典索引数据进行比较计算
        AriesExpressionContent content; // content 中存放了 column 或者 常量的值
        AriesColumnType value_type;
        vector<MultipleTree *> childList;
        bool reverse = false;
        struct BinaryTree* rootObj = NULL;
        int varIndexInKernel = 0;

        MultipleTree(){

        }

        MultipleTree( struct BinaryTree* Obj){
            rootObj = Obj;
            type = Obj->type;
            useDictIndex = Obj->useDictIndex;
            content = Obj->content;
            value_type = Obj->value_type;
            reverse = Obj->reverse;
        };

        MultipleTree( AriesCalculatorOpType symbol){
            type = AriesExprType::CALC;
            content =  static_cast< int >( symbol );
        };

        void CopyNodeInfo( struct MultipleTree* Obj){
            // rootObj = Obj->rootObj;
            type = Obj->type;
            useDictIndex = Obj->useDictIndex;
            content = Obj->content;
            value_type = Obj->value_type;
            reverse = Obj->reverse;
            childList = Obj->childList;
        }
    };

    struct BinaryTree* ListToBinTree(vector<struct BinaryTree*> BinNodelist, int index, AriesCalculatorOpType op, int curIndex){
        if( index == 0){
            // 第一个数如果就是负数应该怎么处理 那么将 保留 reverse == true 到 表达式生成阶段单独运算
            struct BinaryTree *temp = BinNodelist[index];
            return temp;
        }
        else{
            vector<struct BinaryTree*> tempChildren;
            tempChildren.push_back(ListToBinTree(BinNodelist, index-1, op, curIndex));
            tempChildren.push_back(BinNodelist[index]);
            // 如果 当前值是负数的话 那么需要进行取反
            // 由于结构改变 最终结果的精度也可能改变 例如 1+l_tax+l_discount*(0/l_extendedprice)+1 结果精度从 21,6 变为 13,2
            // 另一方面 在生成代码的时候需要各个节点的结果类型
            struct BinaryTree *temp;
            if( op == AriesCalculatorOpType::ADD){
                if( BinNodelist[index]->reverse == true){
                    BinNodelist[index]->reverse = false;
                    temp = new BinaryTree(AriesCalculatorOpType::SUB);
                }
                else{
                    temp = new BinaryTree(AriesCalculatorOpType::ADD);
                }
                temp->value_type.DataType.Precision = max(tempChildren[0]->value_type.DataType.Precision - tempChildren[0]->value_type.DataType.Scale, tempChildren[1]->value_type.DataType.Precision - tempChildren[1]->value_type.DataType.Scale)+1;
                temp->value_type.DataType.Scale = max(tempChildren[0]->value_type.DataType.Scale, tempChildren[1]->value_type.DataType.Scale);
                temp->value_type.DataType.Precision += temp->value_type.DataType.Scale;
                temp->value_type.DataType.AdaptiveLen = __PREC_WORD_ARRAY[temp->value_type.DataType.Precision];
            }
            if( op == AriesCalculatorOpType::MUL){
                temp = new BinaryTree(AriesCalculatorOpType::MUL);
                temp->value_type.DataType.Precision = (tempChildren[0]->value_type.DataType.Precision - tempChildren[0]->value_type.DataType.Scale) + (tempChildren[1]->value_type.DataType.Precision - tempChildren[1]->value_type.DataType.Scale);
                temp->value_type.DataType.Scale = tempChildren[0]->value_type.DataType.Scale + tempChildren[1]->value_type.DataType.Scale;
                temp->value_type.DataType.Precision += temp->value_type.DataType.Scale;
                temp->value_type.DataType.AdaptiveLen = __PREC_WORD_ARRAY[temp->value_type.DataType.Precision];
            }
            if( op == AriesCalculatorOpType::DIV){
                temp = new BinaryTree(AriesCalculatorOpType::DIV);
                uint32_t tmpAdaptiveLen = max(__PREC_WORD_ARRAY[tempChildren[0]->value_type.DataType.Precision + DIV_FIX_EX_FRAC], __PREC_WORD_ARRAY[tempChildren[1]->value_type.DataType.Precision]);
                temp->value_type.DataType.Precision = tempChildren[0]->value_type.DataType.Precision - tempChildren[1]->value_type.DataType.Precision + tempChildren[1]->value_type.DataType.Scale  + DIV_FIX_EX_FRAC + 1;
                temp->value_type.DataType.Scale = tempChildren[0]->value_type.DataType.Scale + DIV_FIX_EX_FRAC;
                temp->value_type.DataType.AdaptiveLen = max(__PREC_WORD_ARRAY[temp->value_type.DataType.Precision], tmpAdaptiveLen);
            }
            if( op == AriesCalculatorOpType::MOD){
                temp = new BinaryTree(AriesCalculatorOpType::MOD);
                temp->value_type.DataType.Precision = tempChildren[1]->value_type.DataType.Precision;
                temp->value_type.DataType.Scale = 0;
                temp->value_type.DataType.AdaptiveLen = max(tempChildren[0]->value_type.DataType.AdaptiveLen, tempChildren[1]->value_type.DataType.AdaptiveLen);
            }
            if(temp->value_type.DataType.AdaptiveLen > NUM_TOTAL_DIG){
                temp->value_type.DataType.AdaptiveLen = NUM_TOTAL_DIG;
            }
            temp->varIndexInKernel = curIndex;
            temp->children = tempChildren;
            return temp;
        }
    }

    struct BinaryTree* MultiToBinTree(struct MultipleTree* multipleTree, int varIndex){
        vector<struct BinaryTree*> temp;
        int curIndex = varIndex++;
        for (size_t i = 0; i < multipleTree->childList.size(); i++){
            if( multipleTree->childList[i]->type != AriesExprType::CALC){
                BinaryTree * tempBinaryTree =  new BinaryTree( multipleTree->childList[i]->type, multipleTree->childList[i]->useDictIndex, multipleTree->childList[i]->content, multipleTree->childList[i]->value_type, multipleTree->childList[i]->reverse, varIndex++);
                tempBinaryTree->value_type.DataType.AdaptiveLen = __PREC_WORD_ARRAY[tempBinaryTree->value_type.DataType.Precision];
                temp.push_back(tempBinaryTree);
            }
            else{
                BinaryTree * tempBinaryTree = MultiToBinTree(multipleTree->childList[i], varIndex*10);
                varIndex++;
                temp.push_back(tempBinaryTree);
            }
        }
        struct BinaryTree* tempNode;
        if( multipleTree->childList.size() == 0){ // 1 + col1 + 2 - 3 这种最终根节点变为 col1
            tempNode = new BinaryTree( multipleTree->type, multipleTree->useDictIndex, multipleTree->content, multipleTree->value_type, multipleTree->reverse, varIndex++);
            tempNode->varIndexInKernel = curIndex;
        }
        else if( multipleTree->type == AriesExprType::CALC && multipleTree->childList.size() == 1){
            if(multipleTree->childList[0]->type == AriesExprType::CALC ){
                tempNode = ListToBinTree(temp, temp.size()-1, static_cast< AriesCalculatorOpType >(boost::get< int >( multipleTree->content )), curIndex);
                tempNode->varIndexInKernel = curIndex;
            }
            else{
                // 对于 1 + 2 的数变为 + 3 时 这种最终根节点会变为 3
                tempNode = new BinaryTree(temp[0]->type, temp[0]->useDictIndex, temp[0]->content, temp[0]->value_type, temp[0]->reverse, varIndex++);
                tempNode->varIndexInKernel = curIndex;
            }
        }
        else{
            tempNode = ListToBinTree(temp, temp.size()-1, static_cast< AriesCalculatorOpType >(boost::get< int >( multipleTree->content )), curIndex);
            tempNode->varIndexInKernel = curIndex;
        } 
        return tempNode;
    }

    struct MultipleTree* RemoveZeroNode(struct MultipleTree* multipleTree){
        vector<MultipleTree *> newChildList;
         for (size_t i = 0; i < multipleTree->childList.size(); i++){
            // 因为 前面已经处理过了 那么它一定是零
            if( multipleTree->childList[i]->type == AriesExprType::INTEGER ){
                // 如果此符号是乘法的话 那么这个节点为零
                if( static_cast< AriesCalculatorOpType >( boost::get< int >( multipleTree->content ) )== AriesCalculatorOpType::MUL){
                    // 这个节点的值为零 结束
                    multipleTree->type = AriesExprType::INTEGER;
                    multipleTree->content = 0;
                    return multipleTree;
                }
            }
            else if( multipleTree->childList[i]->type == AriesExprType::CALC ){
                multipleTree->childList[i] = RemoveZeroNode(multipleTree->childList[i]);
                if( multipleTree->childList[i]->type == AriesExprType::INTEGER ){
                    // 如果此符号是乘法的话 那么这个节点为零
                    if( static_cast< AriesCalculatorOpType >( boost::get< int >( multipleTree->content )) == AriesCalculatorOpType::MUL){
                        // 这个节点的值为零 结束
                        multipleTree->type = AriesExprType::INTEGER;
                        multipleTree->content = 0;
                        return multipleTree;
                    }
                }
                else{
                    newChildList.push_back(multipleTree->childList[i]);
                }
            }
            else{
                newChildList.push_back(multipleTree->childList[i]);
            }
        }
        multipleTree->childList = newChildList;
        return multipleTree;
    }

    // 将 AriesCommonExprUPtr 的树克隆到 struct BinaryTree 上
    struct BinaryTree *CloneDataFromExpr(AriesCommonExprUPtr &expr)
    {
        BinaryTree * result = new  BinaryTree(expr);
        for( int i=0; i<expr->GetChildrenCount(); i++){
            result->children.push_back( CloneDataFromExpr(expr->children[i]) );
        }
        return result;
    }

    // 将 struct BinaryTree 的树克隆到 AriesCommonExprUPtr 上
    AriesCommonExprUPtr CloneDataFromBinaryTree(struct BinaryTree* binTree, int ansDecimalLen)
    {
        binTree->value_type.DataType.AdaptiveLen = ansDecimalLen;
        AriesCommonExprUPtr result = make_unique < AriesCommonExpr > ( binTree->type, binTree->content, binTree->value_type );
        result->SetUseDictIndex(binTree->useDictIndex);
        result->value_reverse = binTree->reverse;
        result->varIndexInKernel = binTree->varIndexInKernel;
        for( auto child : binTree->children )
            result->AddChild( CloneDataFromBinaryTree(child, ansDecimalLen) );
        return result;
    }

    // 由于加法具有交换律和结合律 因此修改减法节点为加法节点
    void ModifySubOperation( struct BinaryTree* binTree){
        queue<BinaryTree *> queueList;
        queueList.push(binTree);
        while(!queueList.empty()){
            struct BinaryTree* node = queueList.front();
            if(node->type == AriesExprType::CALC){
                if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content ) ) == AriesCalculatorOpType::MUL && node->reverse == true){
                    // 当乘法节点如果需要取反时 -(a*b) = (-a) * b 或者 -(a*b) = a * (-b) 优先级是 常量优先 CALC 最后是 Column
                    node->children[0]->reverse = !node->children[0]->reverse;
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content ) ) == AriesCalculatorOpType::DIV && node->reverse == true){
                    // 当除法节点如果需要取反时 -(a/b) = -(a) / b
                    node->children[0]->reverse = !node->children[0]->reverse;
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content ) ) == AriesCalculatorOpType::ADD && node->reverse == true){
                    // 当加法节点如果需要取反时 -(a+b) = (-a) + (-b)
                    node->children[0]->reverse = !node->children[0]->reverse;
                    node->children[1]->reverse = !node->children[1]->reverse;
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content ) ) == AriesCalculatorOpType::SUB){
                    // 当减法节点不需要取反时 (a-b) = a + (-b)
                    if( node->reverse == false)
                        node->children[1]->reverse = !node->children[1]->reverse;
                    // 当减法节点如果需要取反时 -(a-b) = (-a) + b
                    if( node->reverse == true)
                        node->children[0]->reverse = !node->children[0]->reverse;
                    // 减法节点变为加法节点
                    node->content = static_cast< int >( AriesCalculatorOpType::ADD );
                }
                if( node->children.size() == 2){
                    queueList.push(node->children[0]);
                    queueList.push(node->children[1]);
                }
            }
            queueList.pop();
        }
    }

    // 获取 以 binaryTree 为根节点 其父节点符号为 symbol 情况下，可以直接与 其父节点 相连的所有孩子节点
    // 此处 如果 symbol 是 除法符号的话 那么其子节点不能直接连上去
    vector<struct MultipleTree* > getChild( struct BinaryTree* binaryTree, AriesCalculatorOpType symbol){
        vector<struct MultipleTree*> childList;
        if ( symbol == AriesCalculatorOpType::DIV){
            childList.push_back(new MultipleTree(binaryTree->children[0]));
            childList.push_back(new MultipleTree(binaryTree->children[1]));
            return childList;
        }
        
        // 检查其左子树
        // 左子树 若为 非操作符 如果 该列已经取反 则需要生成一个(-)节点，然后该节点与(-)节点相连，否则直接与根节点的父节点相连
        // 左子树 若为 操作符且操作符与父节点一致 说明左子树节点可直接与根节点的父节点相连
        // 左子树 若为 操作符但操作符与父节点不一致 则其直接与根节点的父节点相连
        if( binaryTree->children[0]->type != AriesExprType::CALC){
            childList.push_back(new MultipleTree(binaryTree->children[0]));
        }
        else if( static_cast< AriesCalculatorOpType >( boost::get< int >( binaryTree->children[0]->content ) ) == symbol){
            // 说明子节点与父节点保持一致 说明左子树节点可直接与根节点的父节点相连
            vector<struct MultipleTree*> childListNext = getChild(binaryTree->children[0], symbol);
            childList.insert(childList.end(), childListNext.begin(), childListNext.end());
        }
        else{
            // 操作符但操作符与父节点不一致 则其直接与根节点的父节点相连
            childList.push_back(new MultipleTree(binaryTree->children[0]));
        }

        // 检查右子树 同左子树一样的操作
        if( binaryTree->children[1]->type != AriesExprType::CALC){
                childList.push_back(new MultipleTree(binaryTree->children[1]));
        }
        else if(static_cast< AriesCalculatorOpType >( boost::get< int >( binaryTree->children[1]->content ) ) == symbol ){
            // 说明子节点与父节点保持一致 说明左子树节点可直接与根节点的父节点相连
            vector<struct MultipleTree*> childListNext = getChild(binaryTree->children[1], symbol);
            childList.insert(childList.end(), childListNext.begin(), childListNext.end());
        }
        else{
            // 操作符但操作符与父节点不一致 则其直接与根节点的父节点相连
            childList.push_back(new MultipleTree(binaryTree->children[1]));
        }
        return childList;
    }

    // 将二叉树转换为多叉树
    void BinToMultiTree( struct BinaryTree* binaryTree, struct MultipleTree* multipleTree){
        // 多叉树的子节点列表
        vector<struct MultipleTree*> childList;
        // 获取 以 binaryTree 为根节点 其父节点符号为 symbol 情况下，可以直接与 其父节点 相连的所有孩子节点
        // 这样 可以直接与其父节点相连的所有孩子节点都是 多叉树 的 孩子节点
        childList = getChild(binaryTree, static_cast< AriesCalculatorOpType >( boost::get< int >( multipleTree->rootObj->content ) ));
        // 将其加入到多叉树子节点序列中
        multipleTree->childList = childList;
        // 递归这个多叉树的子节点 进一步构建多叉树
        for( size_t i=0 ; i<childList.size() ; i++){
            if( childList[i]->rootObj != NULL && childList[i]->rootObj->type == AriesExprType::CALC ){
                if( childList[i]->rootObj->children[0] != NULL && childList[i]->rootObj->children[1] != NULL){
                    BinToMultiTree(childList[i]->rootObj, childList[i]);
                }
            }
        }
    }

    bool CmpByScale(struct MultipleTree* a, struct MultipleTree* b){
        // 将 INT 常量放在最前面 Decimal 常量次之 最后 按照 frac 从小到大 排放 Column 或 CALC操作符
        if(a->type == AriesExprType::INTEGER)
            return true;
        if(b->type == AriesExprType::INTEGER)
            return false;
        if(a->type == AriesExprType::DECIMAL && b->type == AriesExprType::DECIMAL)
            return a->value_type.DataType.Scale < b->value_type.DataType.Scale;
        if(a->type == AriesExprType::DECIMAL)
            return true;
        if(b->type == AriesExprType::DECIMAL)
            return false;
        return a->value_type.DataType.Scale < b->value_type.DataType.Scale;
    }

    // 因为加法具有交换律 因此 当节点为 + 号时 按照 frac 从小到大 可以减少对齐的消耗
    void AlignOptMultiTree(struct MultipleTree* multiTree){
        queue<MultipleTree *> queueList;
        queueList.push(multiTree);
        while(!queueList.empty()){
            struct MultipleTree* node = queueList.front();
            if( queueList.front()->type == AriesExprType::CALC && queueList.front()->value_type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL){
                if(  static_cast< AriesCalculatorOpType >( boost::get< int >( node->content )) ==  AriesCalculatorOpType::ADD){
                    // 将节点从小到大 按照 
                    sort(node->childList.begin(), node->childList.end(), CmpByScale);
                    // 将 整数 和 Decimal 进行计算
                    int nodeInt = 0;
                    aries_acc::Decimal nodeDecimal(0);
                    size_t index = 0; 
                    for (index = 0; index < node->childList.size(); index++){
                        if(node->childList[index]->type == AriesExprType::INTEGER){
                            int t = boost::get<int>(node->childList[index]->content);
                            if(node->childList[index]->reverse == true){
                                t = -t;
                            }
                            nodeInt += t;
                        }
                        else if(node->childList[index]->type == AriesExprType::DECIMAL){
                            aries_acc::Decimal decimal = boost::get<aries_acc::Decimal>(node->childList[index]->content);
                            if(node->childList[index]->reverse == true){
                                decimal.sign = 1;
                            }
                            nodeDecimal.AddDecimalOnCpu( decimal );
                        }
                        else{
                            break;
                        }
                    }
                    if(index == 0 ){
                        // 说明没有常量
                        for(size_t i=0 ; i<node->childList.size() ; i++){
                            if( node->childList[i]->type == AriesExprType::CALC ){
                                queueList.push(node->childList[i]);
                            }
                        }
                    }
                    vector<MultipleTree *> newChildList;
                    // 如果 nodeInt 和 nodeDecimal 都等于零 那么分为两种情况
                    if(nodeInt == 0 && nodeDecimal.isZero() == true){
                        // 情况1 : 加法符号中没有 int 常量 和 decimal 常量 情况1 已经包含在上面的情况中
                        // 情况2 : 加法符号中有 int 常量 和 decimal 常量 
                        // 情况2 中要排查一下 当时他们相加后得 0  此时剩下数组中节点的数量 如果只剩下一个 column 结点的话 那么需要 用这个节点代替 + 节点
                        if( index == node->childList.size()-1 ){
                            // 该节点是最后一个节点 那么这个节点就要替代这个根
                            node->CopyNodeInfo( node->childList[index] );
                        }
                        else{
                            for (  ; index < node->childList.size() ; index++){
                                newChildList.push_back( node->childList[index] );
                                if(node->childList[index]->type == AriesExprType::CALC){
                                    queueList.push( node->childList[index] );
                                }
                            }
                            node->childList = newChildList;
                        }
                    }
                    else{
                        // 此时分成了三种情况  nodeInt != 0  || nodeDecimal != 0 || nodeInt 和 nodeDecimal 都不为 0
                        // 我们要做的是把它们相加起来 根据 frac 把他放到合适的位置上
                        if( nodeDecimal.isZero() == true){
                            struct MultipleTree *newNode = new MultipleTree();
                            // 直接将 INTEGER 的常量 构造成 frac = scale 的 Decimal
                            int nearScale = node->childList[index]->value_type.DataType.Scale;
                            newNode->type = AriesExprType::DECIMAL;
                            newNode->value_type.DataType.Scale = nearScale;
                            aries_acc::Decimal decimal(nodeInt, nearScale, false, false, false);
                            newNode->content = decimal;
                            newChildList.push_back( newNode );
                            for (  ; index < node->childList.size() ; index++){
                                newChildList.push_back( node->childList[index] );
                                if(node->childList[index]->type == AriesExprType::CALC){
                                    queueList.push( node->childList[index] );
                                }
                            }
                        }
                        else{
                            bool optInsertFlag = true;
                            if(nodeInt !=0 ){
                                aries_acc::Decimal tempDecimal(nodeInt);
                                nodeDecimal.AddDecimalOnCpu( tempDecimal );
                            }
                            for (  ; index < node->childList.size() ; index++){
                                if(node->childList[index]->value_type.DataType.Scale > nodeDecimal.frac && optInsertFlag == true){
                                    optInsertFlag = false;
                                    aries_acc::Decimal decimal(nodeDecimal, node->childList[index]->value_type.DataType.Scale , false, false, false, false);
                                    struct MultipleTree *newNode = new MultipleTree();
                                    // 直接将 INTEGER 的常量 构造成 frac = scale 的 Decimal
                                    newNode->type = AriesExprType::DECIMAL;
                                    newNode->value_type.DataType.Scale = decimal.frac;
                                    newNode->content = decimal;
                                    newChildList.push_back( newNode );
                                }
                                newChildList.push_back( node->childList[index] );
                                if(node->childList[index]->type == AriesExprType::CALC){
                                    queueList.push( node->childList[index] );
                                }
                            }
                            // 这个常量的frac非常大 没有 column 比它大 那么它需要放在最后
                            if(optInsertFlag == true){
                                struct MultipleTree *newNode = new MultipleTree();
                                newNode->type = AriesExprType::DECIMAL;
                                newNode->value_type.DataType.Precision = nodeDecimal.frac;
                                newNode->value_type.DataType.Scale = nodeDecimal.frac;
                                newNode->content = nodeDecimal;
                                newChildList.push_back( newNode );
                            }
                        }
                        node->childList = newChildList;
                    }
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content )) ==  AriesCalculatorOpType::MUL){
                    int nodeInt = 1;
                    aries_acc::Decimal nodeDecimal(1);
                    vector<MultipleTree *> newChildList;
                    for(size_t i=0 ; i<node->childList.size(); i++){
                        if(node->childList[i]->type == AriesExprType::INTEGER){
                            nodeInt *= boost::get<int>(node->childList[i]->content);
                        }
                        else if(node->childList[i]->type == AriesExprType::DECIMAL){
                            nodeDecimal.MulDecimalOnCpu(boost::get<aries_acc::Decimal>(node->childList[i]->content));
                        }
                        else{
                            newChildList.push_back(node->childList[i]);
                        }
                    }
                    if( nodeInt !=1 || nodeDecimal != 1){
                        if(nodeInt == 0 || nodeDecimal.isZero()){
                            // 说明这个节点会变成零折叠起来
                            node->type = AriesExprType::INTEGER;
                            node->content = 0;
                        }
                        else{
                            aries_acc::Decimal tempDecimal(nodeInt);
                            nodeDecimal.MulDecimalOnCpu( tempDecimal );
                            if(nodeDecimal != 1){
                                struct MultipleTree *newNode = new MultipleTree();
                                newNode->type = AriesExprType::DECIMAL;
                                newNode->value_type.DataType.Scale = nodeDecimal.frac;
                                newNode->content = nodeDecimal;
                                newChildList.push_back( newNode );
                            }
                            node->childList = newChildList;
                        }
                    }
                    for( size_t i=0 ; i<node->childList.size() ; i++){
                        if(node->childList[i]->type == AriesExprType::CALC){
                           queueList.push(node->childList[i]);
                        }
                    }
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content )) ==  AriesCalculatorOpType::DIV){
                    // 除法只有可能左子树为零
                    if(node->childList[0]->type == AriesExprType::INTEGER && boost::get<int>(node->childList[0]->content) == 0){
                        // 说明这个节点会变成零在后面折叠起来
                        node->type = AriesExprType::INTEGER;
                        node->content = 0;
                    }
                    else{
                        for( size_t i=0 ; i<node->childList.size() ; i++){
                            if(node->childList[i]->type == AriesExprType::CALC){
                                queueList.push(node->childList[i]);
                            }
                            if(node->childList[i]->type == AriesExprType::INTEGER){
                                aries_acc::Decimal decimal;
                                switch(node->childList[i]->value_type.DataType.ValueType){
                                    case AriesValueType::INT8: decimal = boost::get<int8_t>(node->childList[i]->content); break;
                                    case AriesValueType::INT16: decimal = boost::get<int16_t>(node->childList[i]->content);  break;
                                    case AriesValueType::INT32: decimal = boost::get<int32_t>(node->childList[i]->content);  break;
                                    case AriesValueType::INT64: decimal = boost::get<int64_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT8: decimal = boost::get<uint8_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT16: decimal = boost::get<uint16_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT32: decimal = boost::get<uint32_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT64: decimal = boost::get<uint64_t>(node->childList[i]->content);  break;
                                    default: ;
                                }

                                struct MultipleTree *newNode = new MultipleTree();
                                newNode->type = AriesExprType::DECIMAL;
                                newNode->value_type.DataType.Precision = decimal.prec;
                                newNode->value_type.DataType.Scale = decimal.frac;
                                newNode->content = decimal;
                                node->childList[i] = newNode;
                            }
                        }
                    }
                }
                else if( static_cast< AriesCalculatorOpType >( boost::get< int >( node->content )) ==  AriesCalculatorOpType::MOD){
                    // 求余只有可能左子树为零
                    if(node->childList[0]->type == AriesExprType::INTEGER && boost::get<int>(node->childList[0]->content) == 0){
                        // 说明这个节点会变成零在后面折叠起来
                        node->type = AriesExprType::INTEGER;
                        node->content = 0;
                    }
                    else{
                        for( size_t i=0 ; i<node->childList.size() ; i++){
                            if(node->childList[i]->type == AriesExprType::CALC){
                                queueList.push(node->childList[i]);
                            }
                            if(node->childList[i]->type == AriesExprType::INTEGER){
                                aries_acc::Decimal decimal;
                                switch(node->childList[i]->value_type.DataType.ValueType){
                                    case AriesValueType::INT8: decimal = boost::get<int8_t>(node->childList[i]->content); break;
                                    case AriesValueType::INT16: decimal = boost::get<int16_t>(node->childList[i]->content);  break;
                                    case AriesValueType::INT32: decimal = boost::get<int32_t>(node->childList[i]->content);  break;
                                    case AriesValueType::INT64: decimal = boost::get<int64_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT8: decimal = boost::get<uint8_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT16: decimal = boost::get<uint16_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT32: decimal = boost::get<uint32_t>(node->childList[i]->content);  break;
                                    case AriesValueType::UINT64: decimal = boost::get<uint64_t>(node->childList[i]->content);  break;
                                    default: ;
                                }

                                struct MultipleTree *newNode = new MultipleTree();
                                newNode->type = AriesExprType::DECIMAL;
                                newNode->value_type.DataType.Precision = decimal.prec;
                                newNode->value_type.DataType.Scale = decimal.frac;
                                newNode->content = decimal;
                                node->childList[i] = newNode;
                            }
                        }
                    }
                }
                else{
                    for( size_t i=0 ; i<node->childList.size() ; i++){
                        if(node->childList[i]->type == AriesExprType::CALC){
                            queueList.push(node->childList[i]);
                        }
                    }
                }
            }
            queueList.pop();
        }
    }

    // 层序遍历
    void UpdateVarIndexInKernel(struct BinaryTree* binaryTree, int varIndex){
        int curIndex = varIndex;
        queue<struct BinaryTree *> que;
        que.push(binaryTree);
        while(!que.empty()){
            struct BinaryTree *tmpNode = que.front();
            tmpNode->varIndexInKernel = curIndex;
            curIndex++;
            if(tmpNode->type == AriesExprType::CALC){
                que.push(tmpNode->children[0]);
                que.push(tmpNode->children[1]);
            }
            que.pop();
        }
    }

    void AlignOptimizeNode(  struct BinaryTree* binTree )
    {
        // 由于加法具有交换律而减法没有交换律，这里需要修改树的结构
        // 此函数把修改二叉树的结构 将所有的 减法 符号运算 a-b 转化为 a+(-b)
        ModifySubOperation(binTree);
        // 构建多叉树 二叉树的根节点也是多叉树的根节点
        struct MultipleTree* multiTree = new MultipleTree(binTree);
        BinToMultiTree(binTree, multiTree);
        // 更新各个节点的 frac 此处应该不需要 更新各个节点的 frac 因为是二叉树变多叉树是 所有可连接父节点的二叉树节点直接连接到父节点上 因此 原本 二叉树节点的结果类型就是多叉树节点的结果类型
        // UpdateMultiTreeNodeFrac(multiTree);
        // 对齐优化
        AlignOptMultiTree(multiTree);
        // 递归消除零
        multiTree = RemoveZeroNode(multiTree);
        // 调用多叉树构建二叉树
        struct BinaryTree* binaryTree = MultiToBinTree(multiTree, 0);
        // 更新 varIndexInKernel
        UpdateVarIndexInKernel(binaryTree, 0);
        // 多叉树转成二叉树 根节点不变
        binTree->varIndexInKernel = binaryTree->varIndexInKernel;
        binTree->type = binaryTree->type;
        binTree->content = binaryTree->content;
        binTree->value_type = binaryTree->value_type;
        if(binTree->value_type.DataType.Precision > SUPPORTED_MAX_PRECISION){
            binTree->value_type.DataType.Precision = SUPPORTED_MAX_PRECISION;
        }
        binTree->useDictIndex = binaryTree->useDictIndex;
        binTree->children = binaryTree->children;
        binTree->reverse = binaryTree->reverse;
    }

    void AlignOptimize( AriesCommonExprUPtr &expr )
    {
        queue<BinaryTree *> nodeQueue;
        vector<BinaryTree *> calcNodeVector;
        // 这个 unique_ptr 只能 移动 不能复制 两个指针不能指向同一 对象
        // 所以此处 将树全部复制一遍
        struct BinaryTree *pStack = CloneDataFromExpr(expr);
        // 利用层序遍历找到所有计算节点的根节点 即以 CALC 为根节点的子树
        nodeQueue.push(pStack);
        // 对于根节点非 CALC 的要记录一下计算过程中使用的最大 Decimal 的长度
        int ansDecimalLen = 0;
        while(nodeQueue.empty() == false){
            // 当节点是 CALC 说明它是一棵计算树
            if ( nodeQueue.front()->type == AriesExprType::CALC && nodeQueue.front()->value_type.DataType.ValueType == AriesValueType::COMPACT_DECIMAL){
                AriesColumnType tmp_value_type = nodeQueue.front()->value_type;
                // 找到了 CALC 计算节点 那么我们直接把这个节点进行改造
                AlignOptimizeNode(nodeQueue.front());
                // 结果精度可能变了 这里要重新算一下 compactdecimal 的长度
                int compactLen = GetDecimalRealBytes( nodeQueue.front()->value_type.DataType.Precision, nodeQueue.front()->value_type.DataType.Scale);
                int calcLen = nodeQueue.front()->value_type.DataType.AdaptiveLen;
                nodeQueue.front()->value_type.DataType.Length = compactLen;
                nodeQueue.front()->value_type.DataType.AdaptiveLen = __PREC_WORD_ARRAY[compactLen]  > calcLen? __PREC_WORD_ARRAY[compactLen] : calcLen;
                nodeQueue.front()->value_type.DataType.ValueType = tmp_value_type.DataType.ValueType;
                nodeQueue.front()->value_type.HasNull = tmp_value_type.HasNull;
                nodeQueue.front()->value_type.IsUnique = tmp_value_type.IsUnique;
            }
            // 如果不是 那么需要查看它子节点中是否有 计算树
            else{
                for( auto child : nodeQueue.front()->children ){
                    nodeQueue.push(child);
                }
            }
            if( __PREC_WORD_ARRAY[nodeQueue.front()->value_type.DataType.Length] > ansDecimalLen ){
                ansDecimalLen = __PREC_WORD_ARRAY[nodeQueue.front()->value_type.DataType.Length];
            }
            nodeQueue.pop();
        }
        // 将 pStack 里面的数据 拷贝到 expr 中
        expr = CloneDataFromBinaryTree(pStack, ansDecimalLen);
    }

    AriesEngineShell::AriesEngineShell()
    {
        // TODO Auto-generated constructor stub
    }

    AriesEngineShell::~AriesEngineShell()
    {
        // TODO Auto-generated destructor stub
    }

    // AriesScanNodeSPtr AriesEngineShell::MakeScanNode( int nodeId, const string& dbName, PhysicalTablePointer arg_table, const vector< int >& arg_columns_id )
    // {
    //     return AriesNodeManager::MakeScanNode( nodeId, dbName, arg_table, arg_columns_id );
    // }

    AriesMvccScanNodeSPtr AriesEngineShell::MakeMvccScanNode( int nodeId, const AriesTransactionPtr& tx, const string& dbName, const string& tableName,
            const vector< int >& arg_columns_id )
    {
        return AriesNodeManager::MakeMvccScanNode( nodeId, tx, dbName, tableName, arg_columns_id );
    }

    static void ConvertConstConditionExpr( AriesCommonExprUPtr& expr )
    {
        AriesExprType exprType = expr->GetType();
        AriesColumnType valueType = expr->GetValueType();

        AriesExprType resultExprType = AriesExprType::TRUE_FALSE;
        AriesValueType _type = AriesValueType::BOOL;
        aries::AriesDataType data_type{_type, 1};
        AriesColumnType resultValueType{ data_type, false, false };

        auto exprId = expr->GetId();
        switch ( exprType )
        {
            case AriesExprType::INTEGER:
            {
                switch ( valueType.DataType.ValueType )
                {
                    case AriesValueType::INT32:
                    {
                        auto content = boost::get<int>( expr->GetContent() );
                        expr = AriesCommonExpr::Create( resultExprType, 0 == content ? false : true, resultValueType );
                        break;
                    }
                    case AriesValueType::INT64:
                    {
                        auto content = boost::get<int64_t>( expr->GetContent() );
                        expr = AriesCommonExpr::Create( resultExprType, 0 == content ? false : true, resultValueType );
                        break;
                    }

                    default:
                        ARIES_ASSERT( 0, "unexpected value type: " + std::to_string( (int) valueType.DataType.ValueType ) );
                        break;
                }
                break;
            }

            case AriesExprType::FLOATING:
            {
                auto content = boost::get<double>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, 0 == content ? false : true, resultValueType );
                break;
            }

            case AriesExprType::DECIMAL:
            {
                auto content = boost::get<aries_acc::Decimal>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, 0 == content ? false : true, resultValueType );
                break;
            }

            case AriesExprType::STRING:
            {
                auto content = boost::get<std::string>( expr->GetContent() );
                int32_t i = 1;
                try
                {
                    i = std::stoi( content );
                }
                catch( std::invalid_argument &e )
                {
                    i = 0;
                }
                catch( ... )
                {
                    // ignore any other exceptions
                }
                
                expr = AriesCommonExpr::Create( resultExprType, 0 == i ? false : true, resultValueType );
                break;
            }

            case AriesExprType::DATE:
            {
                auto content = boost::get<aries_acc::AriesDate>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, AriesDatetimeTrans::GetInstance().ToBool( content ), resultValueType );
                break;
            }
            case AriesExprType::DATE_TIME:
            {
                auto content = boost::get<aries_acc::AriesDatetime>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, AriesDatetimeTrans::GetInstance().ToBool( content ), resultValueType );
                break;
            }
            case AriesExprType::TIME:
            {
                auto content = boost::get<aries_acc::AriesTime>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, AriesDatetimeTrans::GetInstance().ToBool( content ), resultValueType );
                break;
            }
            case AriesExprType::TIMESTAMP:
            {
                auto content = boost::get<aries_acc::AriesTimestamp>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, AriesDatetimeTrans::GetInstance().ToBool( content ), resultValueType );
                break;
            }
            case AriesExprType::YEAR:
            {
                auto content = boost::get<aries_acc::AriesYear>( expr->GetContent() );
                expr = AriesCommonExpr::Create( resultExprType, AriesDatetimeTrans::GetInstance().ToBool( content ), resultValueType );
                break;
            }

            case AriesExprType::NULL_VALUE:
                expr = AriesCommonExpr::Create( resultExprType, false, resultValueType );
                break;
        
            default:
                break;
        }
        expr->SetId( exprId );
    }

    AriesFilterNodeSPtr AriesEngineShell::MakeFilterNode( int nodeId, BiaodashiPointer arg_filter_expr, const vector< int >& arg_columns_id )
    {
        AriesExprBridge bridge;
        AriesCommonExprUPtr expr = bridge.Bridge( arg_filter_expr );
        AlignOptimize(expr);
        int exprId = 0;
        expr->SetId( ++exprId );
        ConvertConstConditionExpr( expr );
        return AriesNodeManager::MakeFilterNode( nodeId, expr, arg_columns_id );
    }

    AriesGroupNodeSPtr AriesEngineShell::MakeGroupNode( int nodeId, const vector< BiaodashiPointer >& arg_group_by_exprs,
            const vector< BiaodashiPointer >& arg_select_exprs )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        vector< AriesCommonExprUPtr > sels;
        for( const auto& sel : arg_select_exprs )
        {
            sels.push_back( bridge.Bridge( sel ) );     // 根据 selectPartStructure 生成 表达式二叉树
            AlignOptimize(sels.back());
            sels.back()->SetId( ++exprId );
        }

        vector< AriesCommonExprUPtr > groups;
        for( const auto& group : arg_group_by_exprs )
        {
            auto bridged = bridge.Bridge( group );
            if( !bridged->IsLiteralValue() )
            {
                bridged->SetId( ++exprId );
                groups.push_back( std::move( bridged ) );
            }
            else
            {
                LOG(INFO)<< "constant expression in group-by clause was filtered here.";
            }
        }
        return AriesNodeManager::MakeGroupNode( nodeId, groups, sels );
    }

    AriesSortNodeSPtr AriesEngineShell::MakeSortNode( int nodeId, const vector< BiaodashiPointer >& arg_order_by_exprs,
            const vector< OrderbyDirection >& arg_order_by_directions, const vector< int >& arg_columns_id )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        vector< AriesCommonExprUPtr > exprs;
        for( const auto& expr : arg_order_by_exprs )
        {
            exprs.push_back( bridge.Bridge( expr ) );
            exprs.back()->SetId( ++exprId );
        }
        return AriesNodeManager::MakeSortNode( nodeId, exprs, bridge.ConvertToAriesOrderType( arg_order_by_directions ), arg_columns_id );
    }

    AriesJoinNodeSPtr AriesEngineShell::MakeJoinNode( int nodeId, BiaodashiPointer equal_join_expr, BiaodashiPointer other_join_expr, int arg_join_hint,
            bool arg_join_hint_2, const vector< int > &arg_columns_id )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        AriesCommonExprUPtr equal = equal_join_expr ? bridge.Bridge( equal_join_expr ) : nullptr;
        AriesCommonExprUPtr other = other_join_expr ? bridge.Bridge( other_join_expr ) : nullptr;
        if( equal )
            equal->SetId( ++exprId );
        if( other )
        {
            other->SetId( ++exprId );
            ConvertConstConditionExpr( other );
        }
        return AriesNodeManager::MakeJoinNode( nodeId, std::move( equal ), std::move( other ), AriesJoinType::INNER_JOIN, arg_join_hint, arg_join_hint_2, arg_columns_id );
    }

    AriesJoinNodeSPtr AriesEngineShell::MakeJoinNodeComplex( int nodeId, BiaodashiPointer equal_join_expr, BiaodashiPointer other_join_expr,
            JoinType arg_join_type, const vector< int > &arg_columns_id )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        AriesCommonExprUPtr equal = equal_join_expr ? bridge.Bridge( equal_join_expr ) : nullptr;
        AriesCommonExprUPtr other = other_join_expr ? bridge.Bridge( other_join_expr ) : nullptr;
        if( equal )
            equal->SetId( ++exprId );
        if( other )
        {
            other->SetId( ++exprId );
            ConvertConstConditionExpr( other );
        }
        return AriesNodeManager::MakeJoinNodeComplex( nodeId, std::move( equal ), std::move( other ), bridge.ConvertToAriesJoinType( arg_join_type ), arg_columns_id );
    }

    AriesColumnNodeSPtr AriesEngineShell::MakeColumnNode( int nodeId, const vector< BiaodashiPointer >& arg_select_exprs, const vector< int >& arg_columns_id,
            int arg_mode )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        vector< AriesCommonExprUPtr > exprs;
        for( const auto& expr : arg_select_exprs )
        {
            exprs.push_back( bridge.Bridge(expr) );   // exprs 中存储了计算表达式树
            AlignOptimize(exprs.back());
            exprs.back()->SetId( ++exprId );
        }
        return AriesNodeManager::MakeColumnNode( nodeId, exprs, arg_mode, arg_columns_id );
    }

    AriesUpdateCalcNodeSPtr AriesEngineShell::MakeUpdateCalcNode( int nodeId, const vector< BiaodashiPointer >& arg_select_exprs,
            const vector< int >& arg_columns_id )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        vector< AriesCommonExprUPtr > exprs;
        for( const auto& expr : arg_select_exprs )
        {
            exprs.push_back( bridge.Bridge( expr ) );
            exprs.back()->SetId( ++exprId );
        }
        return AriesNodeManager::MakeUpdateCalcNode( nodeId, exprs, arg_columns_id );
    }

    AriesOutputNodeSPtr AriesEngineShell::MakeOutputNode()
    {
        return AriesNodeManager::MakeOutputNode();
    }

    AriesLimitNodeSPtr AriesEngineShell::MakeLimitNode( int nodeId, int64_t offset, int64_t size )
    {
        return AriesNodeManager::MakeLimitNode( nodeId, offset, size );
    }

    AriesSetOperationNodeSPtr AriesEngineShell::MakeSetOpNode( int nodeId, SetOperationType type )
    {
        AriesExprBridge bridge;
        return AriesNodeManager::MakeSetOpNode( nodeId, bridge.ConvertToAriesSetOpType( type ) );
    }

    AriesSelfJoinNodeSPtr AriesEngineShell::MakeSelfJoinNode( int nodeId, int joinColumnId, CommonBiaodashiPtr filter_expr,
            const vector< HalfJoinInfo >& join_info, const vector< int >& arg_columns_id )
    {
        int exprId = 0;
        AriesExprBridge bridge;
        SelfJoinParams joinParams;

        if( filter_expr )
        {
            joinParams.CollectedFilterConditionExpr = bridge.Bridge( filter_expr );
            joinParams.CollectedFilterConditionExpr->SetId( ++exprId );
        }

        for( const auto& info : join_info )
        {
            HalfJoinCondition condition;
            condition.JoinType = bridge.ConvertToAriesJoinType( info.HalfJoinType );
            assert( info.JoinConditionExpr );
            condition.JoinConditionExpr = bridge.Bridge( info.JoinConditionExpr );
            condition.JoinConditionExpr->SetId( ++exprId );
            joinParams.HalfJoins.push_back( std::move( condition ) );
        }

        return AriesNodeManager::MakeSelfJoinNode( nodeId, joinColumnId, joinParams, arg_columns_id );
    }

    AriesExchangeNodeSPtr AriesEngineShell::MakeExchangeNode( int nodeId, int dstDeviceId, const vector< int >& srcDeviceId )
    {
        return AriesNodeManager::MakeExchangeNode( nodeId, dstDeviceId, srcDeviceId );
    }

END_ARIES_ENGINE_NAMESPACE
/* namespace AriesEngine */
