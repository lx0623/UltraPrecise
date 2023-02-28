"""
生成compare_function.cu中所需的matrix
python3 gen_matrix.py
"""

from collections import OrderedDict

AriesValueTypeNameList = [
    'UNKNOWN',
    'INT8', 'INT16', 'INT32', 'INT64', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'DECIMAL', 'FLOAT', 'DOUBLE',
    'CHAR', 
    'BOOL', 
    'DATE', 'TIME', 'DATETIME', 'TIMESTAMP', 'YEAR', 
    'LIST', 
    'COMPACTDECIMAL',
]
AriesValueTypeNameList_numeric = ['INT8', 'INT16', 'INT32', 'INT64', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'DECIMAL', 'FLOAT', 'DOUBLE']
AriesValueTypeNameList_timeAbout = ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP', 'YEAR']
AriesValueTypeNameList_compactDecimal = AriesValueTypeNameList_numeric
AriesValueTypeNameList_char = ['CHAR']
AriesValueTypeNameList_other = ['BOOL', 'LIST']

AriesValueTypeDict = OrderedDict(INT8='int8_t', INT16='int16_t', INT32='int32_t', INT64='int64_t',
                                 UINT8='uint8_t', UINT16='uint16_t', UINT32='uint32_t', UINT64='uint64_t',
                                 DECIMAL='Decimal', FLOAT='float', DOUBLE='double',
                                 DATE='AriesDate', TIME='AriesTime', DATETIME='AriesDatetime', TIMESTAMP='AriesTimestamp', YEAR='AriesYear',
                                
                                 )

AriesValueHasNull = [0, 1]

AriesComparisonOpTypeNameList = ['EQ', 'NE', 'GT', 'LT', 'GE', 'LE',
                                 'IN', 'NOTIN',
                                 'LIKE', ]
AriesComparisonOpTypeNameList_numeric = ['EQ', 'NE', 'GT', 'LT', 'GE', 'LE']
AriesComparisonOpTypeNameList_timeAbout = ['EQ', 'NE', 'GT', 'LT', 'GE', 'LE']
AriesComparisonOpTypeNameList_char = ['EQ', 'NE', 'GT', 'LT', 'GE', 'LE', 'LIKE', ]
AriesComparisonOpTypeNameList_compactDecimal = ['EQ', 'NE', 'GT', 'LT', 'GE', 'LE']

AriesComparisonOpTypeDict = OrderedDict(EQ='equal', NE='notEqual', GT='greater', LT='less', GE='greaterOrEqual',
                                        LE='lessOrEqual', IN='in', NOTIN='notIn', LIKE='like',
                                        )

def gen_statement_numeric():
    """数字类型"""
    device_statement = []
    host_matrix_statement = []
    for left_type_name in AriesValueTypeNameList_numeric:
        for right_type_name in AriesValueTypeNameList_numeric:
            for left_has_null in AriesValueHasNull:
                for right_has_null in AriesValueHasNull:
                    for op_type_name in AriesComparisonOpTypeNameList_numeric:
                        op_type = AriesComparisonOpTypeDict.get(op_type_name)
                        if left_has_null and not right_has_null:
                            template_function_name = op_type+'_leftHasNull_right'
                        elif not left_has_null and right_has_null:
                            template_function_name = op_type+'_left_rightHasNull'
                        elif left_has_null and right_has_null:
                            template_function_name = op_type+'_leftHasNull_rightHasNull'
                        else:
                            template_function_name = op_type

                        left_type = AriesValueTypeDict.get(left_type_name)
                        right_type = AriesValueTypeDict.get(right_type_name)
                        statement = f'{template_function_name}<{left_type}, {right_type}>'
                        device_statement.append(f'__device__ CompareFunctionPointer '
                                   f'{left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}'
                                   f' = {statement};')
                        left_type_index = AriesValueTypeNameList.index(left_type_name)
                        l_hn_index = AriesValueHasNull.index(left_has_null)
                        right_type_index = AriesValueTypeNameList.index(right_type_name)
                        r_hn_index = AriesValueHasNull.index(right_has_null)
                        op_type_index = AriesComparisonOpTypeNameList.index(op_type_name)
                        host_matrix_statement.append(f'    cudaMemcpyFromSymbol( &( host_matrix'
                                   f'[{left_type_index}][{l_hn_index}][{right_type_index}][{r_hn_index}][{op_type_index}]'
                                   f' ), {left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}, '
                                   f'sizeof(CompareFunctionPointer) );')
    return device_statement, host_matrix_statement

def gen_statement_timeAbout():
    """时间类型"""
    device_statement = []
    host_matrix_statement = []
    for left_type_name in AriesValueTypeNameList_timeAbout:
        for right_type_name in AriesValueTypeNameList_timeAbout:
            for left_has_null in AriesValueHasNull:
                for right_has_null in AriesValueHasNull:
                    for op_type_name in AriesComparisonOpTypeNameList_timeAbout:
                        if left_type_name != right_type_name:
                            if (left_type_name == 'DATE' and right_type_name == 'DATETIME') or (left_type_name=='DATETIME' and right_type_name=='DATE'):
                                pass
                            else:
                                continue
                        op_type = AriesComparisonOpTypeDict.get(op_type_name)
                        if left_has_null and not right_has_null:
                            template_function_name = op_type+'_leftHasNull_right'
                        elif not left_has_null and right_has_null:
                            template_function_name = op_type+'_left_rightHasNull'
                        elif left_has_null and right_has_null:
                            template_function_name = op_type+'_leftHasNull_rightHasNull'
                        else:
                            template_function_name = op_type

                        left_type = AriesValueTypeDict.get(left_type_name)
                        right_type = AriesValueTypeDict.get(right_type_name)
                        statement = f'{template_function_name}<{left_type}, {right_type}>'
                        device_statement.append(f'__device__ CompareFunctionPointer '
                                   f'{left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}'
                                   f' = {statement};')
                        left_type_index = AriesValueTypeNameList.index(left_type_name)
                        l_hn_index = AriesValueHasNull.index(left_has_null)
                        right_type_index = AriesValueTypeNameList.index(right_type_name)
                        r_hn_index = AriesValueHasNull.index(right_has_null)
                        op_type_index = AriesComparisonOpTypeNameList.index(op_type_name)
                        host_matrix_statement.append(f'    cudaMemcpyFromSymbol( &( host_matrix'
                                   f'[{left_type_index}][{l_hn_index}][{right_type_index}][{r_hn_index}][{op_type_index}]'
                                   f' ), {left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}, '
                                   f'sizeof(CompareFunctionPointer) );')
    return device_statement, host_matrix_statement

def gen_statement_compactDecimal():
    """compactDeciaml"""
    device_statement = []
    host_matrix_statement = []
    left_type_name = 'COMPACTDECIMAL'
    for right_type_name in AriesValueTypeNameList_compactDecimal:
        for left_has_null in AriesValueHasNull:
            for right_has_null in AriesValueHasNull:
                for op_type_name in AriesComparisonOpTypeNameList_compactDecimal:
                    op_type = AriesComparisonOpTypeDict.get(op_type_name)
                    if left_has_null and not right_has_null:
                        template_function_name = op_type+'_compactDecimalHasNull_right'
                    elif not left_has_null and right_has_null:
                        template_function_name = op_type+'_compactDecimal_rightHasNull'
                    elif left_has_null and right_has_null:
                        template_function_name = op_type+'_compactDecimalHasNull_rightHasNull'
                    else:
                        template_function_name = op_type+'_compactDecimal_right'
                    right_type = AriesValueTypeDict.get(right_type_name)
                    statement = f'{template_function_name}<{right_type}>'
                    device_statement.append(f'__device__ CompareFunctionPointer '
                                f'{left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}'
                                f' = {statement};')
                    left_type_index = AriesValueTypeNameList.index(left_type_name)
                    l_hn_index = AriesValueHasNull.index(left_has_null)
                    right_type_index = AriesValueTypeNameList.index(right_type_name)
                    r_hn_index = AriesValueHasNull.index(right_has_null)
                    op_type_index = AriesComparisonOpTypeNameList.index(op_type_name)
                    host_matrix_statement.append(f'    cudaMemcpyFromSymbol( &( host_matrix'
                                f'[{left_type_index}][{l_hn_index}][{right_type_index}][{r_hn_index}][{op_type_index}]'
                                f' ), {left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}, '
                                f'sizeof(CompareFunctionPointer) );')

    right_type_name = 'COMPACTDECIMAL'
    for left_type_name in AriesValueTypeNameList_compactDecimal:
        for left_has_null in AriesValueHasNull:
            for right_has_null in AriesValueHasNull:
                for op_type_name in AriesComparisonOpTypeNameList_compactDecimal:
                    op_type = AriesComparisonOpTypeDict.get(op_type_name)
                    if left_has_null and not right_has_null:
                        template_function_name = op_type+'_leftHasNull_compactDecimal'
                    elif not left_has_null and right_has_null:
                        template_function_name = op_type+'_left_compactDecimalHasNull'
                    elif left_has_null and right_has_null:
                        template_function_name = op_type+'_leftHasNull_compactDecimalHasNull'
                    else:
                        template_function_name = op_type+'_left_compactDecimal'

                    left_type = AriesValueTypeDict.get(left_type_name)
                    statement = f'{template_function_name}<{left_type}>'
                    device_statement.append(f'__device__ CompareFunctionPointer '
                                f'{left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}'
                                f' = {statement};')
                    left_type_index = AriesValueTypeNameList.index(left_type_name)
                    l_hn_index = AriesValueHasNull.index(left_has_null)
                    right_type_index = AriesValueTypeNameList.index(right_type_name)
                    r_hn_index = AriesValueHasNull.index(right_has_null)
                    op_type_index = AriesComparisonOpTypeNameList.index(op_type_name)
                    host_matrix_statement.append(f'    cudaMemcpyFromSymbol( &( host_matrix'
                                f'[{left_type_index}][{l_hn_index}][{right_type_index}][{r_hn_index}][{op_type_index}]'
                                f' ), {left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}, '
                                f'sizeof(CompareFunctionPointer) );')
    return device_statement, host_matrix_statement

def gen_statement_char():
    device_statement = []
    host_matrix_statement = []
    for left_type_name in AriesValueTypeNameList_char:
        for right_type_name in AriesValueTypeNameList_char:
            for left_has_null in AriesValueHasNull:
                for right_has_null in AriesValueHasNull:
                    for op_type_name in AriesComparisonOpTypeNameList_char:
                        op_type = AriesComparisonOpTypeDict.get(op_type_name)
                        if left_has_null and not right_has_null:
                            template_function_name = op_type+'_charHasNull_char'
                        elif not left_has_null and right_has_null:
                            template_function_name = op_type+'_char_charHasNull'
                        elif left_has_null and right_has_null:
                            template_function_name = op_type+'_charHasNull_charHasNull'
                        else:
                            template_function_name = op_type+'_char_char'

                        left_type = AriesValueTypeDict.get(left_type_name)
                        right_type = AriesValueTypeDict.get(right_type_name)
                        statement = f'{template_function_name}'
                        device_statement.append(f'__device__ CompareFunctionPointer '
                                   f'{left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}'
                                   f' = {statement};')
                        left_type_index = AriesValueTypeNameList.index(left_type_name)
                        l_hn_index = AriesValueHasNull.index(left_has_null)
                        right_type_index = AriesValueTypeNameList.index(right_type_name)
                        r_hn_index = AriesValueHasNull.index(right_has_null)
                        op_type_index = AriesComparisonOpTypeNameList.index(op_type_name)
                        host_matrix_statement.append(f'    cudaMemcpyFromSymbol( &( host_matrix'
                                   f'[{left_type_index}][{l_hn_index}][{right_type_index}][{r_hn_index}][{op_type_index}]'
                                   f' ), {left_type_name}_{left_has_null}_{right_type_name}_{right_has_null}_{op_type_name}, '
                                   f'sizeof(CompareFunctionPointer) );')
    return device_statement, host_matrix_statement

def gen_statement():
    """
    generate device_statement like blow:
    __device__ CompareFunctionPointer INT8_0_INT8_0_EQ = equal<int8_t, int8_t>;
    __device__ CompareFunctionPointer INT8_0_INT8_0_NE = notEqual<int8_t, int8_t>;
    ...

    generate host_matrix_statement like blow:
    cudaMemcpyFromSymbol( &( host_matrix[1][0][1][0][0] ), INT8_0_INT8_0_EQ, sizeof(CompareFunctionPointer) );
    cudaMemcpyFromSymbol( &( host_matrix[1][0][1][0][1] ), INT8_0_INT8_0_NE, sizeof(CompareFunctionPointer) );
    ...
    """
    device_statement = []
    host_matrix_statement = []

    l1, l2 = gen_statement_numeric()
    device_statement.extend(l1)
    host_matrix_statement.extend(l2)

    l1, l2 = gen_statement_timeAbout()
    device_statement.extend(l1)
    host_matrix_statement.extend(l2)

    l1, l2 = gen_statement_compactDecimal()
    device_statement.extend(l1)
    host_matrix_statement.extend(l2)
    
    l1, l2 = gen_statement_char()
    device_statement.extend(l1)
    host_matrix_statement.extend(l2)

    with open('device_statement.h', 'w') as f:
        f.write( '\n'.join(device_statement) )
    with open('host_matrix_statement.h', 'w') as f:
        f.write( '\n'.join(host_matrix_statement) )

if __name__ == '__main__':
    gen_statement()

