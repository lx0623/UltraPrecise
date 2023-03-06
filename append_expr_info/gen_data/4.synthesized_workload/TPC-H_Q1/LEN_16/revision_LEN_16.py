
import random
def get_random_str(len):
    random_str = ''
    for i in range(0, len):
        t = random.randint(0, 9)
        if i == 0 and t == 0:
            while t == 0:
                t = random.randint(0, 9)
        random_str = random_str + str(t)
    return random_str
        
file_name = '../lineitem.tbl'
file_lineitem = open(file_name)
lines_lineitem = file_lineitem.readlines()
prec = [140,136] 
frac = [2,59]
index = 0
for lineitem_row in lines_lineitem:
    # print(get_random_str(10))
    lineitem_col_list = lineitem_row.split('|')
    
    # l_quantity
    # print(lineitem_col_list[4], '->', end=' ')
    random_str_quantity = get_random_str(prec[0] - frac[0])
    lineitem_col_list[4] = random_str_quantity
    # print(lineitem_col_list[4], end=' ')

    # l_extendedprice
    # print(lineitem_col_list[5], end=' ')
    random_str_extendedprice = get_random_str(prec[1] - frac[1]) + "." + get_random_str(frac[1])
    lineitem_col_list[5] = random_str_extendedprice
    # print(lineitem_col_list[5], end=' ')

    for col in lineitem_col_list[:-1]:
        print(col, end='|')
    print(lineitem_col_list[-1], end='')
file_lineitem.close()

