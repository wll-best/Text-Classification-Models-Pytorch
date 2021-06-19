#预测结果查看
import csv
plabel_li = []
with open('../data/sem/ntest_sg_label.tsv', 'r', encoding='utf-8') as fi:
    next(fi)
    rowes = csv.reader(fi, delimiter='\t')
    for row in rowes:
        label = row[3]
        # label_li.append(str(int(label)-1))
        plabel_li.append(label)
    print(plabel_li)
    m1=0
    m5=0
    for el in plabel_li:
        if el=='1':
            m1+=1
        else:
            m5+=1
    print(m1,m5)


