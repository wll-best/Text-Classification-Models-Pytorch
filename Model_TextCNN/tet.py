def cat_result():
    #预测结果查看
    import csv
    plabel_li = []
    with open('../data/sem/ntest_cnn_label.tsv', 'r', encoding='utf-8') as fi:
        next(fi)
        rowes = csv.reader(fi, delimiter='\t')
        for row in rowes:
            label = row[3]
            # label_li.append(str(int(label)-1))
            plabel_li.append(label)
        print(plabel_li)
        m1,m2,m3,m4=0,0,0,0
        m5=0
        for el in plabel_li:
            if el=='1':
                m1+=1
            elif(el == '2'):
                m2 += 1
            elif(el == '3'):
                m3 += 1
            elif(el == '4'):
                m4 += 1
            elif(el == '5'):
                m5 += 1
            else:
                print('wrong')
        print(m1,m2,m3,m4,m5)

if __name__ == '__main__':
    cat_result()


