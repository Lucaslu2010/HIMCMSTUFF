def small_big(dList):
    newList = []
    max_d = max(dList)
    for a in dList:
        newList.append(max_d - a)
    return newList


def transform_to_maximizing_criterion(x, m):
    transformed_x = []
    for value in x:
        transformed_value = 1 / (abs(value - m))
        transformed_x.append(transformed_value)
    return transformed_x


def normalization(xList):
    newList = []
    x_max = max(xList)
    x_min = min(xList)
    for x in xList:
        newList.append((x - x_min)/(x_max - x_min))
    return newList


def takeSecond(elem):
    return elem[1]
def get_distance(a,b,c,d,e,f,maxList,minList):
    distance_max = ((a - maxList[0])**2 + (b - maxList[1])**2 + (c-maxList[2])**2 + (d - maxList[3])**2 + (e - maxList[4])**2 + (f - maxList[5])**2)**0.5
    distance_min = ((a - minList[0]) ** 2 + (b - minList[1]) ** 2 + (c - minList[2]) ** 2 + (d - minList[3]) ** 2 + (e - minList[4]) ** 2 + (f - minList[5]) ** 2) ** 0.5
    return distance_min/(distance_min + distance_max)

def topsis_main(dList):
    #极值
    maxList = [-1,-1,-1,-1,-1,-1]
    minList = [2,2,2,2,2,2]
    for i in range(6):
        a = dList[i]
        for b in a:
            if b > maxList[i]:
                maxList[i] = b
            if b < minList[i]:
                minList[i] = b

    print(maxList,minList)
    #判断
    nList = []
    for i in range(len(dList)):
        nList.append([str(i + 1), get_distance(dList[i][0], dList[i][1], dList[i][2], dList[i][3], dList[i][4],dList[i][5],maxList,minList)])
    nList.sort(key=takeSecond,reverse=True)
    return nList

def quanzhong(dList):
    a1 = 0.1205
    a2 = 0.2506
    a3 = 0.0805
    a4 = 0.1785
    a5 = 0.0528
    a6 = 0.3226
    newList = []
    for a in dList:
        newList.append([a[0] * a1, a[1] * a2, a[2] * a3, a[3] * a4, a[4] * a5, a[5] * a6])
    return newList


aList = [[85,85,90,80,85,85],[85,85,85,75,85,85],[85,85,90,75,90,85],[70,85,75,80,85,90],[85,85,90,85,90,80],[80,90,80,85,85,90],[85,85,85,85,90,80],[75,80,70,80,85,80],[80,85,80,85,90,75]]
anList = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
for a in range(6):
    a1List = transform_to_maximizing_criterion([aList[0][a],aList[1][a], aList[2][a],aList[3][a],aList[4][a],aList[5][a],aList[6][a],aList[7][a],aList[8][a]], 100)
    anList[0][a] = a1List[0]
    anList[1][a] = a1List[1]
    anList[2][a] = a1List[2]
    anList[3][a] = a1List[3]
    anList[4][a] = a1List[4]
    anList[5][a] = a1List[5]
    anList[6][a] = a1List[6]
    anList[7][a] = a1List[7]
    anList[8][a] = a1List[8]
print(anList)
an2List = []
for a in range(9):
    a2List_temp = normalization(anList[a])
    #print(a2List_temp)
    an2List.append(a2List_temp)
#print(an2List)
#an3List = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

an2List = quanzhong(an2List)
an3List = an2List
print(topsis_main(an3List))