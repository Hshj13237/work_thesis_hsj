"""------33题-----"""
inputs = input("输入列表：")
low, high = input("输入两个整数：").split(", ")
inputlist = []

for num in inputs:
    inputlist.append(num)

sublist = inputlist[int(low):int(high)+1]

print("输出：", end="")
print(", ".join(sublist))


"""--------34题-------"""
n = 30
totalsum = 0
currentsum = 0

for i in range(1, n + 1):
    currentsum += i
    totalsum += currentsum

print("多项式前30项的和为:", totalsum)


"""--------35题--------"""
inputs = input("输入：")
inputdict = {}

for num in inputs:
    inputdict[num] = inputdict.get(num, 0) + 1

resultlist = []
print("输出数码是：", end="")
for num, count in inputdict.items():
    if count == 1:
        resultlist.append(num)

print(", ".join(resultlist))












