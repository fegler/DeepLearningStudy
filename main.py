
class calculator:
    def __init__(self):
        self.result = 0
    def setfirst(self, start):
        self.result = start
    def add(self, num):
        self.result = self.result + num
    def sub(self, num):
        self.result = self.result - num
    def mul(self, num):
        self.result = self.result * num
    def div(self, num):
        self.result = self.result / num
    def answer(self):
        return self.result

cal1 = calculator()

for i in range(1,4):
    print("menu: 1.set first value   2.add   3.sub   4.mul   5.div")
    menu = int(input())
    print("input number")
    number = int(input())

    if menu == 1:
        cal1.setfirst(number)
    elif menu == 2:
        cal1.add(number)
    elif menu == 3:
        cal1.sub(number)
    elif menu == 4:
        cal1.mul(number)
    else:
        cal1.div(number)

ans = cal1.answer()
print(ans, type(ans))
