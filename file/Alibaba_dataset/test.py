if __name__ == '__main__':
    f1 = open("task_pre_40.csv","r")
    lines = f1.readlines()
    for line in lines:
        info = line.strip("\n").split(",")
        num1 = int(info[0])
        num2 = int(info[1])
        if num1 > num2:
            print(line)