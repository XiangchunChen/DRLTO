import sys

if __name__ == '__main__':
    # f1 = open("result.out", "r")
    f1 = open("res.txt", "r")
    lines = f1.readlines()
    # sum = 0
    sum = sys.maxsize
    for line in lines:
        # if "completion_time:" in line:
        #     print(line)
        # line = line.strip("\n")
        # if "completion" in line:
        #     print(line)
        # if "completion_time" in line:
        #     f2.write(line+"\n")
        if int(line) > 810:
            sum = min(sum, int(line))
        # sum = sum + int(line)
    # print(sum/len(lines))
    print(sum)
    f1.close()
    # f2.close()