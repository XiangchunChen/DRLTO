import random
import re

if __name__ == '__main__':
    num = 40
    # f1 = open("Alibaba_dataset/Ali_task_"+str(num)+".csv", "r")
    # f2 = open("Alibaba_dataset/task_info_"+str(num)+".csv", "w")
    f1 = open("ali_batch_task.csv", "r")
    f2 = open("synthetic_batch_task.csv", "w")
    lines = f1.readlines()
    # lines = ["error R10_9,j_910992,1,0,0,"]

    for line in lines:
        try:
            line = line.strip().replace('\n', '')
            info = line.split(",")
            # print(len(line[5]))
            if "task_" in info[0]:
                # print(line)
                continue
            if len(info) != 6:
                # print(line)
                continue
            if ",0,0," in line:
                # print(line)
                continue
            num = int(info[5])
            f2.write(line+"\n")
        except:
            print("wrong",line)


    f1.close()
    f2.close()