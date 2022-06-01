import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DQN_environment import MultiHopNetwork
from Task import Task
from RL_brain import PolicyGradient
import os
import logging

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_network(env, agent,task_num):
    step = 0
    final_action = 0
    completion_time_dic = {}
    reward_dic = {}

    for episode in range(1):
        # initial observation
        print("episode: ", episode)
        observation = env.reset()
        episode_done = False
        time_step = 1
        max_time = 10
        task_count = 0
        task_time_dic = {}
        subtask_time_dic = {}
        avg_reward_dic = {} # key: task_id value: time
        while not episode_done or time_step < max_time:
            # RL choose action based on observation
            print("-------------------timestep: ", time_step, "-------------------")
            tasks = getTasksByTime(env.taskList, time_step)
            task_count += len(tasks)

            if len(tasks) != 0:
                for i in range(len(tasks)):
                    task = tasks[i]
                    print("task:", task.subId)
                    action = agent.choose_action(observation)

                    observation_, reward, done, finishTime = env.step(action, task, time_step)
                    # average_waitTime, average_ctime
                    if task.taskId in avg_reward_dic.keys():
                        avg_reward_dic[task.taskId] = max(avg_reward_dic[task.taskId], reward)
                    else:
                        avg_reward_dic[task.taskId] = reward

                    subtask_time_dic[task.subId] = finishTime-time_step+1
                    # avg_reward_dic
                    # avg_reward += reward
                    if task.taskId in task_time_dic.keys():
                        task_time_dic[task.taskId] = max(task_time_dic[task.taskId], finishTime-time_step+1)
                    else:
                        task_time_dic[task.taskId] = finishTime-time_step+1
                    # task_time_dic[task] = finishTime-time_step

                    agent.store_transition(observation, action, reward)

                    if (step > 10) and (step % 5 == 0):
                        agent.learn()

                    observation = observation_

            else:
                env.add_new_state(time_step)
            print(task_count)
            # ToRevise
            if task_count == task_num:
                episode_done = True
                completion_time = 0
                for task, time_value in task_time_dic.items():
                    print(task, time_value)
                    completion_time += time_value
                print("subtask_time_dic")
                print(subtask_time_dic)
                completion_time_dic[episode] = completion_time
                avg_reward = 0
                for task, reward in avg_reward_dic.items():
                    avg_reward += reward
                print("completion_time:", completion_time)
                reward_dic[episode] = avg_reward
                print("reward:", avg_reward)
                break

            time_step += 1
        step += 1
    agent.save_net()
    return completion_time_dic, reward_dic

def checkAllocated(taskList):
    res = False
    for task in taskList:
        if not task.isAllocated:
            res = True
    return res

def getTasksByTime(taskList, time_step):
    tasks = []
    for task in taskList:
        if task.release_time == time_step:
            tasks.append(task)
    sorted(tasks, key=lambda task: task.subId)
    return tasks

def destory(destory_path):
    df = pd.read_csv(destory_path)
    df.to_csv("file/now_schedule.csv", index=0)

def plotCompletionTime(completion_time_dic,name):
    f1 = open("result/"+name+".csv", "w")
    x = []
    y = []
    for key, value in completion_time_dic.items():
        f1.write(str(key)+","+str(value)+"\n")
        x.append(key)
        y.append(value)
    f1.close()
    plt.plot(x, y)
    plt.ylabel(name)
    plt.xlabel('training episodes')
    plt.show()

def run_model(env, agent,task_num):

    final_action = 0
    completion_time_dic = {}
    reward_dic = {}
    observation = env.reset()
    episode_done = False
    time_step = 1
    max_time = 10
    task_count = 0
    task_time_dic = {}
    subtask_time_dic = {}
    avg_reward_dic = {} # key: task_id value: time
    agent.restore_net()
    while not episode_done or time_step < max_time:
        # RL choose action based on observation
        print("-------------------timestep: ", time_step, "-------------------")
        tasks = getTasksByTime(env.taskList, time_step)
        task_count += len(tasks)

        if len(tasks) != 0:
            for i in range(len(tasks)):
                task = tasks[i]
                print("task:", task.subId)
                action = agent.choose_action(observation)

                observation_, reward, done, finishTime = env.step(action, task, time_step)
                # average_waitTime, average_ctime
                if task.taskId in avg_reward_dic.keys():
                    avg_reward_dic[task.taskId] = max(avg_reward_dic[task.taskId], reward)
                else:
                    avg_reward_dic[task.taskId] = reward

                subtask_time_dic[task.subId] = finishTime-time_step+1
                # avg_reward_dic
                # avg_reward += reward
                if task.taskId in task_time_dic.keys():
                    task_time_dic[task.taskId] = max(task_time_dic[task.taskId], finishTime-time_step+1)
                else:
                    task_time_dic[task.taskId] = finishTime-time_step+1
                # task_time_dic[task] = finishTime-time_step

                # agent.store_transition(observation, action, reward, observation_)
                # if (step > 10) and (step % 5 == 0):
                #     agent.learn()

                observation = observation_

        else:
            env.add_new_state(time_step)
        print(task_count)
        # ToRevise
        if task_count == task_num:
            episode_done = True
            completion_time = 0
            for task, time_value in task_time_dic.items():
                print(task, time_value)
                completion_time += time_value
            print("subtask_time_dic")
            print(subtask_time_dic)
            # completion_time_dic[episode] = completion_time
            avg_reward = 0
            for task, reward in avg_reward_dic.items():
                avg_reward += reward
            print("completion_time:", completion_time)
            # reward_dic[episode] = avg_reward
            print("reward:", avg_reward)
            break
        time_step += 1

if __name__ == "__main__":
    # network_node_path = "file/device25/graph_node.csv"
    # network_edge_path = "file/device25/graph_edge.csv"
    # device_path = "file/device25/device_info.csv"
    # schedule_path = "file/device25/now_schedule.csv"
    # destory_path = "file/device25/result/now_schedule.csv"
    # # DDQN/file/device25
    # edges_devices_num = 49
    # devices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # task_num = 45
    ali_data = "Alibaba_dataset"
    task_file_path = "file/"+ali_data+"/task_info_40.csv"
    task_pre_path = "file/"+ali_data+"/task_pre_40.csv"
    network_node_path = "file/network_node_info.csv"
    network_edge_path = "file/network_edge_info.csv"
    device_path = "file/device_info.csv"
    schedule_path = "file/now_schedule.csv"
    destory_path = "result/now_schedule.csv"
    edges_devices_num = 16
    devices =[1,2,3,4,5,6,7,8]
    # DQN/file/device25
    # edges_devices_num = 49
    # devices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # network_node_path = "file/device50/graph_node.csv"
    # network_edge_path = "file/device50/graph_edge.csv"
    # device_path = "file/device50/device_info.csv"
    # schedule_path = "file/device50/now_schedule.csv"
    # destory_path = "file/device50/result/now_schedule.csv"
    # edges_devices_num = 100
    # devices =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    # task_num = 45
    # edges_devices_num = 16
    # devices =[1,2,3,4,5,6,7,8]
    # TODO: 120
    task_num = 129

    destory(destory_path)
    dic_task_time = {}
    env = MultiHopNetwork(devices,edges_devices_num, schedule_path, network_edge_path, network_node_path, device_path, task_file_path, task_pre_path)
    agent = PolicyGradient(env.n_actions, env.n_features)
    # run_model(env, agent, task_num)

    completion_time_dic, reward_dic = run_network(env, agent, task_num)
    min_sum = sys.maxsize
    plotCompletionTime(completion_time_dic, "completion_time")
    plotCompletionTime(reward_dic, "reward")
    agent.plot_cost()
    # get TF logger