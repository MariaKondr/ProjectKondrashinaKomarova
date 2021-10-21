import numpy as np
import gym
import matplotlib.pyplot as plt

def info_for_env(env, num_states):
    print("Максимальное положение и скорость: ", env.observation_space.high)
    print("Минимальное положение и скорость: ", env.observation_space.low)
    print("Разность макс и мин:", env.observation_space.high - env.observation_space.low)
    print("Число возможных перемещений: ", env.action_space.n) #(0 Ускорение влево, 1 Не ускоряться, 2 Ускорение вправо)
    print("Количество разных положений и скоростей: ", num_states)
    print('')

env = gym.make('MountainCar-v0') # подключаем среду
env.reset() # начальное состояние среды

def QLearning(env, alpha, gamma, epsilon, episodes, n):
    # смотрим, какие значения может принимать положение по горизонтали и скорость
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1 # округляем значения  для последующего создания Q-таблицы

    info_for_env(env, num_states)

    # заполняем таблицу нулями
    Q_table = np.zeros([num_states[0], num_states[1], env.action_space.n])

    reward_list = [] # хранит все вознаграждения за выбранный нами период
    ave_reward_list = [] # сохраняем средее значение за период

    decrease = epsilon / episodes # насколько уменьшаем эпсилон каждый эпизод

    for i in range(episodes):
        done = False
        res_reward, reward = 0, 0 # вознаграждение за эпизод, награда в текущем состоянии
        state = env.reset() # обнуление среды

        state_rou = (state - env.observation_space.low) * np.array([10, 100]) # положение в конкретный момент
        state_rou = np.round(state_rou, 0).astype(int) # округление значения

        while done != True:
            if i >= (episodes - 1): # для вывода последних игр
                env.render() # отрисовываем эпизод

            if np.random.random() < 1 - epsilon: # np.random.random - [0,1)
                action = np.argmax(Q_table[state_rou[0], state_rou[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            state2, reward, done, info = env.step(action) # берем следующий шаг

            state2_rou = (state2 - env.observation_space.low) * np.array([10, 100]) # считаем его положение
            state2_rou = np.round(state2_rou, 0).astype(int)

            if done and state2[0] >= 0.5: # если игра пройдена в конкретном эпизоде
                Q_table[state_rou[0], state_rou[1], action] = reward

            else:
                new_reward = (1 - alpha) * Q_table[state_rou[0], state_rou[1], action] + alpha * (
                            reward + gamma * np.max(Q_table[state2_rou[0], state2_rou[1]])) # формула бельмана

                Q_table[state_rou[0], state_rou[1], action] = new_reward

            res_reward += reward # накапливем вознаграждения за игру
            state_rou = state2_rou

        if epsilon > 0: # меняем эпсилон(уменьшаем)
            epsilon -= decrease

        reward_list.append(res_reward) # сохраняем вознаграждение за эпизод

        if (i + 1) % n == 0: # считаем среднее за n эпизодов
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print(f"Эпизод {i + 1}, Средняя награда: {ave_reward}")

    return ave_reward_list, Q_table # возвращаем список средних значени и Q таблицу

n = 100 # для скольких выводится ср значение награды
epis = 100000
rewards, Q_table = QLearning(env, 0.2, 0.9, 0.8, epis, n) # начинаем обучение

# вывод диаграммы
plt.plot(rewards)
plt.title('Статистика обучения')
plt.xlabel('Эпизод, *100')
plt.ylabel('Награда')
plt.show()

print('')
print('После тренировки:')
episodes = 100
ave_res_reward = 0

for _ in range(episodes):
    state = env.reset()
    state_rou = (state - env.observation_space.low) * np.array([10, 100])  # состояние в конкретный момент
    state_rou = np.round(state_rou, 0).astype(int)
    res_reward = 0
    done = False
    while not done:
        action = np.argmax(Q_table[state_rou[0], state_rou[1]])
        state, reward, done, info = env.step(action)
        res_reward += reward
        state2_rou = (state - env.observation_space.low) * np.array([10, 100])  # считаем его состояние
        state2_rou = np.round(state2_rou, 0).astype(int)
        state_rou = state2_rou
    ave_res_reward += res_reward

env.close()# конец работы со средой

print(f"Результат после {episodes} эпизодов:")
print(f"Среднее значение награды для эпизодов: {ave_res_reward/episodes}")