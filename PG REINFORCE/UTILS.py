import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def plot(rewards , avg_rewards):
    plt.figure(1)
    plt.clf()
    plt.title("Traning")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.plot(rewards, label = 'Rewards')
    plt.plot(avg_rewards, label = 'Average Rewards')
    plt.pause(0.001)

