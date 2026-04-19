import numpy as np
import socket
import json
import os
import time
import threading

GESTURE_MAP = {
    'door_lock':   0,
    'door_unlock': 1,
    'light_off':   2,
    'light_on':    3
}

ACTIONS = {
    0: 'LIGHT_ON',
    1: 'LIGHT_OFF',
    2: 'DOOR_LOCK',
    3: 'DOOR_UNLOCK',
    4: 'DO_NOTHING'
}
N_ACTIONS = len(ACTIONS)
N_STATES  = 4 * 2 * 2 * 2 * 2 * 2  # 128

Q_TABLE_FILE  = "q_table.npy"
ALPHA         = 0.1
GAMMA         = 0.9
EPSILON       = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.995

ESP32_IP   = "192.168.1.XXX"   # change to your ESP32 IP
ESP32_PORT = 5006

sensor_data = {
    'temp':     25.0,
    'humidity': 60.0,
    'distance': 100.0,
    'motion':   0
}


class QLearningAgent:
    def __init__(self):
        if os.path.exists(Q_TABLE_FILE):
            self.q_table = np.load(Q_TABLE_FILE)
            print("Q-table loaded from file")
        else:
            self.q_table = np.zeros((N_STATES, N_ACTIONS))
            print("New Q-table created")

        self.epsilon    = EPSILON
        self.episode    = 0
        self.light_state = 0
        self.door_state  = 0

    def encode_state(self, gesture, light, door, motion, temp_cat, dist_cat):
        return (gesture  * 32 +
                light    * 16 +
                door     *  8 +
                motion   *  4 +
                temp_cat *  2 +
                dist_cat)

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return np.argmax(self.q_table[state_idx])

    def get_reward(self, gesture, action, motion, temp_cat, dist_cat):
        reward = 0
        if gesture == GESTURE_MAP['light_on']    and action == 0: reward += 10
        if gesture == GESTURE_MAP['light_off']   and action == 1: reward += 10
        if gesture == GESTURE_MAP['door_lock']   and action == 2: reward += 10
        if gesture == GESTURE_MAP['door_unlock'] and action == 3: reward += 10
        if gesture == GESTURE_MAP['light_on']    and action == 1: reward -= 5
        if gesture == GESTURE_MAP['light_off']   and action == 0: reward -= 5
        if gesture == GESTURE_MAP['door_lock']   and action == 3: reward -= 5
        if gesture == GESTURE_MAP['door_unlock'] and action == 2: reward -= 5
        if action == 1 and motion == 0:  reward += 5
        if action == 4 and gesture != 4: reward -= 2
        if gesture == GESTURE_MAP['door_unlock'] and dist_cat == 1: reward += 3
        if temp_cat == 1 and action == 1: reward += 2
        return reward

    def update(self, state_idx, action, reward, next_state_idx):
        old_q    = self.q_table[state_idx, action]
        next_max = np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action] = old_q + ALPHA * (reward + GAMMA * next_max - old_q)

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        self.episode += 1

    def save(self):
        np.save(Q_TABLE_FILE, self.q_table)
        print(f"Q-table saved — episode {self.episode}, epsilon {self.epsilon:.4f}")


agent = QLearningAgent()


def send_command_to_esp32(command):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((ESP32_IP, ESP32_PORT))
        s.sendall((command + "\n").encode())
        s.close()
        print(f"Sent to ESP32: {command}")
    except Exception as e:
        print(f"ESP32 send error: {e}")


def sensor_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 5005))
    server.listen(5)
    print("Sensor receiver listening on port 5005")
    while True:
        conn, addr = server.accept()
        data = conn.recv(1024).decode().strip()
        conn.close()
        try:
            parts = data.split(",")
            sensor_data['temp']     = float(parts[0])
            sensor_data['humidity'] = float(parts[1])
            sensor_data['distance'] = float(parts[2])
            sensor_data['motion']   = int(parts[3])
            print(f"Sensors — Temp:{sensor_data['temp']}C  "
                  f"Dist:{sensor_data['distance']}cm  "
                  f"Motion:{sensor_data['motion']}")
        except Exception as e:
            print(f"Sensor parse error: {e}")


def gesture_receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 5007))
    server.listen(5)
    print("Gesture receiver listening on port 5007")

    while True:
        conn, addr = server.accept()
        data = conn.recv(1024).decode().strip()
        conn.close()

        try:
            payload    = json.loads(data)
            gesture    = payload['gesture']
            confidence = payload['confidence']
        except Exception as e:
            print(f"Gesture parse error: {e}")
            continue

        if confidence < 0.80:
            print(f"Low confidence {confidence:.2f} — skipping")
            continue

        gesture_idx = GESTURE_MAP.get(gesture, -1)
        if gesture_idx == -1:
            continue

        temp     = sensor_data['temp']
        distance = sensor_data['distance']
        motion   = sensor_data['motion']
        temp_cat = 1 if temp >= 30 else 0
        dist_cat = 1 if distance <= 50 else 0

        state_idx = agent.encode_state(
            gesture_idx,
            agent.light_state,
            agent.door_state,
            motion, temp_cat, dist_cat
        )

        action      = agent.choose_action(state_idx)
        action_name = ACTIONS[action]

        print(f"\nGesture:{gesture}  Temp:{temp:.1f}C  "
              f"Dist:{distance:.1f}cm  Motion:{motion}")
        print(f"State:{state_idx}  Action:{action_name}  "
              f"Epsilon:{agent.epsilon:.3f}")

        if action_name == 'LIGHT_ON':
            send_command_to_esp32("LIGHT_ON")
            agent.light_state = 1
        elif action_name == 'LIGHT_OFF':
            send_command_to_esp32("LIGHT_OFF")
            agent.light_state = 0
        elif action_name == 'DOOR_LOCK':
            send_command_to_esp32("DOOR_LOCK")
            agent.door_state = 0
        elif action_name == 'DOOR_UNLOCK':
            send_command_to_esp32("DOOR_UNLOCK")
            agent.door_state = 1

        reward = agent.get_reward(
            gesture_idx, action, motion, temp_cat, dist_cat
        )

        next_state_idx = agent.encode_state(
            gesture_idx,
            agent.light_state,
            agent.door_state,
            motion, temp_cat, dist_cat
        )

        agent.update(state_idx, action, reward, next_state_idx)
        agent.decay_epsilon()

        print(f"Reward:{reward}  Episode:{agent.episode}")

        if agent.episode % 50 == 0:
            agent.save()


if __name__ == "__main__":
    t1 = threading.Thread(target=sensor_receiver, daemon=True)
    t2 = threading.Thread(target=gesture_receiver, daemon=True)
    t1.start()
    t2.start()
    print("RL Agent running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.save()
        print("Saved and exiting.")