import numpy as np

"""
MDPを方策反復法によって迷路を解くプログラム
"""

# 2:ゴール, 1:壁, 0: 道
map = np.array([[0, 1, 0, 1, 2],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0]])

height, width = map.shape

ACTIONS = {
  "UP": 0,
  "RIGHT": 1,
  "DOWN": 2,
  "LEFT": 3
}

def can_move(x, y, a):
  """
  その行動によって移動できるかどうか
  Args:
    x int: x座標
    y int: y座標
    a int: 行動
  
  Returns:
    bool
  """
  if a == ACTIONS["UP"]:
    y -= 1
  if a == ACTIONS["RIGHT"]:
    x += 1
  if a == ACTIONS["DOWN"]:
    y += 1
  if a == ACTIONS["LEFT"]:
    x -= 1

  if x == width or x == -1:
    return False
  if y == height or y == -1:
    return False
  if map[y, x] == 1:
    return False

  return True


def move(x, y, a):
  """
  移動先と報酬を返す
  Args:
    x int: 現在のx座標
    y int: 現在のy座標
    a int: 行動

  Returns
    x int: 移動先のx座標
    y int: 移動先のy座標
    reward float: 報酬
  """
  if can_move(x, y, a) == False:
    return x, y, -1 # 動かずとどまる、動けなかった時の報酬

  if a == ACTIONS["UP"]:
    y -= 1
  if a == ACTIONS["RIGHT"]:
    x += 1
  if a == ACTIONS["DOWN"]:
    y += 1
  if a == ACTIONS["LEFT"]:
    x -= 1
  
  reward = 1 if map[y, x] == 2 else -1 # ゴールだったときと普通に移動したときの報酬を設定

  return x, y, reward

def policy_evaluation(pi, v, gamma):
  """
  方策の評価
  Args:
    pi array: 方策
    v array: 状態価値関数
    gamma: 割引率
  """
  theta = 0.001 # 更新終了のしきい値　更新される値の差がこれより小さくなったら終了
  delta = np.inf
  while delta > theta:
    delta = 0.0
    for y in range(height):
      for x in range(width):
        if map[y, x] != 0: # 道以外は計算しなくていい
          continue
        v_temp = v
        a = pi[y, x]
        x2, y2, r = move(x, y, a)
        if map[y2, x2] == 2:
          v[y, x] = r
        else:
          v[y, x] = r + gamma * v_temp[y2, x2] # ベルマン方程式

        delta = max(delta, abs(v_temp[y, x] - v[y, x]))

        
    

def policy_improvement(pi, v, gamma):
  """
  方策の改善
  Args:
    pi array: 方策
    v array: 状態価値関数
    gamma: 割引率

  Returns:
    has_converged bool: 収束したかどうか
  """
  has_converged = True
  for y in range(height):
    for x in range(width):
      if map[y, x] != 0: #
        continue
      b = pi[y, x]
      max_q = -np.inf
      for a in ACTIONS.values():
        x2, y2, r = move(x, y, a)
        if map[y2, x2] == 2:
          q = r
        else:
          q = r + gamma * v[y2, x2] # 

        if q > max_q:
          max_q = q
          argmax_a = a
          pi[y, x] = argmax_a
      if b != argmax_a:
        has_converged = False
        
  return has_converged


if __name__ == '__main__':
  gamma = 0.9 # 割引率

  v = np.zeros_like(map, dtype=float) # 各位置に対する状態価値関数
  pi = np.zeros_like(map) # 各位置に対する方策

  has_converged = False # 収束したかどうか

  while has_converged is False:
    policy_evaluation(pi, v, gamma) # 方策評価、現在の方策での状態価値vを計算
    has_converged = policy_improvement(pi, v, gamma) # 状態関数に基づいてgreedyに方策を更新

    print("v")
    print(v)

    print("pi")
    print(pi)