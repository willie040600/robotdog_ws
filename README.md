# 8 自由度機器狗：Gazebo 模擬 + Nav2 導航 + Isaac Lab 強化學習

在 ROS 2 Jazzy + Gazebo Harmonic 裡模擬一隻 8 自由度機器狗（每隻腳只有髖、膝兩個 pitch 關節），用 slam_toolbox 建圖、Nav2 自主導航。走路的控制器有兩種，可以切換：

- **`gait`**：手寫的 trot 步態加上腿部 IK（預設）
- **`rl`**：在 Isaac Lab 用強化學習訓練出來的走路策略

```
Nav2 ──/cmd_vel──► 控制器（gait 或 rl） ──/dog/<joint>/cmd_pos──► Gazebo 關節 PD 控制器
  ▲                                                                │
  └── slam_toolbox / AMCL ◄── /scan_filtered ◄── 地面過濾 ◄── /scan ◄─┘
```

---

## 快速開始

```bash
cd ~/robotdog_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

# 邊建圖邊導航，使用 RL 策略（會開 Gazebo 和 RViz）
ros2 launch dog_bringup nav.launch.py controller:=rl
```

大約 10 秒後狗會站起來，Nav2 也啟動完成。在 RViz 工具列按「**Nav2 Goal**」，在地圖上點一下並拖出方向，狗就會走過去。

> 每開一個新的終端機都要先 `source ~/robotdog_ws/install/setup.bash`，否則會出現 `Package 'dog_bringup' not found`。

---

## 環境需求

| 項目 | 版本 |
|---|---|
| OS | Ubuntu 24.04 |
| ROS 2 | Jazzy（含 `navigation2`、`nav2_bringup`、`slam_toolbox`、`ros_gz`、`xacro`、`teleop_twist_keyboard`） |
| Gazebo | Harmonic（gz-sim 8） |
| Python | numpy（RL 推論只用 numpy，不需要 PyTorch / onnxruntime） |
| RL 訓練（選用） | Isaac Lab 3.0 + Isaac Sim 6.2，安裝在 `~/IsaacLab`；NVIDIA GPU |

---

## 常用操作

### 導航

```bash
# SLAM 模式：邊建圖邊導航（預設）
ros2 launch dog_bringup nav.launch.py controller:=rl

# AMCL 模式：用存好的地圖 dog_bringup/maps/dog_world.yaml 定位導航
ros2 launch dog_bringup nav.launch.py controller:=rl slam:=False

# 指定其他地圖
ros2 launch dog_bringup nav.launch.py controller:=rl slam:=False map:=$HOME/robotdog_ws/dog_bringup/maps/xxx.yaml

# 不開 Gazebo 視窗（模擬速度較快，RViz 仍會顯示一切）
ros2 launch dog_bringup nav.launch.py controller:=rl gui:=false
```

AMCL 的初始位置設在原點，也就是狗的生成位置，所以啟動後不用手動點「2D Pose Estimate」。如果雷射點和地圖上的牆對不上，再用它手動校正。

### 存地圖

Nav2 保持開著，在另一個終端機執行：

```bash
source ~/robotdog_ws/install/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/robotdog_ws/dog_bringup/maps/dog_world
```

- 會產生 `dog_world.pgm` 和 `dog_world.yaml`，並**覆蓋**預設地圖。想保留舊圖，就換一個檔名。
- **不要**加 `use_sim_time:=true`，會逾時失敗（`Failed to spin map subscription`）。

### 只開模擬、用鍵盤遙控

```bash
ros2 launch dog_bringup sim.launch.py controller:=rl rviz:=true
ros2 run teleop_twist_keyboard teleop_twist_keyboard     # 另一個終端機
```

這隻狗不能側移，`teleop` 的側移按鍵沒有作用。

---

## Launch 參數

`nav.launch.py`（會自動包含 `sim.launch.py`）：

| 參數 | 預設 | 說明 |
|---|---|---|
| `controller` | `gait` | `gait` 手寫步態 / `rl` 強化學習策略 |
| `slam` | `True` | `True` 用 slam_toolbox 建圖；`False` 用 map_server + AMCL |
| `map` | `maps/dog_world.yaml` | AMCL 模式使用的地圖 |
| `params_file` | `config/nav2_params.yaml` | Nav2 / slam_toolbox / AMCL 參數 |
| `gui` | `true` | 是否開 Gazebo 視窗 |
| `rviz` | `true` | 是否開 RViz |
| `rl_policy` | `dog_flat_v7.npz` | RL 策略檔（在 `dog_gait/policies/`） |
| `rl_stand_when_idle` | `true` | 指令為 0 時，不跑策略、直接維持站姿 |

`sim.launch.py` 另有 `world`（世界檔）參數，`rviz` 的預設是 `false`。

---

## 套件結構

```
robotdog_ws/
├── dog_description/        原始機器人模型（dog.urdf + STL，不修改）
├── dog_bringup/            模擬與導航設定
│   ├── urdf/dog_sim.urdf.xacro    在 dog.urdf 外加 Gazebo 插件：關節 PD、IMU、LiDAR、里程計
│   ├── worlds/dog_world.sdf       8×8 m 房間，有兩道內牆、方塊、圓柱
│   ├── config/gz_bridge.yaml      Gazebo ↔ ROS topic 對應
│   ├── config/nav2_params.yaml    Nav2、slam_toolbox、AMCL 參數
│   ├── config/gait.yaml           手寫步態參數
│   ├── maps/                      存好的地圖
│   ├── rviz/dog.rviz
│   └── launch/{sim,nav}.launch.py
├── dog_gait/               ROS 2 節點（Python）
│   ├── gait_node.py               手寫 trot 步態 + 腿部 IK
│   ├── rl_policy_node.py          執行 RL 策略（numpy MLP，50 Hz）
│   ├── odom_footprint_node.py     3D 里程計 → odom→base_footprint→base_link
│   ├── scan_ground_filter_node.py 移除打到地面的雷射點 → /scan_filtered
│   └── policies/*.npz             訓練好的策略權重
└── dog_rl/                 Isaac Lab 訓練專案（有 COLCON_IGNORE，colcon 不會編）
    ├── assets/dog/usd/            從 dog.urdf 轉出的 USD 模型
    ├── dog_rl/dog_cfg.py          機器人設定（PD 增益、站姿、力矩上限）
    ├── dog_rl/tasks/              訓練任務：環境、獎勵、指令、延遲動作
    └── scripts/                   train / play / gait_eval / export_npz / stand_test
```

### 重要的 topic 和 TF

| Topic | 說明 |
|---|---|
| `/cmd_vel` | 速度指令（Twist：`linear.x`、`angular.z`） |
| `/dog/<joint>/cmd_pos` | 8 個關節的目標角度 |
| `/joint_states`, `/imu` | 關節狀態（200 Hz）、IMU |
| `/dog/odom_3d` → `/odom` | Gazebo 的 3D 里程計 → 2D 里程計（`base_footprint`） |
| `/scan` → `/scan_filtered` | 原始雷射 → 移除地面回波後的雷射（SLAM、AMCL、costmap 都用這個） |

TF 樹：`map → odom → base_footprint → base_link → {FL,FR,RL,RR}_{thigh,calf}, lidar_link, imu_link`

- `base_footprint` 是貼地、永遠水平的座標系；身體的起伏和傾斜都在 `base_footprint → base_link` 這一段。
- Nav2、SLAM、AMCL 都以 `base_footprint` 為機器人座標，所以走路時身體晃動不會讓地圖飄移。

> 模擬中的里程計直接取自 Gazebo 的真實位置（代替真機上的腿部里程計 + IMU 融合），所以 `map → odom` 的大小就等於 SLAM 的誤差。

---

## 兩種控制器

### `gait`：手寫步態（`dog_gait/gait_node.py`）

- trot 步態，週期 0.36 s；用左右步幅差（像坦克一樣）來轉向。
- 參數在 `config/gait.yaml`：
  - `turn_gain`：轉向時左右步幅差的放大倍數。
  - `turn_vx_comp`：原地轉時抵銷往前挪的後退補償。

### `rl`：強化學習策略（`dog_gait/rl_policy_node.py`）

策略的觀測（35 維）和動作（8 維）必須和 Isaac Lab 訓練時完全一致，細節寫在節點檔案開頭的註解。節點另外做了幾件事，讓策略能配合 Nav2：

| 參數 | 預設 | 作用 |
|---|---|---|
| `stand_when_idle` / `idle_time` | `true` / 0.3 s | 指令為 0 超過 0.3 s，就改成維持站姿 |
| `min_turn_wz` | 0.3 | 原地轉的轉速至少 0.3 rad/s（訓練過的範圍） |
| `turn_vx_comp` | 0.2 | 原地轉時送 `vx = -0.2·|wz|`，抵銷 Gazebo 裡往前挪的現象 |
| `yaw_map_want` / `yaw_map_send` | v7 校準值 | 邊走邊轉時，把想要的轉速換算成實際要送的指令（補償小轉速的死區） |

> ⚠️ `turn_vx_comp` 和 `yaw_map_*` 是在 **Gazebo 針對 v7 策略**量出來的。換策略版本、改物理參數或上真機時，都要重新量測（量測方法見下方「步態評估與校準」）。兩個 `yaw_map_*` 都設成 `[0.0, 1.0]` 就停用查表。

---

## RL 訓練（Isaac Lab）

Isaac Lab 安裝在 `~/IsaacLab`（用 `uv` 管理環境）。以下指令都在 `~/IsaacLab` 底下執行：

```bash
cd ~/IsaacLab
export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1
UV=~/.local/bin/uv
```

### 1. 確認機器人模型（站立測試）

```bash
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/stand_test.py --num_envs 16
```

身體高度應約 0.166 m、沒有一隻倒下。

### 2. 訓練

```bash
# 從頭訓練（4096 隻狗、1000 輪，約 35 分鐘；不開畫面）
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/train.py --task Dog-Velocity-Flat-v0 --run_name v8

# 邊訓練邊看（畫面只顯示 64 隻，訓練速度幾乎不受影響）
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/train.py --task Dog-Velocity-Flat-v0 \
    --run_name v8 --viz kit --max_visible_envs 64

# 從現有的檢查點接著訓練（觀測、動作沒變時，比從頭訓練快很多）
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/train.py --task Dog-Velocity-Flat-v0 \
    --run_name v8 --checkpoint ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/model_1600.pt --max_iterations 500
```

紀錄和檢查點會存在 `~/IsaacLab/logs/rsl_rl/dog_flat/<日期>_<run_name>/`，每 50 輪存一次。表現通常在中後段最好，不一定是最後一輪，請比較各輪的獎勵和追蹤誤差後再挑選。

### 3. 播放並匯出

```bash
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/play.py --task Dog-Velocity-Flat-Play-v0 \
    --checkpoint ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/model_XXXX.pt --viz kit
```

播放時會自動匯出到 `<run>/exported/policy.pt` 和 `policy.onnx`。

### 4. 步態評估與校準

```bash
# Isaac 內：固定指令下的步頻、騰空時間、抬腳高度、實際速度
$UV run --extra isaacsim python ~/robotdog_ws/dog_rl/scripts/gait_eval.py \
    --policy ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/exported/policy.pt --vx 0.3 --wz 0.0
```

Gazebo 內的轉向校準：開一套用獨立 `GZ_PARTITION` / `ROS_DOMAIN_ID` 的不開畫面模擬，以 `vx=0.2` 給不同的 `wz`，量測實際轉速，再依結果更新 `yaw_map_*`。

### 5. 部署到 ROS

```bash
# 轉成 numpy 權重（會自動驗證和 TorchScript 的輸出一致）
.venv/bin/python ~/robotdog_ws/dog_rl/scripts/export_npz.py \
    ~/IsaacLab/logs/rsl_rl/dog_flat/<run>/exported/policy.pt \
    ~/robotdog_ws/dog_gait/policies/dog_flat_v8.npz

cd ~/robotdog_ws && colcon build --symlink-install    # 新的 .npz 要重新 build 才會安裝
ros2 launch dog_bringup nav.launch.py controller:=rl rl_policy:=dog_flat_v8.npz
```

### 訓練任務摘要（`dog_rl/dog_rl/tasks/flat_env_cfg.py`）

- **觀測（35 維）**：IMU 角速度、重力方向、速度指令、關節角度與角速度、上一步動作、步態節拍時鐘（週期 0.4 s 的 sin/cos）。
- **動作（8 維）**：關節目標角度 = 站姿 + 0.25 × 動作，以 50 Hz 輸出（物理模擬 200 Hz）。
- **指令**：前進 -0.3 ~ 0.5 m/s、轉向 ±1 rad/s、不側移；20% 站立、25% 原地轉。
- **主要獎勵**：
  - 追蹤速度指令。
  - 對角步態同步，加上著地排程（固定步頻）。
  - 抬腳高度、不拖腳。
  - 維持身高與水平。
  - 站立時靜止、原地轉時不平移。
  - 平滑度。
- **終止**：身體碰地，或任一膝蓋低於 3.5 cm（跪地）。
- **域隨機化**：
  - 摩擦力 0.3–1.5。
  - 身體質量 ±25%、重心偏移。
  - PD 增益 ±30–50%。
  - 關節摩擦、轉子慣量。
  - 50% 的狗動作延遲 20 ms。
  - 隨機推擠。

### 版本演進

| 版本 | 改動 | 解決的問題 |
|---|---|---|
| v1 | 基本速度追蹤 | — |
| v2 | 對角步態同步、抬腳、不拖腳；跪地即終止 | v1 跪著走、小碎步 |
| v3 | 步態節拍時鐘 + 著地排程 | v2 步頻過快 |
| v4 | 25% 原地轉指令、轉向權重加倍 | v3 原地轉只有 21% |
| v5 | 大範圍域隨機化、20 ms 延遲、站立靜止獎勵 | v4 在 Gazebo 轉不動、站著會往後漂 |
| v6 | 原地轉不准平移 | v5 轉向時撞上障礙物 |
| v7 | 轉向追蹤容忍度收緊、練習慢速轉向 | v6 對小轉速沒反應，偏離路徑 |

---

## 疑難排解

| 症狀 | 原因與處理 |
|---|---|
| `Package 'dog_bringup' not found` | 還沒 `source ~/robotdog_ws/install/setup.bash`。 |
| 新增 `.npz` 或節點後找不到 | 要重新 `colcon build --symlink-install`。重 build 前要先 `source /opt/ros/jazzy/setup.bash`。 |
| 存地圖 `Failed to spin map subscription` | 去掉 `use_sim_time:=true`。 |
| 狗亂跳、兩隻狗的位置來回跳、地圖一開始就亂 | 有之前沒關掉的 Gazebo 還在跑（`ps aux \| grep "gz sim"`），先結束它再重啟。 |
| 模擬很慢（RTF 遠低於 1） | Gazebo 畫面很吃效能，改用 `gui:=false`，或不要同時跑 Isaac Lab 訓練。 |
| Nav2 回報 `Failed to make progress`，狗原地晃 | RPP 的角加速度太低時，Nav2 只會送極小的轉速。已把 `max_angular_accel` 調成 16，換參數檔時要保留這個設定。 |
| 狗貼牆或撞到障礙物而卡住 | 用 `/cmd_vel` 手動讓牠走開（例如 `ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.15}}"`，按 Ctrl+C 停止）。可以加大 `footprint_padding` 或 `inflation_radius`。 |
| 地圖飄移、牆出現重影 | 確認 SLAM 用的是 `base_footprint` 和 `/scan_filtered`（已是預設）。 |
| AMCL 模式到了目標附近卻一直失敗、`Resulting plan has 0 poses` | AMCL 定位跳動。`nav2_params.yaml` 已把 `alpha1–4` 調成 0.1、`z_hit` 0.8、`z_rand` 0.2、`sigma_hit` 0.1、`max_beams` 120（Nav2 範例值會讓誤差跳到 0.5 m）。上真機時，`alpha` 要依實際里程計的誤差調大。 |

---

## 已知限制

- **只在模擬中驗證過**，里程計使用的是 Gazebo 的真實位置。
- **Isaac Lab（PhysX）和 Gazebo（DART）的物理差異**：策略在 Gazebo 轉向比在 Isaac 差，目前靠節點裡的補償（`turn_vx_comp`、`yaw_map_*`）修正，這些都要依實際平台重新校準。
- **原地轉在 0.2 rad/s 以下反應很弱**；轉彎時後腳偶爾會多踩幾步。
- **只訓練過平地**，沒有斜坡、台階或不平整地形。
