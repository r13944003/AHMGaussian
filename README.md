# AHMGaussian: Automatic Hybrid-Material Simulation with Gaussians

## My Update
### preprocess.py

| 參數名稱              | 說明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| `--model_path`    | 輸入的 Gaussian 模型目錄，需包含 `point_cloud/iteration_xx/point_cloud.ply` |
| `--output_path`   | 輸出分群 `.ply` 與對應物理參數 `.json` 的資料夾路徑                                |
| `--camera_config` | NeRF 格式的相機設定（例如 `transforms_train.json`）                         |
| `--camera_list`   | 這部分需要是 julia 的 mask folder，我在跟她確認格式                                      |

**目前是指透過一張 segment mask 做分群，把 seg_map.png 放在與 preprocess.py 同一層資料夾下面，只有做分群沒有 filling, transorm 等其他的 preprocess**
**目前只會儲存 maertial 的 label 在 json file裡面，直到 simulation 時才會透過 table 讀取 physic parameter**
```shell
python preprocess.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --output_path output_groups \
  --camera_config ../nerf_synthetic/ficus/transforms_train.json \
  --camera_list ../nerf_synthetic/ficus/
```
###  my_simulation.py
| 參數名稱               | 說明                                    |
| ------------------ | ------------------------------------- |
| `--model_path`     | 原始 Gaussian 模型所在資料夾（get_camera_view 需要讀取 camera data 來做 render 時的 camera 參數） |
| `--merge_folder`      | Preprocess.py 的 output_path `.ply` 模型，           |
| `--config`         | 模擬與渲染的全域設定檔（包含時間、邊界條件、材質設定等）          |
| `--output_path`    | 渲染圖片與影片的輸出資料夾                         |
| `--render_img`     | 是否輸出每一幀的渲染影像                          |
| `--compile_video`  | 是否將渲染的影像合成為影片                         |
| `--white_bg`       | 是否啟用白色背景（預設為黑色）                       |

```bash
python my_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --merge_folder output_groups \
  --output_path test_output \
  --config ./config/myficus_config_sand.json \
  --render_img \
  --compile_video \
  --white_bg
```
### Setup
* 建議先安裝原本 physGaussian 的 requirement.txt 再補齊剩下的，剩下的都很好裝，跑一跑就知道那些沒裝了
* 或是直接安裝 environment.yml
