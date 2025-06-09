# PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics
## My Update
### preprocess.py

| 參數名稱              | 說明                                                               |
| ----------------- | ---------------------------------------------------------------- |
| `--model_path`    | 輸入的 Gaussian 模型目錄，需包含 `point_cloud/iteration_xx/point_cloud.ply` |
| `--output_path`   | 輸出分群 `.ply` 與對應 `.json` 的資料夾路徑(如果不是"output_groups", merge_gaussian.py 裡面在 load 的時候要改變 path)                                   |
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
### merge_gaussian.py
* 預設將 output_groups 底下的 ply 與 json 合併成一個 gaussian ply file 與 json file
* 預設存在 merge_dir 下面 
```shell
python merge_gaussians.py 
```
###  my_simulation.py
| 參數名稱               | 說明                                    |
| ------------------ | ------------------------------------- |
| `--model_path`     | 原始 Gaussian 模型所在資料夾（get_camera_view 需要讀取 camera data 來做 render 時的 camera 參數） |
| `--merge_gaussian` | 合併後的 `.ply` 模型，包含所有高斯點（沒做 filling）           |
| `--phys_config`    | 對應 `.ply` 的 material label 的 JSON，例如 `*_phys.json` |
| `--config`         | 模擬與渲染的全域設定檔（包含時間、邊界條件、材質設定等）          |
| `--output_path`    | 渲染圖片與影片的輸出資料夾                         |
| `--render_img`     | 是否輸出每一幀的渲染影像                          |
| `--compile_video`  | 是否將渲染的影像合成為影片                         |
| `--white_bg`       | 是否啟用白色背景（預設為黑色）                       |

```bash
python my_simulation.py \
  --model_path ./model/ficus_whitebg-trained/ \
  --merge_gaussian merge_dir/merged_gaussians.ply \
  --output_path test_output \
  --config ./config/my_ficus.json \
  --phys_config merge_dir/merged_gaussians_phys.json \
  --render_img \
  --compile_video \
  --white_bg
```
### Setup
* 建議先安裝原本 physGaussian 的 requirement.txt 再補齊剩下的，剩下的都很好裝，跑一跑就知道那些沒裝了
* 或是直接安裝 environment.yml
