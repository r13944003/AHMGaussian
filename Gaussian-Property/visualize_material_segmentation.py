import os
import argparse
import numpy as np
import matplotlib as mpl
from PIL import Image

from utils.vis_utils import parse_txt_file, filter_and_process, \
    filter_and_process, parse_material, make_legend, visualize_and_save_segmentation,cat_rgba

def vis_material_seg(base_path, vis_seg_save_base):
    case_list = os.listdir(base_path)
    cmap_tab10 = mpl.colormaps['tab10']

    for path in case_list:
        case_name = os.path.join(base_path, path)

        case_name_last = case_name.split("/")[-1]
        image_base = f"{case_name}/images"
        feature_base = f"{case_name}/seg"
        number_view = len(os.listdir(image_base))

        gpt_txt = os.path.join(base_path, path, f"{case_name_last}.txt")
        parsed_data = parse_txt_file(gpt_txt)

        result_vis_seg_base = os.path.join(vis_seg_save_base, case_name_last)
        os.makedirs(result_vis_seg_base, exist_ok=True)

        for i in range(number_view):
            id = os.listdir(image_base)[i].split('.')[0]
            materials = filter_and_process(parsed_data, int(id))
            converted_list, mat_names = parse_material(materials)

            show = False
            legend = make_legend([cmap_tab10(i) for i in range(len(mat_names))], mat_names, 
                        savefile=os.path.join(result_vis_seg_base, f'{id}_legend.png'), show=show)

            img_path = os.path.join(image_base, id + '.png')
            s_path = os.path.join(feature_base, id + '_s.npy')
            ss = np.load(s_path)

            output_path = os.path.join(result_vis_seg_base, f"{id}_seg.png")
            vlm_result = visualize_and_save_segmentation(img_path, s_path, output_path, converted_list, alpha=0.9)

            print(case_name_last)
            final_result_vis = cat_rgba(vlm_result, legend)
            final_result_vis.save(os.path.join(result_vis_seg_base,f"{id}_combined_image.png"))
            
            material_to_color = {
                "sand":    (255, 127, 14),
                "soil":    (140, 86, 75),
                "metal":    (127, 127, 127),
                "jelly":   	(188, 189, 34),
                "wood":    (44, 160, 44),
                "plastic": 	(148, 103, 189),
                "liquid":  	(31, 119, 180),
                "paste":   	(214, 39, 40),
                "plush":   (227, 119, 194),
                "ceramic":   (23, 190, 207),
            }
            material_idx_mask = np.full(ss.shape, -1, dtype=int)
            for part_idx, mat_idx in enumerate(converted_list):
                if mat_idx != -1:
                    material_idx_mask[ss == part_idx] = mat_idx

            seg_rgba = np.zeros((ss.shape[0], ss.shape[1], 4), dtype=np.uint8)
            for idx, mat_name in enumerate(mat_names):
                rgb = material_to_color.get(mat_name, (0, 0, 0))
                seg_rgba[material_idx_mask == idx, :3] = rgb
                seg_rgba[material_idx_mask == idx, 3] = 255

            seg_rgba[material_idx_mask == -1, :3] = (0, 0, 0)
            seg_rgba[material_idx_mask == -1, 3] = 0

            seg_mask_path = os.path.join(result_vis_seg_base, f"{id}_mask.png")
            Image.fromarray(seg_rgba).save(seg_mask_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    args = parser.parse_args()

    base_path = args.dataset_path
    vis_seg_save_base = "./Results_" + base_path.split("/")[-1]
    vis_material_seg(base_path, vis_seg_save_base)





        