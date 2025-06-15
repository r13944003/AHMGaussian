import os
import argparse
from utils.vlm_utils import get_image_files, Qwen, GPT4V


def query_vlm(base_path, case_name, vlm_type = "qwen"):
    input_image_path = os.path.join(base_path, case_name, "gpt_input")
    image_files = get_image_files(input_image_path)

    material_list = "sand, soil, metal, jelly, wood, plastic, liquid, paste, plush, ceramic"
    material_list = material_list.split(", ")
    material_library = "{" + ", ".join(material_list) + "}"

    prompt = f"""Provided a picture. The left image is the original picture of the object (Original Image), and the middle image is a partial segmentation diagram (Mask Overlay), mask is in red. The right image is a partial of the object. 
    Based on the image, firstly provide a brief caption of the part. Secondly, describe what the part is made of (provide the major one). Finally, we combine what the object is and the material of the object to predict the hardness, density, Young's modulus and Poisson's Ratio of the part. Choose whether to use Shore A hardness or Shore D hardness depending on the material. You may provide a range of values for hardness instead of a single value.
    
    Format Requirement:
    You must provide your answer as a (brief caption of the part, material of the part, hardness, Shore A/D, density, Young's modulus and Poisson's Ratio) pair. Do not include any other text in your answer, as it will be parsed by a code script later.
    common material library: {material_library}. 
    Your answer must look like: caption, material, hardness low-high, <Shore A or Shore D>, density low-high kg/m^3, Young's modulus low-high MPa, Poisson's Ratio low-high.
    The material type must be chosen from the above common material library. Make sure to use Shore A or Shore D hardness, not Mohs hardness."""
    
    output_file = f'{case_name}.txt'
    results_file_path = os.path.join(base_path, case_name, output_file)
    case_msg = ""
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    with open(results_file_path, 'w') as file:
        for image_file in image_files:
            try:
                if vlm_type == 'qwen':
                    message = str(Qwen(image_file, prompt))
                else:
                    message = str(GPT4V(image_file, prompt))
            except KeyError as e:
                message = "error,-1"
            except Exception as e:
                message = "error,-1"
            write_msg = image_file + "," + message
            case_msg += case_msg
            file.write(f"{write_msg}\n")
            file.flush()

    print("Messages have been written to", results_file_path)


def run_vlm(base_path, vlm_type):
    all_cases = os.listdir(base_path)
    for case_name in all_cases:
        query_vlm(base_path, case_name, vlm_type=vlm_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('--vlm', type=str, default="qwen", help="gpt, qwen")
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    args = parser.parse_args()
    run_vlm(args.dataset_path, args.vlm)

