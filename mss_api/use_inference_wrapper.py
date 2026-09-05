import os
from types import SimpleNamespace  # Import SimpleNamespace to construct the configuration

# Import the distributed inference function
from inference_wrapper import AudioSeparator, run_distributed_inference


def main():
    # =======================================================
    # Configure paths shared by all examples
    # =======================================================
    model_type = "bdc_sg_bs_roformer"
    config_path = "ckpt/bdc_sg_bs_roformer/config.yaml"
    ckpt_path = "ckpt/bdc_sg_bs_roformer/model_bdc_sg_bs_roformer_ep_20_sisdr_9.5412.ckpt"

    # Output directory
    output_root = "results/api_inference"

    # =======================================================
    # 1. Single-process model initialization for examples 1-4
    #    Skip this initialization to save GPU memory when running only example 5.
    # =======================================================
    print("Loading the single-process model...")
    separator = AudioSeparator(
        model_type=model_type,
        config_path=config_path,
        model_path=ckpt_path,
        device_ids=[4],
    )
    print("Model loaded.")

    # =======================================================
    # Example 1: process a single file
    # =======================================================
    single_file_path = "test_sample/tmp/1.mp3"
    output_root_1 = "test_sample/output_single"
    if os.path.exists(single_file_path):
        separator.inference_file(single_file_path, output_root_1)

    # =======================================================
    # Example 2: process a directory
    # =======================================================
    # folder_path = "test_sample/tmp"
    # if os.path.exists(folder_path):
    #     separator.inference_folder(folder_path, output_root)

    # =======================================================
    # Example 3: return data without saving
    # =======================================================
    # print(f"\n--- [Example 3] Example of loading data into memory ---")
    # if os.path.exists(single_file_path):
    #     data_dict = separator.separate_audio(single_file_path)
    #     print(f"Returned stems: {list(data_dict.keys())}")
    #     if 'vocals' in data_dict:
    #         print(f"Vocal waveform shape (Channels, Samples): {data_dict['vocals'].shape}")
    # else:
    #     print("File missing; skipping example 3")


if __name__ == "__main__":
    main()
