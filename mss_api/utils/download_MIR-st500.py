import json
import subprocess
import os
import argparse


def batch_download_youtube_audio(json_path, output_dir):
    """
    Download YouTube audio from a JSON manifest as numbered WAV files
    :param json_path: Path to MIR-ST500_link.json
    :param output_dir: Audio output directory
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read the JSON file
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            link_dict = json.load(f)
        print(f"✅ Loaded JSON with {len(link_dict)} audio links")
    except Exception as e:
        print(f"❌ Failed to read JSON：{e}")
        return

    # Download each audio link
    total = len(link_dict)
    success_count = 0
    fail_list = []

    for idx, (seq_num, url) in enumerate(link_dict.items(), 1):
        # Name output files by index, such as 1.wav, 10.wav, and 100.wav
        output_path = os.path.join(output_dir, f"{seq_num}.wav")

        # Skip files already downloaded
        if os.path.exists(output_path):
            print(f"[{idx}/{total}] ⏩ {seq_num}.wav Already exists; skipping")
            success_count += 1
            continue

        # Build the yt-dlp command
        cmd = [
            "yt-dlp",
            "--no-check-certificate",
            "--extractor-args",
            "youtube:skip=js",  # Keep JavaScript skipping enabled
            "-x",  # Extract audio only
            "--audio-format",
            "wav",  # Output WAV format
            "--audio-quality",
            "0",  # Highest audio quality to avoid compression loss
            "-o",
            output_path,  # Output path
            url,  # YouTube URL
        ]

        try:
            print(f"[{idx}/{total}] 📥 Downloading：{seq_num}.wav ({url})")
            # Run the download command, retaining errors and suppressing other output
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,  # Suppress standard output
                stderr=subprocess.PIPE,  # Capture error output
                check=True,
            )
            print(f"[{idx}/{total}] ✅ Download completed：{seq_num}.wav")
            success_count += 1
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode("utf-8", errors="ignore")[
                :200
            ]  # Truncate the error message
            print(f"[{idx}/{total}] ❌ Download failed：{seq_num}.wav | Error：{error_msg}")
            fail_list.append((seq_num, url, error_msg))
        except Exception as e:
            print(f"[{idx}/{total}] ❌ Download error：{seq_num}.wav | Error：{str(e)}")
            fail_list.append((seq_num, url, str(e)))

    # Print the download summary
    print("\n" + "=" * 50)
    print(f"📊 Download summary：")
    print(f"   Total count：{total}")
    print(f"   Success：{success_count}")
    print(f"   Failure：{len(fail_list)}")
    if fail_list:
        print(f"\n❌ Failed downloads ({len(fail_list)} total)：")
        for seq_num, url, err in fail_list[:10]:  # Show only the first 10 failures
            print(f"   {seq_num}.wav | {url} | Error：{err}")
        if len(fail_list) > 10:
            print(f"   ... {len(fail_list) - 10} additional failures omitted")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download MIR-ST500 YouTube audio in batches")
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to MIR-ST500_link.json, such as ./MIR-ST500_link.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./MIR-ST500_audio",
        help="Audio output directory (default: ./MIR-ST500_audio)",
    )
    args = parser.parse_args()

    # Run batch download
    batch_download_youtube_audio(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()
