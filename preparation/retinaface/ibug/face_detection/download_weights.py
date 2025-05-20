import os
import requests

# URLs for the pre-trained model weights (raw binary files)
WEIGHT_URLS = {
    "Resnet50_Final.pth": "https://raw.githubusercontent.com/biubug6/Pytorch_Retinaface/main/weights/Resnet50_Final.pth",
    "mobilenet0.25_Final.pth": "https://raw.githubusercontent.com/biubug6/Pytorch_Retinaface/main/weights/mobilenet0.25_Final.pth",
}

if __name__ == "__main__":
    # Determine the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "retina_face", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    for filename, url in WEIGHT_URLS.items():
        dest_path = os.path.join(weights_dir, filename)
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size = os.path.getsize(dest_path)
        print(f"Saved {filename} ({size//1024} KB) to {dest_path}")
    print("All weights downloaded successfully.") 